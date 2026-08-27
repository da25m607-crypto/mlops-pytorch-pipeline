"""FastAPI model-serving application.

Loads a trained checkpoint on startup and exposes:
  GET  /health   -> 200 once the model is loaded (used by k8s liveness/readiness probes)
  POST /predict  -> multipart image upload, returns class probabilities

Run locally:
    uvicorn serve:app --host 0.0.0.0 --port 8080

The checkpoint path, dataset, and architecture are all resolved from
environment variables so the same image works whether the checkpoint was
produced locally or via the Kubernetes training Job (mounted PVC).
"""
from __future__ import annotations

import io
import os
from pathlib import Path
from typing import Optional

import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile
from PIL import Image

from dataset import CLASS_NAMES, get_inference_transform
from model import get_model

CHECKPOINT_PATH = os.environ.get("CHECKPOINT_PATH", "/app/checkpoints/classifier_v1.pt")

app = FastAPI(title="Image Classifier Serving API")

_state = {"model": None, "transform": None, "class_names": None, "device": None}


def load_model() -> None:
    checkpoint_path = Path(CHECKPOINT_PATH)
    if not checkpoint_path.exists():
        # Leave the model unset; /health will report not-ready rather than
        # crashing the process, so the pod stays up while checkpoints land.
        return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(checkpoint_path, map_location=device)

    architecture = checkpoint.get("architecture", "resnet18")
    num_classes = checkpoint.get("num_classes", 10)
    dataset = checkpoint.get("dataset", "cifar10")

    model = get_model(architecture=architecture, num_classes=num_classes)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

    _state["model"] = model
    _state["transform"] = get_inference_transform(dataset)
    _state["class_names"] = CLASS_NAMES.get(dataset, [str(i) for i in range(num_classes)])
    _state["device"] = device


@app.on_event("startup")
def on_startup() -> None:
    load_model()


@app.get("/health")
def health():
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")
    return {"status": "ok"}


@app.post("/predict")
async def predict(image: UploadFile = File(...)):
    if _state["model"] is None:
        raise HTTPException(status_code=503, detail="Model not loaded")

    try:
        contents = await image.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception as exc:  # noqa: BLE001
        raise HTTPException(status_code=400, detail=f"Invalid image: {exc}") from exc

    tensor = _state["transform"](img).unsqueeze(0).to(_state["device"])

    with torch.no_grad():
        logits = _state["model"](tensor)
        probs = F.softmax(logits, dim=1).squeeze(0).tolist()

    class_names = _state["class_names"]
    results = sorted(
        [{"class": class_names[i], "probability": round(p, 4)} for i, p in enumerate(probs)],
        key=lambda x: x["probability"],
        reverse=True,
    )
    return {"predictions": results, "top_class": results[0]["class"]}
