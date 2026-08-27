"""Training entrypoint for the containerized/Kubernetes training job.

Reads all hyperparameters from a YAML config (mounted via ConfigMap in
Kubernetes, or passed as a local file path), trains the selected model,
logs structured JSON-lines metrics to stdout, and checkpoints the best
model by validation loss with early stopping.
"""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch
import torch.nn as nn
import yaml

from dataset import get_dataloaders
from model import get_model


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def resolve_config_path(cli_path: str | None) -> Path:
    """Config resolution order: CLI arg > $TRAINING_CONFIG env var >
    mounted ConfigMap path > local repo path. This lets the same image run
    unchanged locally, in `docker run`, and in a Kubernetes Job.
    """
    candidates = [
        cli_path,
        os.environ.get("TRAINING_CONFIG"),
        "/app/configs/training_config.yaml",
        "configs/training_config.yaml",
    ]
    for candidate in candidates:
        if candidate and Path(candidate).exists():
            return Path(candidate)
    raise FileNotFoundError(
        "No training config found. Checked --config, $TRAINING_CONFIG, "
        "/app/configs/training_config.yaml, configs/training_config.yaml"
    )


def train_one_epoch(model, loader, optimizer, criterion, device) -> tuple[float, float]:
    model.train()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device) -> tuple[float, float]:
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    for inputs, targets in loader:
        inputs, targets = inputs.to(device), targets.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        total_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()
    return total_loss / total, correct / total


def log(entry: dict) -> None:
    print(json.dumps(entry), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default=None, help="Path to training_config.yaml")
    args = parser.parse_args()

    config_path = resolve_config_path(args.config)
    config = load_config(str(config_path))
    log({"event": "config_loaded", "path": str(config_path)})

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    log({"event": "device_selected", "device": str(device)})

    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
    ).to(device)

    train_loader, val_loader = get_dataloaders(
        data_dir=config["data"]["data_dir"],
        dataset=config["data"].get("dataset", "cifar10"),
        batch_size=config["training"]["batch_size"],
        num_workers=config["training"].get("num_workers", 2),
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=config["training"]["learning_rate"])
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    patience_counter = 0
    patience = config["training"]["early_stopping_patience"]

    checkpoint_dir = Path(config["output"]["checkpoint_dir"])
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    save_path = checkpoint_dir / config["output"]["model_name"]

    for epoch in range(config["training"]["epochs"]):
        train_loss, train_acc = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        log({
            "epoch": epoch + 1,
            "train_loss": round(train_loss, 4),
            "train_accuracy": round(train_acc, 4),
            "val_loss": round(val_loss, 4),
            "val_accuracy": round(val_acc, 4),
        })

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            torch.save({
                "epoch": epoch + 1,
                "model_state_dict": model.state_dict(),
                "architecture": config["model"]["architecture"],
                "num_classes": config["model"]["num_classes"],
                "dataset": config["data"].get("dataset", "cifar10"),
                "val_loss": val_loss,
                "val_accuracy": val_acc,
            }, save_path)
            log({"event": "checkpoint_saved", "path": str(save_path)})
        else:
            patience_counter += 1
            if patience_counter >= patience:
                log({"event": "early_stopping", "epoch": epoch + 1})
                break

    log({"event": "training_complete", "best_val_loss": round(best_val_loss, 4)})


if __name__ == "__main__":
    main()
