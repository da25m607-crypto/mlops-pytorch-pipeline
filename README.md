# mlops-pytorch-pipeline

**MLOps \& Infrastructure for Machine Learning — Assignment 3**
*(Deploying PyTorch ML Workloads with Docker \& Kubernetes)*



*\*\*Repository:\*\* https://github.com/da25m607-crypto/mlops-pytorch-pipeline*



A production-style pipeline that takes a PyTorch image classifier from local
development to containerized training (Docker) to orchestrated training and
serving (Kubernetes).

## Architecture

```
                        ┌─────────────────────────┐
                        │   configs/training\_      │
                        │   config.yaml             │
                        └───────────┬──────────────┘
                                    │ mounted via ConfigMap
                                    ▼
┌───────────────┐   docker build   ┌───────────────────┐   kubectl apply   ┌─────────────────────┐
│ src/train.py   │ ───────────────▶│ mlops-train:v1      │ ──────────────▶ │  k8s Job             │
│ src/model.py   │  Dockerfile.train│ (training image)   │   training-job    │  (ml-training ns)    │
│ src/dataset.py │                 └───────────────────┘   .yaml            │  PVC: data + ckpts   │
└───────────────┘                                                          └──────────┬───────────┘
                                                                                        │ writes checkpoint
                                                                                        ▼
┌───────────────┐   docker build   ┌───────────────────┐   kubectl apply   ┌─────────────────────┐
│ src/serve.py   │ ───────────────▶│ mlops-serve:v1      │ ──────────────▶ │  Deployment (2 pods) │
│ src/model.py   │  Dockerfile.serve│ (serving image)    │  serving-\*.yaml   │  + Service (:80)     │
│ src/dataset.py │                 └───────────────────┘                   │  + HPA (2-5 pods)     │
└───────────────┘                                                          └──────────┬───────────┘
                                                                                        │
                                                                                        ▼
                                                                          curl POST /predict (via
                                                                          port-forward or LB)
```

Training and serving are deliberately separate images: the training image
carries the full PyTorch + torchvision + data-loading stack, while the
serving image installs only inference dependencies, runs as a non-root
user, and stays small.

## Repository layout

```
mlops-pytorch-pipeline/
├── README.md
├── .gitignore
├── .github/workflows/ci.yml
├── src/               # train.py, model.py, dataset.py, serve.py
├── configs/           # training\_config.yaml
├── docker/            # Dockerfile.train, Dockerfile.serve
├── k8s/               # namespace, configmap, pvc, training-job,
│                      # serving-deployment, serving-service, hpa,
│                      # secret.example.yaml
├── requirements/      # train.txt, serve.txt (pinned)
└── tests/             # test\_model.py
```

## Local setup

```bash
python -m venv .venv \&\& source .venv/bin/activate
pip install -r requirements/train.txt
pip install pytest ruff
pytest tests/ -v
```

## Docker: build and run locally

```bash
# Training
docker build -f docker/Dockerfile.train -t mlops-train:v1 .
docker run --rm \\
  -v $(pwd)/data:/app/data \\
  -v $(pwd)/checkpoints:/app/checkpoints \\
  mlops-train:v1

# Serving
docker build -f docker/Dockerfile.serve -t mlops-serve:v1 .
docker run --rm -p 8080:8080 \\
  -v $(pwd)/checkpoints:/app/checkpoints \\
  mlops-serve:v1

# Test the endpoint
curl -X POST http://localhost:8080/predict -F "image=@test\_image.png"
curl http://localhost:8080/health
```

> Capture terminal output/screenshots of the above for your PR description —
> the assignment asks for evidence these commands succeed locally.

## Kubernetes: deploy end-to-end

Requires a running cluster (Minikube, kind, or cloud-managed) and `kubectl`
pointed at it. If using Minikube/kind, load the locally built images into
the cluster first (`minikube image load mlops-train:v1`, etc.) since they
won't exist in a remote registry.

```bash
kubectl apply -f k8s/namespace.yaml
kubectl apply -f k8s/pvc.yaml
kubectl apply -f k8s/configmap.yaml
kubectl apply -f k8s/training-job.yaml

kubectl get pods -n ml-training -w        # wait for the Job pod to Complete
kubectl logs job/model-training -n ml-training

kubectl apply -f k8s/serving-deployment.yaml
kubectl apply -f k8s/serving-service.yaml
kubectl apply -f k8s/hpa.yaml   # requires metrics-server in the cluster

kubectl get pods -n ml-training
kubectl describe deployment model-serving -n ml-training

kubectl port-forward svc/model-serving 8080:80 -n ml-training
curl -X POST http://localhost:8080/predict -F "image=@test\_image.png"
```

Secrets (e.g. a model-registry token) are never committed as plaintext —
see `k8s/secret.example.yaml` for the expected keys and the imperative
`kubectl create secret` command to create the real one locally.

## Git workflow

This repo follows trunk-based feature branching:

1. `main` — protected, always deployable.
2. `develop` — integration branch, created from `main`.
3. Feature branches off `develop`: `feature/docker-training`,
`feature/k8s-deployment`, `feature/serving-api`, `feature/ci-pipeline`, etc.
4. Every feature branch is merged back via a Pull Request with a description
covering what changed and why, plus verification output/screenshots.
5. Commits follow [Conventional Commits](https://www.conventionalcommits.org/)
(`feat:`, `fix:`, `docs:`, `chore:`, `test:`).
6. Target: 2 merged PRs in Week 1 (repo scaffolding, model + Docker), 2 more
in Week 2 (Kubernetes manifests + end-to-end validation) — 4+ total.

Suggested PR breakdown:

|PR|Branch|Contents|
|-|-|-|
|1|`feature/repo-scaffolding`|Directory structure, `.gitignore`, README skeleton, CI workflow|
|2|`feature/pytorch-model`|`model.py`, `dataset.py`, `train.py`, `serve.py`, `tests/test\_model.py`|
|3|`feature/docker-training` + `feature/docker-serving`|Both Dockerfiles, local `docker run` verification screenshots|
|4|`feature/k8s-deployment`|All `k8s/\*.yaml`, end-to-end cluster verification screenshots|

## Configuration

All hyperparameters live in `configs/training\_config.yaml` and are mounted
into the training Job via the `training-config` ConfigMap — nothing is
hardcoded in `train.py`.

