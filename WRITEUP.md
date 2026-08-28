# 

# Reflection — Assignment 3: Deploying PyTorch ML Workloads with Docker & Kubernetes

The most challenging part of this assignment was that nearly every problem I hit
looked, at first, like a different problem than it actually was — and the
process of narrowing each one down taught me more than a clean run would have.

The first real issue was the CIFAR-10 download itself. `torchvision`'s default
host (`cave.cs.toronto.edu`) failed twice mid-download, corrupting the partial
file both times. I fixed this by downloading the archive manually through a
browser (which handles flaky connections and resumption far better than a
script) and letting `torchvision`'s own checksum verification confirm the file
was valid before training ever touched it.

The second issue was a real deadlock, not just slowness: with
`num_workers=2` in the DataLoader config, the training container hung
indefinitely at high CPU with zero progress. The cause was Docker's default
64MB shared-memory limit colliding with PyTorch's multi-process DataLoader
workers, which rely on shared memory for inter-process tensor transfer.
Setting `num_workers=0` (and, for the standalone `docker run` case,
`--shm-size=2g`) resolved it immediately. This exact fix had to be applied in
two places — the local `configs/training_config.yaml` and the Kubernetes
ConfigMap — since they're independent copies of the same values, and I nearly
shipped the Kubernetes Job with the old, deadlocking setting still baked in.

The third issue only showed up once I moved to Kubernetes: the training Job
was requesting and limiting itself to 2 full CPU cores on a Minikube cluster
that only had 2 cores *total*, leaving nothing for the control plane (etcd,
kubelet, the API server) to actually schedule work. Reducing the request
helped, but even after that, setting a *hard* CPU limit alongside the request
caused a second, subtler problem: Kubernetes' CFS bandwidth quota throttling
interacting badly with a multi-threaded PyTorch process. When several threads
hit the quota boundary simultaneously, the whole process could stall for a
disproportionately long time waiting for the next scheduling window — a
well-documented but easy-to-miss pathology when running CPU-bound,
multi-threaded workloads under Kubernetes' `Guaranteed` QoS class. Removing
the CPU limit entirely (keeping only a request, i.e. `Burstable` QoS) and
reducing `OMP_NUM_THREADS`/`MKL_NUM_THREADS` to match the available budget
fixed it.

A smaller but genuinely useful lesson: separating `requirements/train.txt`
and `requirements/serve.txt` didn't shrink the serving image the way I
expected — both ended up around 8.5GB, because the dominant weight in either
image is the CUDA-enabled PyTorch/torchvision wheels themselves, not the
handful of extra training-only packages. On a CPU-only deployment target,
switching to the CPU-only PyTorch build (`--index-url https://download.pytorch.org/whl/cpu`) would have been the change that
actually mattered for image size — a good reminder to profile *what's* large
before assuming *why* it's large.

Finally, I made a deliberate trade-off worth stating plainly: full model
training (10 epochs) completed successfully via plain `docker run` on my
local machine, reaching 87% validation accuracy by epoch 9. The equivalent
Kubernetes training Job, however, never completed a full epoch in a
reasonable time — even after fixing the deadlock and the CPU throttling
issue, a single-node, 2-core Minikube VM genuinely isn't enough compute to
train a ResNet-18 on CIFAR-10 in a practical timeframe alongside everything
else Kubernetes itself needs to run. Rather than let that block the rest of
the assignment, I injected the already-trained checkpoint directly into the
`checkpoint-pvc` (the same `kubectl cp`-via-helper-pod technique used to work
around the flaky dataset download) and used it to bring up and fully verify
the serving Deployment, Service, and HPA. The orchestration is proven
correct — ConfigMap wiring, PVC persistence, resource limits, health probes,
and the full `Service → Deployment → pod → model` request path all verified
working end-to-end — even though the training Job itself is better suited to
a cluster with real compute behind it. On a properly resourced cluster (cloud
or otherwise), the same manifests, unmodified in their logic, would train to
completion; the constraint here was hardware, not the pipeline design.



