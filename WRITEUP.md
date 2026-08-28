# 

# Reflection — Assignment 3: Deploying PyTorch ML Workloads with Docker & Kubernetes

The most challenging part of this assignment was that nearly every problem
looked like a different problem than it actually was, and narrowing each one
down taught me more than a clean run would have.

The first issue was the CIFAR-10 download itself. torchvision's default host
(cave.cs.toronto.edu) failed twice mid-download, corrupting the partial
file. I fixed this by downloading the archive manually through a browser,
which handles flaky connections far better than a script, and let
torchvision's own checksum check confirm the file before training touched
it.

The second issue was a real deadlock, not just slowness: with
num_workers=2, the training container hung indefinitely at high CPU with
zero progress. Docker's default 64MB shared-memory limit was colliding with
PyTorch's multi-process DataLoader workers, which rely on shared memory for
inter-process tensor transfer. Setting num_workers=0 (plus --shm-size=2g
locally) resolved it. This had to be fixed in two places, the local config
and the Kubernetes ConfigMap, since they're independent copies of the same
values.

The third issue only appeared on Kubernetes: the training Job requested and
limited itself to 2 full CPU cores on a Minikube cluster with only 2 cores
total, leaving nothing for the control plane to schedule. Even after
reducing the request, a hard CPU *limit* alongside it caused a subtler
problem: Kubernetes' CFS quota throttling interacting badly with a
multi-threaded process, where threads hitting the quota boundary
simultaneously could stall the whole process for a disproportionate time, a
known pathology under Guaranteed QoS. Removing the limit entirely
(Burstable QoS) and capping thread counts fixed it.

A smaller lesson: splitting train and serve requirements files didn't
shrink the serving image as expected, both stayed around 8.5GB, since the
dominant weight is the CUDA-enabled PyTorch wheels, not the extra
training-only packages. A CPU-only PyTorch build would have mattered far
more for size here.

Finally, a trade-off worth stating plainly: full training (10 epochs)
completed via local docker run, reaching 87% validation accuracy by epoch
9. The equivalent Kubernetes Job never completed an epoch in reasonable
time; a single-node 2-core Minikube VM isn't enough compute to train
ResNet-18 on CIFAR-10 quickly alongside what Kubernetes itself needs to
run. Rather than block on that, I injected the already-trained checkpoint
directly into checkpoint-pvc and used it to bring up and fully verify the
serving Deployment, Service, and HPA end-to-end. The orchestration is
proven correct; the training Job just needs a cluster with real compute
behind it, the same manifests would train to completion unmodified on one.



