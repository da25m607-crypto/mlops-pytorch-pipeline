# Reflection — Assignment 3: Deploying PyTorch ML Workloads with Docker & Kubernetes

*(Fill in with your own experience once you've actually run the steps below —
this draft covers the parts of the pipeline most people find hardest, so you
can edit it into your own words after replicating the build/deploy locally.)*

The most challenging part of this assignment was getting the checkpoint
handed off cleanly between the training Job and the serving Deployment.
Kubernetes Jobs and Deployments are independent objects with no built-in
signal that says "the checkpoint is ready" — the natural fix is a shared
PersistentVolumeClaim mounted read-write by the training Job and read-only
by the serving pods, but that only works if the training Job actually
finishes (and writes to the PVC) before the Deployment is applied. In a
real pipeline this ordering would be enforced with an init container that
polls for the checkpoint file, or a CI/CD step that waits on `kubectl wait
--for=condition=complete job/model-training` before rolling out serving;
here it's called out explicitly in the README's apply order instead.

A second non-obvious issue was resizing the model for CIFAR-10's 32x32
images. Stock `torchvision.models.resnet18` assumes ImageNet-sized inputs
and downsamples aggressively in its stem (a 7x7 stride-2 conv followed by a
stride-2 max-pool), which collapses a 32x32 image down to roughly 1x1
before it ever reaches the residual blocks. The fix — replacing the first
conv with a 3x3 stride-1 version and swapping the max-pool for an identity
— is a standard adaptation for small images, but it's the kind of detail
that silently produces a model that "trains" while learning almost
nothing, rather than throwing an error.

Separating the training and serving images was conceptually simple but
easy to get sloppy on in practice: it's tempting to reuse one Dockerfile
and one requirements file for both, but that pulls training-only packages
(and their weight) into the serving image. Splitting `requirements/train.txt`
and `requirements/serve.txt`, and only copying the source files each
container actually needs, keeps the serving image meaningfully smaller and
reduces its dependency-vulnerability surface — which is also why it runs as
a non-root user with a `HEALTHCHECK` baked in rather than relying solely on
Kubernetes probes.

Finally, coordinating the ConfigMap-mounted config path with `train.py`'s
own config resolution logic took a couple of iterations. Making the script
check `--config`, then `$TRAINING_CONFIG`, then the well-known mounted
path, then a local fallback meant the exact same Docker image runs
unmodified locally, under `docker run`, and inside the Kubernetes Job —
which ended up being the single design choice that made local verification
actually predictive of cluster behavior, instead of the two diverging.
