# Build environment for all versions (CPU, CUDA, MPI + CUDA).
# CUDA 13.0: same release as Google Colab (nvcc 13.0.88, driver 580.82.07, checked on a
# Tesla T4 runtime); see README.md ("Toolchain").
#
# Build:  docker build -t lbm-d2q9 -f docker/Dockerfile .
# Run:    docker run --rm -it --gpus all -v "$PWD":/workspace lbm-d2q9
#
# The source tree is mounted at run time (not copied), so the image only holds the toolchain.
FROM nvidia/cuda:13.0.2-devel-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
 && apt-get install -y --no-install-recommends \
      cmake \
      g++ \
      make \
      git \
      openmpi-bin \
      libopenmpi-dev \
      python3 \
      python3-numpy \
      python3-matplotlib \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace
CMD ["bash"]
