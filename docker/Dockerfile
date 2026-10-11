# Build environment for all versions (CPU, CUDA, MPI + CUDA).
# CUDA 12.4 is chosen to stay at or below the toolkit/driver available on Google Colab;
# see README.md ("Toolchain") for how this was checked.
#
# Build:  docker build -t lbm-d2q9 -f docker/Dockerfile .
# Run:    docker run --rm -it --gpus all -v "$PWD":/workspace lbm-d2q9
#
# The source tree is mounted at run time (not copied), so the image only holds the toolchain.
FROM nvidia/cuda:12.4.1-devel-ubuntu22.04

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
