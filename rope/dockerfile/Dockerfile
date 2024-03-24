ARG BASE_IMAGE=nvcr.io/nvidia/pytorch
ARG BASE_TAG=24.02-py3

FROM ${BASE_IMAGE}:${BASE_TAG} as base

RUN git clone https://github.com/lkm2835/triton_practice.git

RUN pip install git+https://github.com/NVIDIA/TransformerEngine.git@stable
