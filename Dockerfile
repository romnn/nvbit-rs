FROM nvidia/cuda:11.8.0-devel-ubuntu20.04

# install and update basic dependencies
# ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update && \
  apt-get upgrade -y && \
  DEBIAN_FRONTEND=noninteractive apt-get install -y --no-install-recommends tzdata && \
  apt-get install -y curl build-essential libssl-dev pkg-config libclang-dev

# install rust build toolchains
RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
ENV PATH="/root/.cargo/bin:${PATH}"

ENV CARGO_HOME /cache
WORKDIR /src
