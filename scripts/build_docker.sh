#!/bin/bash

PARENT='arudl/my-stable-baselines3'
PARENT_VERSION='1.6.7'

TAG=arudl/my-rl-baselines3-zoo
LATEST=latest
VERSION=$(cat version.txt)  # version of RL Zoo

if [[ ${USE_GPU} == "True" ]]; then
  PARENT="${PARENT}:${PARENT_VERSION}"
else
  LATEST="${LATEST}-cpu"
  VERSION="${VERSION}-cpu"
  PARENT_VERSION="${PARENT_VERSION}-cpu"
  PARENT="${PARENT}:${PARENT_VERSION}"
fi

echo "docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg USE_GPU=${USE_GPU} --tag ${TAG}:${VERSION} . -f docker/Dockerfile"
docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg USE_GPU=${USE_GPU} --tag ${TAG}:${VERSION} . -f docker/Dockerfile
echo "docker tag ${TAG}:${VERSION} ${TAG}:${LATEST}"
docker tag ${TAG}:${VERSION} ${TAG}:${LATEST}

