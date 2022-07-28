#!/bin/bash

PARENT=arudl/my-stable-baselines3

TAG=arudl/my-rl-baselines3-zoo
LATEST=latest
VERSION=$(cat version.txt)  # version of RL Zoo

if [[ ${USE_GPU} == "True" ]]; then
  PARENT="${PARENT}:${LATEST}"
else
  LATEST="${LATEST}-cpu"
  PARENT="${PARENT}:${LATEST}"
  #TAG="${TAG}-cpu" # tag must always be the same since using only one repo
  # Mark the images as CPU via versions
  VERSION="${VERSION}-cpu"
fi

echo "docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg USE_GPU=${USE_GPU} --tag ${TAG}:${VERSION} . -f docker/Dockerfile"
docker build --build-arg PARENT_IMAGE=${PARENT} --build-arg USE_GPU=${USE_GPU} --tag ${TAG}:${VERSION} . -f docker/Dockerfile
echo "docker tag ${TAG}:${VERSION} ${TAG}:${LATEST}"
docker tag ${TAG}:${VERSION} ${TAG}:${LATEST}

