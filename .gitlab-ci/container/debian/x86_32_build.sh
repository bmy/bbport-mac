#!/usr/bin/env bash

arch=i386

. .gitlab-ci/container/cross_build.sh

. .gitlab-ci/container/build-rust.sh build

. .gitlab-ci/container/build-bindgen.sh
