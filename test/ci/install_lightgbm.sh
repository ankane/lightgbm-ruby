#!/usr/bin/env bash

set -e

CACHE_DIR=$HOME/lightgbm/$LIGHTGBM_VERSION

if [ ! -d "$CACHE_DIR" ]; then
  git clone --recursive https://github.com/microsoft/LightGBM
  mv LightGBM $CACHE_DIR
  cd $CACHE_DIR
  mkdir build
  cd build
  cmake ..
  make -j4
else
  echo "LightGBM cached"
fi
