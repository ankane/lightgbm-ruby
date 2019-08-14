#!/usr/bin/env bash

set -e

CACHE_DIR=$HOME/lightgbm/$LIGHTGBM_VERSION

if [ ! -d "$CACHE_DIR" ]; then
  wget https://github.com/microsoft/LightGBM/releases/download/v$LIGHTGBM_VERSION/lightgbm-$LIGHTGBM_VERSION.tar.gz
  tar xvfz lightgbm-$LIGHTGBM_VERSION.tar.gz
  mv lightgbm-$LIGHTGBM_VERSION $CACHE_DIR
  cd $CACHE_DIR
  mkdir build
  cd build
  cmake ..
  make -j4
else
  echo "LightGBM cached"
fi
