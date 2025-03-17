#!/bin/bash

# If linux check if cargo is installed
if [ "$(uname)" == "Linux" ]; then
  if ! command -v cargo &>/dev/null; then
    curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
  fi

  # Install dependencies
  # apt-get install build-essential
  # apt-get install pkg-config
  # apt-get install libudev-dev
fi
