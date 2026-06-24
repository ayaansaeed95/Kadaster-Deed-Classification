#!/bin/bash
set -e

if [[ "$(uname -s)" == "Darwin" ]]; then
  pip install torch
else
  pip install torch --index-url https://download.pytorch.org/whl/cu124
fi

pip install -r requirements.txt
