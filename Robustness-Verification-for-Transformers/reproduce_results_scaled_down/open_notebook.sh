#!/usr/bin/env bash

# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier

CURRENT_DIR=$(realpath "$(dirname "$0")")

cd "$CURRENT_DIR/../results" || exit
jupyter notebook
