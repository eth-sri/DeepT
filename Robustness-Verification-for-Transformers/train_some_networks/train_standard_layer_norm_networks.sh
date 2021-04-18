#!/usr/bin/env bash

# Table 7: Fast vs BaF (Standard Layer Normalization)

# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier

CURRENT_DIR=$(realpath "$(dirname "$0")")

cd "$CURRENT_DIR/../scripts/train" || exit
chmod +x train_networks_small_standard_layer_norm.sh
bash train_networks_small_standard_layer_norm.sh





