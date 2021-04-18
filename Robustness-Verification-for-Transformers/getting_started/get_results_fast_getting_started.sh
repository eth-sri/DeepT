#!/usr/bin/env bash

# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier

CURRENT_DIR=$(realpath "$(dirname "$0")")

# Get results for Fast (2)
cd "$CURRENT_DIR/../scripts/fast" || exit
chmod +x run_zonotope_fast_2_few_samples.sh
bash run_zonotope_fast_2_few_samples.sh



