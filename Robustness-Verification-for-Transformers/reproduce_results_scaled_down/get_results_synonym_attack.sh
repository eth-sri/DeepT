#!/usr/bin/env bash

# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier


# Synonym attack results

CURRENT_DIR=$(realpath "$(dirname "$0")")


# Get results for Synonym attack (3 layer network)
cd "$CURRENT_DIR/../scripts/synonym" || exit
chmod +x run_synonym_attack_fast_pretrained1.sh
bash run_synonym_attack_fast_pretrained1.sh

# TODO: add script to process the output

