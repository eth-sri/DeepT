#!/usr/bin/env bash


# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier

# Table 7: Fast vs BaF (Standard Layer Normalization)
# --results-directory "results/normal_case"

CURRENT_DIR=$(realpath "$(dirname "$0")")



# Get results for BaF (norms 1, 2, inf)
cd "$CURRENT_DIR/../scripts/baf" || exit
chmod +x run_baf_1_standard_layer_norm.sh
bash run_baf_1_standard_layer_norm.sh

cd "$CURRENT_DIR/../scripts/baf" || exit
chmod +x run_baf_2_standard_layer_norm.sh
bash run_baf_2_standard_layer_norm.sh


cd "$CURRENT_DIR/../scripts/baf" || exit
chmod +x run_baf_inf_standard_layer_norm.sh
bash run_baf_inf_standard_layer_norm.sh


# Get results for Fast (norms 1, 2, inf)
cd "$CURRENT_DIR/../scripts/fast" || exit
chmod +x run_zonotope_fast_1_standard_layer_norm.sh
bash run_zonotope_fast_1_standard_layer_norm.sh

cd "$CURRENT_DIR/../scripts/fast" || exit
chmod +x run_zonotope_fast_2_standard_layer_norm.sh
bash run_zonotope_fast_2_standard_layer_norm.sh


cd "$CURRENT_DIR/../scripts/fast" || exit
chmod +x run_zonotope_fast_inf_standard_layer_norm.sh
bash run_zonotope_fast_inf_standard_layer_norm.sh
