#!/usr/bin/env bash


# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier

# Table 3: Fast vs BaF (Big)
# --results-directory "results/normal_case"

CURRENT_DIR=$(realpath "$(dirname "$0")")



# Get results for BaF (norms 1, 2, inf)
cd "$CURRENT_DIR/../scripts/baf" || exit
chmod +x run_baf_1_big.sh
bash run_baf_1_big.sh

cd "$CURRENT_DIR/../scripts/baf" || exit
chmod +x run_baf_2_big.sh
bash run_baf_2_big.sh


cd "$CURRENT_DIR/../scripts/baf" || exit
chmod +x run_baf_inf_big.sh
bash run_baf_inf_big.sh


# Get results for Fast (norms 1, 2, inf)
cd "$CURRENT_DIR/../scripts/fast" || exit
chmod +x run_zonotope_fast_1_big.sh
bash run_zonotope_fast_1_big.sh

cd "$CURRENT_DIR/../scripts/fast" || exit
chmod +x run_zonotope_fast_2_big.sh
bash run_zonotope_fast_2_big.sh


cd "$CURRENT_DIR/../scripts/fast" || exit
chmod +x run_zonotope_fast_inf_big.sh
bash run_zonotope_fast_inf_big.sh




