#!/usr/bin/env bash

# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier


# Table 3: Fast vs Backward vs BaF - L1 and L2 - small network, due to memory issues with Backward)
# --results-directory "results/l1l2"

CURRENT_DIR=$(realpath "$(dirname "$0")")

# Get results for BaF (norm 1 and 2)
cd "$CURRENT_DIR/../scripts/baf" || exit
chmod +x run_baf_1_smaller_and_subset_scaled_down.sh
bash run_baf_1_smaller_and_subset_scaled_down.sh

cd "$CURRENT_DIR/../scripts/baf" || exit
chmod +x run_baf_2_smaller_and_subset_scaled_down.sh
bash run_baf_2_smaller_and_subset_scaled_down.sh



# Get results for Fast (norm 1 and 2)
cd "$CURRENT_DIR/../scripts/fast" || exit
chmod +x run_zonotope_fast_1_smaller_and_subset_scaled_down.sh
bash run_zonotope_fast_1_smaller_and_subset_scaled_down.sh

cd "$CURRENT_DIR/../scripts/fast" || exit
chmod +x run_zonotope_fast_2_smaller_and_subset_scaled_down.sh
bash run_zonotope_fast_2_smaller_and_subset_scaled_down.sh



# Get results for Backward (norm 1 and 2)
cd "$CURRENT_DIR/../scripts/backward" || exit
chmod +x run_backward_1_smaller_and_subset_scaled_down.sh
bash run_backward_1_smaller_and_subset_scaled_down.sh

cd "$CURRENT_DIR/../scripts/backward" || exit
chmod +x run_backward_2_smaller_and_subset_scaled_down.sh
bash run_backward_2_smaller_and_subset_scaled_down.sh
