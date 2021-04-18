#!/usr/bin/env bash

# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier


# Table 2: Fast vs Precise vs Backward - Linf - small network, due to memory issues with Backward)
# --results-directory "results/smaller_network_results"

CURRENT_DIR=$(realpath "$(dirname "$0")")

# Get results for BaF (norm inf)
cd "$CURRENT_DIR/../scripts/backward" || exit
chmod +x run_backward_inf_smaller_and_subset_scaled_down.sh
bash run_backward_inf_smaller_and_subset_scaled_down.sh

# Get results for Fast (norm inf)
cd "$CURRENT_DIR/../scripts/fast" || exit
chmod +x run_zonotope_fast_inf_smaller_and_subset_scaled_down.sh
bash run_zonotope_fast_inf_smaller_and_subset_scaled_down.sh

# Get results for Precise (norm inf)
cd "$CURRENT_DIR/../scripts/precise" || exit
chmod +x run_zonotope_slow_inf_smaller_and_subset_scaled_down.sh
bash run_zonotope_slow_inf_smaller_and_subset_scaled_down.sh
