#!/usr/bin/env bash

# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier


# Table 4: effects of the softmax sum refinement
# --results-directory "results/no_constraint_results" \

CURRENT_DIR=$(realpath "$(dirname "$0")")


# Get results for Fast (norms 1, 2, inf) - with constraint
cd "$CURRENT_DIR/../scripts/no_softmax_constraint" || exit
chmod +x run_zonotope_fast_1_semi_scaled_down.sh
bash run_zonotope_fast_1_semi_scaled_down.sh

cd "$CURRENT_DIR/../scripts/no_softmax_constraint" || exit
chmod +x run_zonotope_fast_2_semi_scaled_down.sh.sh
bash run_zonotope_fast_2_semi_scaled_down.sh.sh

cd "$CURRENT_DIR/../scripts/no_softmax_constraint" || exit
chmod +x run_zonotope_fast_inf_semi_scaled_down.sh.sh
bash run_zonotope_fast_inf_semi_scaled_down.sh.sh


# Get results for Fast (norms 1, 2, inf) - without constraint
cd "$CURRENT_DIR/../scripts/no_softmax_constraint" || exit
chmod +x run_zonotope_fast_1_no_constraint_semi_scaled_down.sh
bash run_zonotope_fast_1_no_constraint_semi_scaled_down.sh

cd "$CURRENT_DIR/../scripts/no_softmax_constraint" || exit
chmod +x run_zonotope_fast_2_no_constraint_semi_scaled_down.sh
bash run_zonotope_fast_2_no_constraint_semi_scaled_down.sh

cd "$CURRENT_DIR/../scripts/no_softmax_constraint" || exit
chmod +x run_zonotope_fast_inf_no_constraint_semi_scaled_down.sh
bash run_zonotope_fast_inf_no_constraint_semi_scaled_down.sh


