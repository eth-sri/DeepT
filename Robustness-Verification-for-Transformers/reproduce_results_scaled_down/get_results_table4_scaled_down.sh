#!/usr/bin/env bash

# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier


# Table 4: effects of the softmax sum refinement
# --results-directory "results/no_constraint_results" \

CURRENT_DIR=$(realpath "$(dirname "$0")")

if [ ! -d "$CURRENT_DIR/../results/normal_case_scaled_down/" ]; then
  # Need the results for the normal case (with the softmax sum refinement)
  cd "$CURRENT_DIR" || exit
  bash ./get_results_table1_scaled_down.sh
fi

mkdir -p "$CURRENT_DIR/../results/no_constraint_results_scaled_down/" || exit
cd "$CURRENT_DIR" || exit
cp "$CURRENT_DIR"/../results/normal_case_scaled_down/*zonotope*WithConstraint* "$CURRENT_DIR/../results/no_constraint_results_scaled_down/" || exit

# Get results for Fast (norms 1, 2, inf)
cd "$CURRENT_DIR/../scripts/no_softmax_constraint" || exit
chmod +x run_zonotope_fast_1_no_constraint_scaled_down.sh
bash run_zonotope_fast_1_no_constraint_scaled_down.sh

cd "$CURRENT_DIR/../scripts/no_softmax_constraint" || exit
chmod +x run_zonotope_fast_2_no_constraint_scaled_down.sh
bash run_zonotope_fast_2_no_constraint_scaled_down.sh

cd "$CURRENT_DIR/../scripts/no_softmax_constraint" || exit
chmod +x run_zonotope_fast_inf_no_constraint_scaled_down.sh
bash run_zonotope_fast_inf_no_constraint_scaled_down.sh


