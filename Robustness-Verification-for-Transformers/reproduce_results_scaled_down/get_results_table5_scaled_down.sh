#!/usr/bin/env bash

# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier


# Table 5: effect of the ordering of the norms in the fast version of the dot product transformer
# --results-directory "results/other_dot_product_results" \

CURRENT_DIR=$(realpath "$(dirname "$0")")

if [ ! -d "$CURRENT_DIR/../results/normal_case_scaled_down/" ]; then
  # Need the results for the normal case (with the softmax sum refinement)
  cd "$CURRENT_DIR" || exit
  bash ./get_results_table1_scaled_down.sh
fi

mkdir -p "$CURRENT_DIR/../results/other_dot_product_results_scaled_down/" || exit
cd "$CURRENT_DIR" || exit
cp "$CURRENT_DIR"/../results/normal_case_scaled_down/*zonotope_1_*WithConstraint* "$CURRENT_DIR/../results/other_dot_product_results_scaled_down/" || exit
cp "$CURRENT_DIR"/../results/normal_case_scaled_down/*zonotope_2_*WithConstraint* "$CURRENT_DIR/../results/other_dot_product_results_scaled_down/" || exit

# Get results for Fast (norms 1, 2, inf)
cd "$CURRENT_DIR/../scripts/dot_product_ordering" || exit
chmod +x run_zonotope_fast_1_other_dot_product_scaled_down.sh
bash run_zonotope_fast_1_other_dot_product_scaled_down.sh

cd "$CURRENT_DIR/../scripts/dot_product_ordering" || exit
chmod +x run_zonotope_fast_2_other_dot_product_scaled_down.sh
bash run_zonotope_fast_2_other_dot_product_scaled_down.sh
