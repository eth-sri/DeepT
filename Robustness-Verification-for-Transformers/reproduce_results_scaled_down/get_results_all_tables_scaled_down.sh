#!/usr/bin/env bash

CURRENT_DIR_HERE=$(realpath "$(dirname "$0")")

# Table 1: Fast vs BaF (normal)
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table1_scaled_down.sh

# Table 2: Fast vs Precise vs Backward - Linf - small network, due to memory issues with Backward)
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table2_scaled_down.sh

# Table 3: Fast vs Backward vs BaF - L1 and L2 - small network, due to memory issues with Backward)
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table3_scaled_down.sh

# Table 4: effects of the softmax sum refinement
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table4_scaled_down.sh

# Table 5: effect of the ordering of the norms in the fast version of the dot product transformer
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table5_scaled_down.sh

# Table 6: example of sentence verified with Fast against a synonym attack
# nothing to do
