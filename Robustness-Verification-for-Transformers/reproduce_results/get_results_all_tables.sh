#!/usr/bin/env bash

CURRENT_DIR_HERE=$(realpath "$(dirname "$0")")

# Table 1: Fast vs BaF (normal)
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table1.sh

# Table 2: Fast vs BaF (Yelp)
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table2.sh

# Table 3: Fast vs BaF (Big)
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table3.sh

# Table 4: Fast vs Precise vs Backward - Linf - small network, due to memory issues with Backward)
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table4.sh

# Table 5: Fast vs Backward vs BaF - L1 and L2 - small network, due to memory issues with Backward)cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table5.sh

# Table 6: effect of the ordering of the norms in the fast version of the dot product transformer
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table6.sh

# Table 7: Fast vs BaF (Standard Layer Normalization)
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table7.sh

# Table 13: effects of the softmax sum refinement
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_table13.sh

# Table 8: synonym attack results
cd "$CURRENT_DIR_HERE" || exit
bash ./get_results_synonym_attack.sh
