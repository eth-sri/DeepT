#!/usr/bin/env bash
cd ../..

norm=2
echo "The norm is $norm"

for nlayers in 3 6 12; do
    echo "Verifying network with $nlayers layers"

    python3 main.py --verify \
          --results-directory "results/normal_case" \
          --data sst \
          --dir sst_bert_small_$nlayers --num_layers $nlayers \
          --method baf \
          --p "$norm" \
          --empty_cache \
          --max_eps 0.04
done