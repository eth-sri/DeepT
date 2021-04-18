#!/usr/bin/env bash
cd ../..

norm=100
echo "The norm is $norm (e.g. inf)"

for nlayers in 3 6 12; do
    echo "Verifying network with $nlayers layers - Backward"

    python3 main.py --verify \
          --results-directory "results/smaller_network_results" \
          --data sst \
          --dir sst_bert_smaller_$nlayers --num_layers $nlayers \
          --method backward \
          --p "$norm" \
          --empty_cache \
          --max_eps 0.04 \
          --one-word-per-sentence
done