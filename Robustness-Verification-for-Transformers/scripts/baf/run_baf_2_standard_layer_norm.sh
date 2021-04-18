#!/usr/bin/env bash
cd ../..

norm=2
echo "The norm is $norm"

for nlayers in 3 6 12; do
    echo "Verifying network with $nlayers layers"

    python3 main.py --verify \
          --results-directory "results/standard_layer_norm" \
          --data sst \
          --dir sst_bert_standard_layer_norm_$nlayers --num_layers $nlayers \
          --method baf \
          --p "$norm" \
          --empty_cache \
          --max_eps 0.04 \
	        --layer_norm standard
done