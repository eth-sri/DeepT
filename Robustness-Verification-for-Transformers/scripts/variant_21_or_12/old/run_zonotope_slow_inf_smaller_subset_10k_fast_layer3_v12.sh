#!/usr/bin/env bash
cd ../..

norm=100
echo "The norm is $norm (e.g. inf)"

for nlayers in 6 12; do
    startFast=3

    echo "Verifying network with $nlayers layers - starting fast at layer $startFast"

    python3 main.py --verify \
          --data sst \
          --dir sst_bert_smaller_$nlayers --num_layers $nlayers \
          --method zonotope \
          --p $norm \
          --empty_cache \
          --max_eps 0.04 \
          --max-num-error-terms 10000 --error-reduction-method box \
          --add-softmax-sum-constraint \
          --zonotope-slow \
          --one-word-per-sentence \
          --fast-dot-product-switch-at-layer $startFast \
          --variant1plus2
done