#!/usr/bin/env bash
cd ../..

norm=100
echo "The norm is $norm (e.g. inf)"

for nlayers in 6; do
    echo "Verifying smaller network with $nlayers layers - 3 fast layers"
    python3 main.py --verify \
          --data sst \
          --dir sst_bert_smaller_$nlayers --num_layers $nlayers \
          --method zonotope \
          --p $norm \
          --empty_cache \
          --max_eps 0.04 \
          --max-num-error-terms 7000 --max-num-error-terms-fast-layers 14000 --error-reduction-method box \
          --add-softmax-sum-constraint \
          --zonotope-slow \
          --one-word-per-sentence \
          --num-fast-dot-product-layers-due-to-switch 3 \
          --variant1plus2
done