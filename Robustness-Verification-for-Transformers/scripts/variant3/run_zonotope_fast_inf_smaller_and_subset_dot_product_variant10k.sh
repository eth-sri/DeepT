#!/usr/bin/env bash
cd ../..

norm=100
echo "The norm is $norm (i.e. infinity)"

for nlayers in 3 6 12; do
    echo "Verifying network with $nlayers layers"

    python3 main.py --verify \
          --data sst \
          --dir sst_bert_smaller_$nlayers --num_layers $nlayers \
          --method zonotope \
          --p $norm \
          --empty_cache \
          --max_eps 0.32 \
          --max-num-error-terms 10000 --error-reduction-method box \
          --add-softmax-sum-constraint \
          --one-word-per-sentence \
          --use-dot-product-variant3
done