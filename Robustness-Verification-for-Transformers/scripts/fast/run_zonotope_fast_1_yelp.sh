#!/usr/bin/env bash
cd ../..

norm=1
echo "The norm is $norm"

for nlayers in 3 6 12; do
    echo "Verifying network with $nlayers layers"

    python3 main.py --verify \
          --results-directory "results/yelp" \
          --data yelp \
          --dir yelp_bert_small_$nlayers --num_layers $nlayers \
          --method zonotope \
          --p $norm \
          --empty_cache \
          --max_eps 0.04 \
          --max-num-error-terms 14000 --error-reduction-method box \
          --add-softmax-sum-constraint
done