#!/usr/bin/env bash
cd ../..

norm=100
echo "The norm is $norm (e.g. inf)"

for nlayers in 3 6; do
    echo "Verifying network with $nlayers layers"

    python3 main.py --verify \
          --results-directory "results/smaller_network_results_scaled_down" \
          --data sst \
          --dir sst_bert_smaller_$nlayers --num_layers $nlayers \
          --method zonotope \
          --p $norm \
          --empty_cache \
          --max_eps 0.04 \
          --max-num-error-terms 14000 --error-reduction-method box \
          --add-softmax-sum-constraint \
          --one-word-per-sentence \
          --start-sentence 10
done