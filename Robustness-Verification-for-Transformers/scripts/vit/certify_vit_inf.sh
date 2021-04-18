#!/usr/bin/env bash
cd ../..

echo "Certifying Visual Transformer with p=inf"
python3 vit_certify.py \
  --method zonotope \
  --p 100 \
  --empty_cache \
  --max_eps 0.04 \
  --max-num-error-terms 14000 --error-reduction-method box \
  --add-softmax-sum-constraint




