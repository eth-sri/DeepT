#!/usr/bin/env bash
cd ../..

echo "Certifying Visual Transformer with p=1"
python3 vit_certify.py \
  --method zonotope \
  --p 1 \
  --empty_cache \
  --max_eps 0.04 \
  --max-num-error-terms 14000 --error-reduction-method box \
  --add-softmax-sum-constraint




