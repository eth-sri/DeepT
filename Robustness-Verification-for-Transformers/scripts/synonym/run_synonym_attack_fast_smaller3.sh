#!/usr/bin/env bash
cd ../..

LIRPA_CKPT="certifiably_trained_networks/bert_smaller3/ckpt_5"
echo "Doing synonym attacks on checkpoint '$LIRPA_CKPT'"

python3 main.py --verify \
      --data sst \
      --lirpa-ckpt $LIRPA_CKPT \
      --num_layers 3 \
      --attack-type synonym \
      --method zonotope \
      --p 100 \
      --empty_cache \
      --max_eps 0.04 \
      --max-num-error-terms 14000 \
      --error-reduction-method box \
      --add-softmax-sum-constraint \
      --one-word-per-sentence \
      --batch-softmax-computation