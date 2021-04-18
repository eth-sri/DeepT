#!/usr/bin/env bash
cd ../..

# --verify --data sst --lirpa-ckpt certifiably_trained_networks/bert_small3/ckpt_1 --num_layers 3 --attack-type synonym --method zonotope --p 100 --empty_cache --max_eps 0.04 --max-num-error-terms 7000 --error-reduction-method box --add-softmax-sum-constraint --one-word-per-sentence

LIRPA_CKPT="certifiably_trained_networks/bert_smaller5/ckpt_10"
echo "Doing synonym attacks on checkpoint '$LIRPA_CKPT'"

python3 main.py --verify \
  --data sst \
  --lirpa-ckpt $LIRPA_CKPT \
  --num_layers 5 \
  --attack-type synonym \
  --method baf \
  --p 100 \
  --empty_cache \
  --max_eps 0.04 \
  --one-word-per-sentence
