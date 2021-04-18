#!/usr/bin/env bash
cd ../..

LIRPA_CKPT="adversarially_trained_networks/bert_small_3_adv/ckpt_10"
echo "Doing synonym attacks on checkpoint '$LIRPA_CKPT'"

python3 main.py --verify \
      --data sst \
      --lirpa-ckpt $LIRPA_CKPT \
      --num_layers 3 \
      --attack-type synonym \
      --method backward \
      --p 100 \
      --empty_cache \
      --max_eps 0.04 \
      --add-softmax-sum-constraint \
      --one-word-per-sentence