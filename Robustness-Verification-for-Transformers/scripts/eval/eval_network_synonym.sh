#!/usr/bin/env bash
cd ../..

LIRPA_CKPT="certifiably_trained_networks/bert_small3/ckpt_10"
echo "Evaluating synonym attack network on checkpoint '$LIRPA_CKPT'"

python3 main.py \
      --data sst \
      --lirpa-data \
      --lirpa-ckpt $LIRPA_CKPT \
      --num_layers 3 \
      --attack-type synonym