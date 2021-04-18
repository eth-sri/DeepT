#!/usr/bin/env bash
cd ../..

nlayers=12
echo "Verifying network with $nlayers layers - Zonotope slow"

python3 main.py \
      --data sst \
      --dir sst_bert_small_$nlayers --num_layers $nlayers