#!/usr/bin/env bash
cd ../..

nlayers=6
echo "Verifying network with $nlayers layers - standard layer norm"

python3 main.py \
      --data sst \
      --dir sst_bert_standard_layer_norm_$nlayers --num_layers $nlayers