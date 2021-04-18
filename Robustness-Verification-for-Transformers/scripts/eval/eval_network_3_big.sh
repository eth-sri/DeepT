#!/usr/bin/env bash
cd ../..

nlayers=3
echo "Evaluating big network with $nlayers layers"

python3 main.py \
      --data sst \
      --dir sst_bert_big_$nlayers --num_layers $nlayers