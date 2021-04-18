#!/usr/bin/env bash
cd ../..

nlayers=3
echo "Verifying network with $nlayers layers - Zonotope slow"

python3 main.py \
      --data yelp \
      --dir yelp_bert_small_$nlayers --num_layers $nlayers