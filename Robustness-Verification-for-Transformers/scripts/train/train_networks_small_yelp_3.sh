#!/usr/bin/env bash
cd ../..

for i in 3; do
	python3 main.py --train \
	  --display_interval 50 \
	  --data yelp \
	  --batch_size 32 \
	  --base-dir bert-base-uncased \
	  --dir yelp_bert_small_$i --num_layers $i \
	  --dont_load_pretrained \
	  --num-epoches 5 \
	  --hidden_size 128 \
	  --intermediate_size 128
done
