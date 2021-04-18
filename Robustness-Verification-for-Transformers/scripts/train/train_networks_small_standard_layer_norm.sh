#!/usr/bin/env bash
cd ../..

for i in 3 6 12; do
	python3 main.py \
	      --train \
	      --display_interval 50 \
	      --data sst \
	      --batch_size 32 \
	      --base-dir bert-base-uncased \
	      --dir sst_bert_standard_layer_norm_$i --num_layers $i \
	      --dont_load_pretrained \
	      --num-epoches 5 \
	      --hidden_size 128 \
	      --intermediate_size 128 \
	      --layer_norm standard
done
