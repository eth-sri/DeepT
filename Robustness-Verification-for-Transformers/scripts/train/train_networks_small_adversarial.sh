#!/usr/bin/env bash
cd ../..

for i in 12; do
	CUDA_VISIBLE_DEVICES=7 taskset -c 0-10 python3 main.py \
	  --train \
	  --train-adversarial \
	  --display_interval 50 \
	  --data sst \
	  --batch_size 128 \
	  --base-dir bert-base-uncased \
	  --dir sst_bert_smaller_adversarial_$i \
	  --num_layers $i \
	  --dont_load_pretrained \
	  --num-epoches 5 \
	  --hidden_size 64 \
	  --intermediate_size 128
done
