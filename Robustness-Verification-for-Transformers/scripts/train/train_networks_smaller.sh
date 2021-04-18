#!/usr/bin/env bash
cd ../..

for i in 3 6 12; do
	CUDA_VISIBLE_DEVICES=0 taskset -c 6-11 python3 main.py --train --display_interval 50 --data sst --batch_size 32 --base-dir bert-base-uncased --dir sst_bert_smaller_$i --num_layers $i --dont_load_pretrained --num-epoches 5 --hidden_size 64 --intermediate_size 128
done
