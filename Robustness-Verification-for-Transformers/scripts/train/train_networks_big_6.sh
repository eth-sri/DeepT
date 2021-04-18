#!/usr/bin/env bash
cd ../..

for i in 6; do
	python3 main.py --train --display_interval 50 --data sst --batch_size 32 --base-dir bert-base-uncased --dir sst_bert_big_$i --num_layers $i --dont_load_pretrained --num-epoches 5 --hidden_size 256 --intermediate_size 512
done
