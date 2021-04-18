#!/usr/bin/env bash
cd ../..

for i in 2; do
  echo "Training 2 layer word network"
	CUDA_VISIBLE_DEVICES=7 taskset -c 6-11 python3 main.py --train --display_interval 50 --data sst --batch_size 32 --base-dir bert-custom-word-vocab-uncased --dir sst_bert_word_small_$i --num_layers $i --dont_load_pretrained --num-epoches 5 --hidden_size 64 --intermediate_size 128 --min_word_freq 2
done
