#!/usr/bin/env bash
cd ../..

norm=100  # inf

for i in 3; do
	python3 -m pdb main.py \
	      --display_interval 50 \
	      --data sst \
	      --batch_size 32 \
	      --base-dir sst_bert_small_$i \
	      --dir sst_bert_small_diffai_$i --num_layers $i \
	      --num-epoches 5 \
	      --hidden_size 128 \
	      --intermediate_size 128 \
	      --diffai-eps 0.01 \
	      --diffai \
	      --p $norm \
        --empty_cache \
        --max_eps 0.04 \
        --max-num-error-terms 14000 \
        --error-reduction-method box \
#        --add-softmax-sum-constraint
	      #	      --dont_load_pretrained \
done
