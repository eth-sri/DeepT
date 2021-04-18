#!/usr/bin/env bash


# Activate conda
. "$HOME/miniconda3/etc/profile.d/conda.sh"
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier

# Table 2: Fast vs BaF (Yelp)
# --results-directory "results/normal_case"

CURRENT_DIR=$(realpath "$(dirname "$0")")


# Train small networks (3, 6, 12) layers on Yelp
cd "$CURRENT_DIR/../scripts/train" || exit
chmod +x train_networks_small_yelp_3.sh
bash train_networks_small_yelp_3.sh

cd "$CURRENT_DIR/../scripts/train" || exit
chmod +x train_networks_small_yelp_6.sh
bash train_networks_small_yelp_6.sh


cd "$CURRENT_DIR/../scripts/train" || exit
chmod +x train_networks_small_yelp_12.sh
bash train_networks_small_yelp_12.sh




