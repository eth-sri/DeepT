#!/usr/bin/env bash

echo "Step 1: install miniconda (if not present)"
if [ ! -d ~/miniconda3 ]; then
  echo "Installing Miniconda"
  wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh || exit
  bash Miniconda3-latest-Linux-x86_64.sh -b -p ~/miniconda3 # -b is there to install miniconda silently
fi


echo "Step 2: activate conda and install the environment with the name 'py37_transformers_verifier'"
# shellcheck disable=SC1090

. "$HOME/miniconda3/etc/profile.d/conda.sh"
conda create -n py37_transformers_verifier python=3.7 -y
VIRTUAL_ENV_DISABLE_PROMPT=true conda activate py37_transformers_verifier

conda config --add channels pytorch
conda config --add channels conda-forge

conda install -y \
  pytorch-pretrained-bert=0.6.2 \
  termcolor=1.1.0 \
  torchaudio=0.7.0 \
  torchvision=0.8.1 \
  pytorch=1.7.0 \
  dataclasses=0.7 \
  pandas=1.1.5 \
  colorama=0.4.4 \
  tqdm=4.58.0 \
  scikit-learn=0.24.1 \
  nltk=3.4.4 \
  opt_einsum=3.3.0 \
  tensorflow=1.14.0 \
  matplotlib=3.3.4 \
  seaborn=0.11.1 \
  numpy=1.19.5 \
  notebook=6.2.0 \
  psutil=5.8.0 \
  cudatoolkit=10

echo "Step 3: install the 'punkt' tokenizer required by NTLK"
python3 -c "import nltk; nltk.download('punkt')" || exit

echo "Step 4: download Yelp and SST datasets"
wget "https://www.dropbox.com/s/bpzgu93py6j2tq0/data.tar.gz?dl=0" -O data.tar.gz
mkdir -p ../../data
tar -xvzf data.tar.gz -C ../../data
