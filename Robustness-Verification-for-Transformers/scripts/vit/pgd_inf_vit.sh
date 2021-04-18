#!/usr/bin/env bash
cd ../..

echo "Running PGD with p=inf on Visual Transformer"
python3 vit_certify.py --pgd

