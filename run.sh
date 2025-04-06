#!/bin/bash

echo "Starting Training for Custom SRGAN..."
python scripts/train_srgan_custom.py --config configs/srgan_custom.yaml

echo "Evaluating All Models..."
python scripts/evaluate_all.py --models srgan_custom srgan_baseline srcnn esrgan bilinear

echo "Classifying Images via DRNet..."
python scripts/classify_images.py --input_dir experiments/ --output_dir classification_results/
