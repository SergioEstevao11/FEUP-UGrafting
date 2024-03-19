#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# Step 1: Pretraining
echo "Starting pretraining phase..."
python main_pretrain.py --cosine --dataset cifar10 --lamda1 1 --lamda2 0.08 --epochs 800
echo "Pretraining completed."

# Optional: wait or check for pretraining completion
sleep 10s

# Step 2: Linear Evaluation
echo "Starting linear evaluation phase..."
python main_linear.py --dataset cifar10 --lamda1 1 --lamda2 0.08 --epochs 100 --ckpt "./cifar10_experiments/simclr_cifar10_0_epoch800_10heads_lamda11.0_lamda20.08.pt"
echo "Linear evaluation completed."

echo "All training phases completed successfully."
