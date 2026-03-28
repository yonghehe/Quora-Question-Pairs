#!/bin/bash
#SBATCH --job-name=quora_embed
#SBATCH --output=embed_%j.log
#SBATCH --error=embed_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4
#SBATCH --gpus=1

cd ~/Quora-Question-Pairs

# uv creates and manages the venv automatically
uv run python embed_quora.py