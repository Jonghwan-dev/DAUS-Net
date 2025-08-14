#!/usr/bin/env bash
set -euo pipefail

# Run from baseline/
export CKPT="train_output/final_run/best_model.pth"
export FILM_SCALE=0.7
export PRIOR_LAMBDA=0.5
export OUTPUT_DIR="submission_result/final"
export CLS_HEAD_VARIANT="linear"
export CLS_DROPOUT=0.0

python pred.py
cd submission_result/final
zip -r final_submission.zip segmentation/ classification.json