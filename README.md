DAUS-Net: A Dynamically Adaptable Framework for Multi-task Ultrasound Analysis
==============================================================================

> Participating in the UUSIC25: Universal Ultrasound Image Challenge (MICCAI 2025 Deep-Breath Workshop | https://uusic2025.github.io/).  
> Official baseline: | Evaluation via Codabench (see baseline repo| https://github.com/uusic2025/challenge)  
  
Overview
--------
DAUS-Net is a hybrid CNN–Transformer framework based on TransUNet for joint ultrasound tumor segmentation and classification. It introduces a Gated Low-Rank Adaptation (LoRA) mechanism to dynamically modulate reasoning pathways using positional and contextual prompts. For classification, DAUS-Net fuses Feature-wise Linear Modulation (FiLM) with prior logits to synergize visual evidence and contextual knowledge. The approach consistently improves upon the UniUSNet baseline.

Key highlights:
- Hybrid CNN + Transformer backbone to capture both local and global features.
- Gated-LoRA for dynamic, prompt-driven adaptation.
- FiLM + prior-logits fusion for robust classification.
- Multi-task training for segmentation and classification with carefully balanced losses.

Abstract (from the paper)
-------------------------
Ultrasound imaging is widely used in the clinic due to its accessibility and cost-effectiveness. However, ROI variability due to probe positioning complicates joint classification and segmentation. We propose DAUS-Net, a dynamically adaptable hybrid CNN–Transformer architecture that leverages a Gated-Low-Rank Adaptation (LoRA) module to modulate reasoning based on positional and contextual prompts. For classification, FiLM-based fusion with prior logits augments visual features with explicit context. DAUS-Net surpasses the UniUSNet baseline, suggesting that dynamically integrating robust visual features with explicit clinical context is a promising direction for medical AI.

Repository structure
--------------------
- `model.py`: Default entry for in-container inference (Docker CMD).
- `train.py`, `trainer.py`: Training entrypoint and training utilities.
- `pred.py`: Inference and result generation (segmentation masks + classification JSON).
- `z_train.sh`: Single-GPU training helper (Torchrun-ready). Default output dir: `train_output/final_run`.
- `z_create_submission.sh`: Packaging helper to generate `final_submission.zip`.
- `datasets/`: Dataset loader(s), e.g., `datasets/omni_dataset.py`.
- `networks/`: Model definitions and components.
- `train_checkpoint/`: Checkpoints directory (also used by Docker example).
- `data/`: Example data directory (see inference section for expected inputs).
- `util/utils.py`: Utility functions.
- `Dockerfile`: Reproducible container with pinned base image.

UUSIC25 Challenge (Essential Links)
-----------------------------------
- Official baseline repository: [Challenge](https://github.com/uusic2025/challenge)
- Workshop: Deep-Breath 2025 (MICCAI 2025 satellite) — announcement of winners during the workshop in Daejeon, South Korea.
- Evaluation: Codabench automated submission/evaluation with live leaderboards (see links from the baseline repository).

Requirements and environment
----------------------------
- Python dependencies are listed in `requirements.txt`.
- Torch environment is provided via the base image: `pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`.

Quickstart (Docker, recommended)
--------------------------------
For most users, the fastest way to run inference is via the pre-built Docker image.

1) Pull the image

```bash
docker pull armyjh/ncck_bc:latest
```

2) Prepare paths and run

```bash
mkdir -p output
docker run --rm --gpus all \
  -v "/path/to/your/test_data/data":/input:ro \
  -v "/path/to/your/test_data/data.json":/input_json:ro \
  -v "$PWD/output":/output \
  armyjh/ncck_bc:latest
```

Notes:
- `/input_json` must be a single JSON file (file mount), not a directory.
- The model checkpoint is already inside the image; no extra mount is needed.

Local setup (without Docker)
----------------------------
1) Create a virtual environment and install deps:

```bash
python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```

2) Verify GPU availability (optional):

```python
python - << 'PY'
import torch
print('CUDA available:', torch.cuda.is_available())
PY
```

Docker usage
------------
The repository includes a Dockerfile that encapsulates runtime, dependencies, and default paths.

Base image (fixed by competition rules):
- `FROM pytorch/pytorch:1.12.1-cuda11.3-cudnn8-runtime`

Environment variables set in the image:
- `INPUT_DIR=/input`
- `OUTPUT_DIR=/output`
- `JSON_PATH=/input_json`
- `CKPT=/weights/best_model.pth`

Run the default entry with explicit envs (executes `python model.py`):

```bash
docker run --rm --gpus all \
  -e INPUT_DIR=/input \
  -e OUTPUT_DIR=/output \
  -e JSON_PATH=/input_json \
  -e CKPT=/weights/best_model.pth \
  armyjh/ncck_bc:latest
```

Note:
- The Dockerfile demonstrates copying sample data and weights into the image (which increases image size). In practice, you may bind-mount your data and weights at runtime instead.
- `/input_json` must be a single JSON file (file mount), not a directory.

```bash
docker run --rm \
  -v "$PWD/data/Val":/input:ro \
  -v "$PWD/train_checkpoint/best_model.pth":/weights/best_model.pth:ro \
  -v "$PWD/data/private_val_for_participants.json":/input_json:ro \
  -v "$PWD/output":/output \
  armyjh/ncck_bc:latest
```

Training
--------
We provide `z_train.sh` as a convenient wrapper around `train.py` using `torchrun`.

Defaults (see `z_train.sh`):
- `OUT=${OUT:-train_output/final_run}` → checkpoints and logs are saved under this directory. The best model is saved as `best_model.pth`.
- Typical hyperparameters are exposed via environment variables (override as needed):
  - Data and I/O: `ROOT=data`, `OUT=train_output/final_run`.
  - Optimization: `BASE_LR`, `MAX_EPOCHS` (default 200), `PATIENCE` (early stop), `PLATEAU_PATIENCE`, warmup/min lr ratio.
  - Batch/size: `IMG` (default 224), `BS` (default 32).
  - AMP/precision: `AMP_ON`, `AMP_DTYPE`.
  - LoRA: `LORA_RANK`, `LORA_ALPHA`, `LORA_DROPOUT`, `LORA_ONLY`.
  - FiLM/prior fusion: `FILM_SCALE` (default 0.7), `PRIOR_LAMBDA` (default 0.5).
  - Classification head: `CLS_HEAD_VARIANT` (default `linear`), `CLS_DROPOUT` (default 0.0).
  - Regularization/misc: `CLIP_NORM`, EMA options, class weights, OHEM, etc.
  - Reproducibility: `SEED` (default 42), `DETERMINISTIC` (default 1).
  - W&B logging: set `WANDB_OFF=1` to disable (default), or provide `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_RUN`.

Single-GPU example:

```bash
# Using defaults (outputs to train_output/final_run)
bash z_train.sh

# Override output directory and some knobs
OUT=train_output/exp1 IMG=256 BS=24 BASE_LR=5e-4 MAX_EPOCHS=150 \
FILM_SCALE=0.7 PRIOR_LAMBDA=0.5 \
WANDB_OFF=0 WANDB_PROJECT=uusic25_tm \
bash z_train.sh
```

Multi-GPU example (set `NPROC`):

```bash
NPROC=2 OUT=train_output/exp_ddp bash z_train.sh
```

Inference and submission
------------------------
We provide `z_create_submission.sh` to reproduce the expected submission artifacts:

```bash
bash z_create_submission.sh
```

This script sets:
- `CKPT="train_output/final_run/best_model.pth"`
- `FILM_SCALE=0.7`, `PRIOR_LAMBDA=0.5`
- `OUTPUT_DIR="submission_result/final"`
- `CLS_HEAD_VARIANT="linear"`, `CLS_DROPOUT=0.0`

It then runs `python pred.py`, switches to `submission_result/final`, and zips:
- `segmentation/` directory (predicted masks)
- `classification.json` (predicted labels)

Final archive: `submission_result/final/final_submission.zip`.

Manual inference (without the helper script):

```bash
export CKPT="train_output/final_run/best_model.pth"
export FILM_SCALE=0.7
export PRIOR_LAMBDA=0.5
export OUTPUT_DIR="submission_result/manual"
export CLS_HEAD_VARIANT="linear"
export CLS_DROPOUT=0.0

python pred.py
```

Data layout
-----------
The training and inference pipelines expect an ultrasound dataset with both segmentation targets and classification labels. Please refer to `datasets/omni_dataset.py` for the exact on-disk structure and JSON metadata expectations. By default, `z_train.sh` uses `ROOT=data`. For containerized inference, defaults are provided via `INPUT_DIR` and `JSON_PATH` (see Docker section).

Reproducibility
---------------
- Global seeds and determinism flags are set in `z_train.sh` (`SEED`, `DETERMINISTIC`).
- AMP, EMA, and early-stopping settings are exposed and can be tuned via environment variables.

Citation
--------
If you find DAUS-Net useful, please cite:

```
@inproceedings{kim2025dausnet,
  title={DAUS-Net: A Dynamically Adaptable Framework for Multi-task Ultrasound Analysis},
  author={Kim, Jonghwan and Choi, Bo Hwa and Ho, David Joon},
  booktitle={Proceedings of ...},
  year={2025}
}
```

Contact
-------
- Department of Public Health and AI, Graduate School of Cancer Science and Policy, National Cancer Center Korea
- Assistant Researcher : Jonghwan Kim 
- Email: army@ncc.re.kr

License
-------
