#!/usr/bin/env bash
# 단일 GPU 학습 스크립트 (UUSIC25 TM, v2: Lovasz+Dice seg, LabelSmooth CE cls)
set -euo pipefail

# ========== 사용자 설정 ==========
ROOT=${ROOT:-data}
OUT=${OUT:-train_output/run_$(date +%y%m%d_%H%M%S)}
mkdir -p "${OUT}"

IMG=${IMG:-224}
BS=${BS:-32}
MAX_EPOCHS=${MAX_EPOCHS:-200}
PATIENCE=${PATIENCE:-45}
ES_METRIC=${ES_METRIC:-total_mean}  # total_mean | seg_mean | cls_mean
DEL_OUTLIER=${DEL_OUTLIER:-0}  # 1 to remove outliers, 0 to keep (default)

# 재현성: 전역 시드와 결정성
SEED=${SEED:-42}
DETERMINISTIC=${DETERMINISTIC:-1}
export PYTHONHASHSEED="${SEED}"
export CUBLAS_WORKSPACE_CONFIG=":4096:8"

# W&B
export WANDB_PROJECT=${WANDB_PROJECT:-uusic25_tm}
if [[ "${WANDB_OFF:-0}" == "1" ]]; then
  WANDB_ARGS=(--wandb_off)
else
  WANDB_ARGS=(--wandb_project "${WANDB_PROJECT}" ${WANDB_ENTITY:+--wandb_entity "$WANDB_ENTITY"} ${WANDB_RUN:+--wandb_run "$WANDB_RUN"})
fi

# 선택적 고급 옵션
AMP_DTYPE=${AMP_DTYPE:-}
LORA_RANK=${LORA_RANK:-8}
LORA_ALPHA=${LORA_ALPHA:-16}
LORA_DROPOUT=${LORA_DROPOUT:-0.0}
LORA_ONLY=${LORA_ONLY:-}
W_ALIGN=${W_ALIGN:-0.05}
MAX_LORA_SCALE=${MAX_LORA_SCALE:-0.5}
SCALE_MODE=${SCALE_MODE:-sigmoid}

EXTRA_ARGS=()
if [[ -n "${AMP_DTYPE}" ]]; then EXTRA_ARGS+=(--amp_dtype "${AMP_DTYPE}"); fi
EXTRA_ARGS+=(--lora_rank "${LORA_RANK}" --lora_alpha "${LORA_ALPHA}" --lora_dropout "${LORA_DROPOUT}")
if [[ -n "${LORA_ONLY}" ]]; then EXTRA_ARGS+=(--lora_only); fi
if [[ "${DEL_OUTLIER}" == "1" ]]; then EXTRA_ARGS+=(--del_outlier); fi

# ========== 실행 ==========
torchrun --nproc_per_node=${NPROC:-1} train.py \
  --root_path "${ROOT}" \
  --output_dir "${OUT}" \
  --img_size "${IMG}" \
  --batch_size "${BS}" \
  --base_lr ${BASE_LR:-0.0005} \
  --max_epochs "${MAX_EPOCHS}" \
  --early_stop_patience "${PATIENCE}" \
  --early_stop_metric "${ES_METRIC}" \
  --seed "${SEED}" \
  --deterministic "${DETERMINISTIC}" \
  --prompt \
  --seg_skip_bg_only_prob ${SEG_SKIP_BG_ONLY_PROB:-0.5} \
  --seg_bce_weight ${SEG_BCE_W:-0.5} \
  --seg_dice_weight ${SEG_DICE_W:-0.5} \
  --amp ${AMP_ON:-1} \
  --seg_loss bce_dice \
  --cls_loss ls \
  --label_smoothing 0.05 \
  --warmup_epochs ${WARMUP_EPOCHS:-10} \
  --min_lr_ratio ${MIN_LR_RATIO:-0.03} \
  --head_lr_mult ${HEAD_LR_MULT:-1.2} \
  --clip_grad_norm ${CLIP_NORM:-1.0} \
  --w_align ${W_ALIGN} \
  --max_lora_scale ${MAX_LORA_SCALE} \
  --scale_mode ${SCALE_MODE} \
  --w_cls_start ${WCLS_START:-0.3} \
  --w_cls_end ${WCLS_END:-1.0} \
  --w_cls_ramp_epochs ${WCLS_RAMP_EPOCHS:-10} \
  --seg_area_weight ${SEG_AREA_W:-0.1} \
  --seg_fp_weight ${SEG_FP_W:-0.2} \
  --seg_fp_topk ${SEG_FP_TOPK:-0.1} \
  --cls_hard_weight_gamma ${CLS_HARD_GAMMA:-2.0} \
  --cls_ohem_frac ${CLS_OHEM_FRAC:-0.0} \
  --use_ema \
  --ema_decay ${EMA_DECAY:-0.999} \
  --plateau_patience ${PLATEAU_PATIENCE:-20} \
  --film_scale ${FILM_SCALE:-0.7} \
  --prior_lambda ${PRIOR_LAMBDA:-0.5} \
  --cls_dropout ${CLS_DROPOUT:-0.0} \
  --cls_head_variant ${CLS_HEAD_VARIANT:-linear} \
  "${WANDB_ARGS[@]}" \
  "${EXTRA_ARGS[@]}"