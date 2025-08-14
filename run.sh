#!/bin/bash

# ==================================================================
# 아래 경로들을 실제 데이터가 있는 절대 경로로 수정해주세요.
# ==================================================================
HOST_PROJECT_DIR="/home/army/workspace/DAUS-Net" # 프로젝트 최상위 폴더
HOST_DATA_DIR="${HOST_PROJECT_DIR}/data"
HOST_OUTPUT_DIR="${HOST_PROJECT_DIR}/sample_result_submission"
HOST_CKPT_PATH="${HOST_PROJECT_DIR}/train_output/final_run/best_model_tm_147_0.8093.pth"
# ==================================================================


echo "실행에 사용될 경로 정보:"
echo "입력 데이터 폴더: ${HOST_DATA_DIR}/Val"
echo "JSON 파일: ${HOST_DATA_DIR}/private_val_for_participants.json"
echo "결과물 저장 폴더: ${HOST_OUTPUT_DIR}"
echo "모델 가중치 파일: ${HOST_CKPT_PATH}"
echo "-------------------------------------------------------------"


docker run --gpus all --rm \
  -v "${HOST_DATA_DIR}/Val":/input/:ro \
  -v "${HOST_OUTPUT_DIR}":/output \
  -v "${HOST_DATA_DIR}/private_val_for_participants.json":/input.json:ro \
  -v "${HOST_CKPT_PATH}":/weights/best_model_tm.pth:ro \
  -e CKPT=/weights/best_model_tm.pth \
  -e JSON_PATH=/input.json \
  -it uusic
