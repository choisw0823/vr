#!/bin/bash

# MSRVTT 전체 특징 추출 스크립트

export MASTER_PORT=$((12000 + $RANDOM % 20000))
export OMP_NUM_THREADS=1

echo "PYTHONPATH: ${PYTHONPATH}"
which_python=$(which python)
echo "Using Python at: ${which_python}"
export PYTHONPATH=${PYTHONPATH}:${which_python}
export PYTHONPATH=${PYTHONPATH}:.
echo "Updated PYTHONPATH: ${PYTHONPATH}"

JOB_NAME="extract_all_features_$(date +"%Y%m%d_%H%M%S")"
OUTPUT_DIR="./outputs/msrvtt_features/$JOB_NAME"

NNODE=1
NUM_GPUS=1

echo "🚀 Starting MSRVTT feature extraction..."
echo "📁 Output directory: $OUTPUT_DIR"

# 특징 추출 실행 (텍스트만, 메모리 절약 모드)
torchrun \
    --nnodes=${NNODE} \
    --nproc_per_node=${NUM_GPUS} \
    --rdzv_backend=c10d \
    --rdzv_endpoint=localhost:${MASTER_PORT} \
    extract_all_msrvtt_features.py \
    scripts/evaluation/stage2/zero_shot/6B/config_msrvtt.py \
    output_dir ${OUTPUT_DIR} \
    evaluate True \
    pretrained_path '/home/work/smoretalk/seo/reranking/new_d/InternVideo2-Stage2_6B-224p-f4/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt' \
    --text_only
echo "✅ Feature extraction completed!"
echo "📊 Check results in: $OUTPUT_DIR"
