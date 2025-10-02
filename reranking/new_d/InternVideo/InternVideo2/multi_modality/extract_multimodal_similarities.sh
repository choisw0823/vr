#!/bin/bash

# MSRVTT Multimodal Similarity Extraction Script
# Extract all text-video similarities using cross-attention

# 환경 변수 설정
export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

# Python 경로 확인
PYTHON_PATH=$(which python)
echo "Using Python at: $PYTHON_PATH"

# PYTHONPATH 업데이트
export PYTHONPATH="${PYTHONPATH}:$PYTHON_PATH:."
echo "Updated PYTHONPATH: $PYTHONPATH"

echo "🚀 Starting MSRVTT multimodal similarity extraction..."
echo "📁 Output directory: ./outputs/msrvtt_features/multimodal_similarities_$(date +%Y%m%d_%H%M%S)"

OUTPUT_DIR="./outputs/msrvtt_features/multimodal_similarities_$(date +%Y%m%d_%H%M%S)"

torchrun \
    --nproc_per_node=1 \
    --master_port=12345 \
    extract_msrvtt_multimodal_similarities.py \
    ./scripts/evaluation/stage2/zero_shot/6B/config_msrvtt.py \
    output_dir $OUTPUT_DIR \
    pretrained_path /home/work/smoretalk/seo/reranking/new_d/InternVideo2-Stage2_6B-224p-f4/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt

echo "✅ Multimodal similarity extraction completed!"
echo "📊 Check results in: ./outputs/msrvtt_features/multimodal_similarities_*"
