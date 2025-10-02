#!/bin/bash

# MSRVTT Similarity Extraction Script for InternVideo2

export PYTHONPATH="${PYTHONPATH}:$(pwd)"
export CUDA_VISIBLE_DEVICES=0

NUM_GPUS=1
NUM_NODES=1

OUTPUT_DIR="./outputs/msrvtt_similarities/$(date +'%Y%m%d_%H%M%S')"
mkdir -p $OUTPUT_DIR

echo "Starting MSRVTT similarity extraction..."
echo "Output directory: $OUTPUT_DIR"

torchrun \
  --nnodes=${NUM_NODES} \
  --nproc_per_node=${NUM_GPUS} \
  --rdzv_backend=c10d \
  --rdzv_endpoint=localhost:0 \
  extract_msrvtt_similarities.py \
  scripts/evaluation/stage2/zero_shot/6B/config_msrvtt.py \
  output_dir ${OUTPUT_DIR} \
  evaluate True \
  pretrained_path '/home/work/smoretalk/seo/reranking/new_d/InternVideo2-Stage2_6B-224p-f4/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt'

echo "Similarity extraction completed!"
echo "Results saved in: $OUTPUT_DIR"











