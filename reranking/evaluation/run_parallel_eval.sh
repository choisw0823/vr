#!/bin/bash

# 병렬 평가 실행 스크립트

# 설정
DATASET="/home/work/smoretalk/seo/reranking/evaluation/evaluation_dataset.json"
MODEL_PATH="Qwen/Qwen2.5-VL-3B-Instruct"
OUTPUT_DIR="./output/parallel_eval_$(date +%Y%m%d_%H%M%S)"
FINAL_OUTPUT="./output/parallel_eval_$(date +%Y%m%d_%H%M%S)/parallel_evaluation_result.json"
NUM_CHUNKS=1

echo "=== 병렬 평가 시작 ==="
echo "데이터셋: $DATASET"
echo "모델: $MODEL_PATH"
echo "출력 디렉토리: $OUTPUT_DIR"
echo "최종 결과: $FINAL_OUTPUT"
echo "청크 수: $NUM_CHUNKS"
echo ""

# 병렬 평가 실행
python3 /home/work/smoretalk/seo/reranking/evaluation/run_parallel_eval.py \
    --dataset "$DATASET" \
    --model_path "$MODEL_PATH" \
    --output_dir "$OUTPUT_DIR" \
    --final_output "$FINAL_OUTPUT" \
    --num_chunks "$NUM_CHUNKS"

echo ""
echo "=== 평가 완료 ==="
echo "결과 파일: $FINAL_OUTPUT"
echo "임시 파일: $OUTPUT_DIR"
