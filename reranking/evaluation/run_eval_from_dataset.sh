#!/bin/bash

# 생성된 평가 데이터셋을 사용한 Qwen VL 2.5 평가 실행 스크립트

set -e

# 기본 설정
MODEL_PATH="/home/work/smoretalk/seo/reranking/sft_training/outputs"  # 모델 경로 수정 필요
DATASET_PATH="evaluation_dataset.json"  # 생성된 평가 데이터셋
OUTPUT_PATH="evaluation_results.json"
MAX_SAMPLES=3000  # 테스트용으로 100개만 평가 (전체: 3000개)

echo "=== Qwen VL 2.5 Reranking Evaluation (from Dataset) ==="
echo "Model: $MODEL_PATH"
echo "Dataset: $DATASET_PATH"
echo "Max samples: $MAX_SAMPLES"
echo ""

# 데이터셋 파일 확인
if [ ! -f "$DATASET_PATH" ]; then
    echo "Error: Dataset file not found: $DATASET_PATH"
    echo "Please run ./create_dataset.sh first to generate the evaluation dataset"
    exit 1
fi

# Python 환경 확인
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# 평가 실행
python3 evaluate_from_dataset.py \
    --model_path "$MODEL_PATH" \
    --dataset_path "$DATASET_PATH" \
    --output_path "$OUTPUT_PATH" \
    --max_samples "$MAX_SAMPLES" \
    --device "cuda"

echo ""
echo "=== Evaluation Complete ==="
echo "Results saved to: $OUTPUT_PATH"

# 결과 요약 출력
if command -v jq &> /dev/null; then
    echo ""
    echo "=== Results Summary ==="
    echo "Overall R@1: $(cat $OUTPUT_PATH | jq -r '.overall_metrics.r1_score')"
    echo "Overall nDCG@5: $(cat $OUTPUT_PATH | jq -r '.overall_metrics.ndcg_score')"
    echo "Total samples: $(cat $OUTPUT_PATH | jq -r '.overall_metrics.total_samples')"
    echo ""
    echo "Case-wise analysis:"
    cat $OUTPUT_PATH | jq -r '.case_analysis | to_entries[] | "  \(.key): R@1=\(.value.r1_score | . * 100 | floor / 100), nDCG@5=\(.value.ndcg_score | floor / 100)"'
fi
