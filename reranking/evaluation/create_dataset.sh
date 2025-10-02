#!/bin/bash

# 평가 데이터셋 생성 스크립트

set -e

# 기본 설정
SIMILARITY_DATA="/home/work/smoretalk/seo/reranking/new_d/InternVideo/InternVideo2/multi_modality/outputs/msrvtt_features/multimodal_similarities_20250921_093544/multimodal_similarities/test_similarity_matrix_1000x1000.pkl"
VIDEO_BASE_PATH="/home/work/smoretalk/seo/reranking/new_d/MSR-VTT/video"  # 비디오 파일 경로 수정 필요
OUTPUT_PATH="evaluation_dataset.json"
MAX_SAMPLES=1000  # 전체 테스트 데이터 1000개

echo "=== Evaluation Dataset Creation ==="
echo "Similarity data: $SIMILARITY_DATA"
echo "Video base path: $VIDEO_BASE_PATH"
echo "Output: $OUTPUT_PATH"
echo "Max samples: $MAX_SAMPLES"
echo ""

# Python 환경 확인
if ! command -v python3 &> /dev/null; then
    echo "Error: python3 not found"
    exit 1
fi

# 데이터셋 생성
python3 create_evaluation_dataset.py \
    --similarity_data "$SIMILARITY_DATA" \
    --video_base_path "$VIDEO_BASE_PATH" \
    --output_path "$OUTPUT_PATH" \
    --max_samples "$MAX_SAMPLES"

echo ""
echo "=== Dataset Creation Complete ==="
echo "Dataset saved to: $OUTPUT_PATH"
echo ""
echo "Dataset structure:"
echo "- Total samples: $(cat $OUTPUT_PATH | jq '.metadata.total_samples')"
echo "- Unique texts: $(cat $OUTPUT_PATH | jq '.metadata.texts_count')"
echo "- Cases per text: $(cat $OUTPUT_PATH | jq '.metadata.cases_per_text')"
