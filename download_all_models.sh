#!/bin/bash

# Get the directory where the script is located
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )"

echo "Starting download of all models and data from Hugging Face Hub..."

# 1. Download SFT models
echo "Downloading SFT models..."
huggingface-cli download ccchhhoi/qwen2_5_vl_3b_sft_msrvtt --local-dir "$SCRIPT_DIR/reranking/sft_training/outputs/" --exclude "*.gitattributes" "*.gitignore"
if [ $? -ne 0 ]; then
    echo "Error: Failed to download SFT models."
    exit 1
fi
echo "SFT models download complete."

# 2. Download Easy-r1 checkpoints
echo "Downloading Easy-r1 checkpoints..."
huggingface-cli download ccchhhoi/qwen2_5_vl_3b_easy_r1_checkpoints --local-dir "$SCRIPT_DIR/reranking/r1/checkpoints/" --exclude "*.gitattributes" "*.gitignore"
if [ $? -ne 0 ]; then
    echo "Error: Failed to download Easy-r1 checkpoints."
    exit 1
fi
echo "Easy-r1 checkpoints download complete."

# 3. Download InternVideo2 model
echo "Downloading InternVideo2 model..."
huggingface-cli download ccchhhoi/qwen2_5_vl_3b_internvideo2 --local-dir "$SCRIPT_DIR/reranking/new_d/InternVideo2-Stage2_6B-224p-f4/" --exclude "*.gitattributes" "*.gitignore"
if [ $? -ne 0 ]; then
    echo "Error: Failed to download InternVideo2 model."
    exit 1
fi
echo "InternVideo2 model download complete."

# 4. Download MSRVTT features
echo "Downloading MSRVTT features..."
huggingface-cli download datasets/ccchhhoi/qwen2_5_vl_3b_msrvtt_features --local-dir "$SCRIPT_DIR/reranking/new_d/InternVideo/InternVideo2/multi_modality/outputs/msrvtt_features/" --exclude "*.gitattributes" "*.gitignore"
if [ $? -ne 0 ]; then
    echo "Error: Failed to download MSRVTT features."
    exit 1
fi
echo "MSRVTT features download complete."

echo "All models and data download complete."
