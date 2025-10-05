#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/work/smoretalk/seo/reranking/sft_training/outputs  # replace it with your local file path

python3 -m verl.trainer.main \
    config=msrvtt.yaml \
    worker.actor.model.model_path=${MODEL_PATH} \
