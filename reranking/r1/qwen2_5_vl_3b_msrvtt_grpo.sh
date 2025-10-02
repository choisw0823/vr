#!/bin/bash

set -x

export PYTHONUNBUFFERED=1

MODEL_PATH=/home/work/smoretalk/seo/reranking/sft_training/outputs  # replace it with your local file path

python3 -m verl.trainer.main \
    config=msrvtt.yaml \
    data.train_files=/home/work/smoretalk/seo/reranking/r1/data/msrvtt_ranking_dataset.json \
    data.val_files=/home/work/smoretalk/seo/reranking/r1/data/val_ranking_dataset.json \
    worker.actor.model.model_path=${MODEL_PATH} \
    worker.rollout.tensor_parallel_size=1 \
    trainer.experiment_name=qwen2_5_vl_3b_msrvtt_grpo \
    trainer.n_gpus_per_node=1
