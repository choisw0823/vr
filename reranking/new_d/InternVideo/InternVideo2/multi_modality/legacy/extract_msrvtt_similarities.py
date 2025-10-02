#!/usr/bin/env python3
"""
MSRVTT Similarity Matrix Extraction Script
Extract all text-video similarities from InternVideo2 model
"""

import json
import logging
import os

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader

from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, init_distributed_mode, is_main_process
from utils.logger import setup_logger
from dataset import create_dataset, create_loader
from models import *
from tasks.shared_utils import setup_model
from models.criterions import get_sim
from tqdm import tqdm
import pickle


def extract_text_feats(texts, max_txt_l, tokenizer, model, device):
    """Extract text features"""
    num_text = len(texts)
    text_bs = 64  # Batch size for text processing
    
    text_feats = []
    text_atts = []
    
    for i in tqdm(range(0, num_text, text_bs), desc="Extracting text features"):
        text_batch = texts[i:i+text_bs]
        
        text_input = tokenizer(
            text_batch,
            padding="max_length",
            truncation=True,
            max_length=max_txt_l,
            return_tensors="pt",
        ).to(device)
        
        text_feat = model.encode_text(text_input)
        text_feats.append(text_feat.cpu())
        text_atts.append(text_input.attention_mask.cpu())
    
    text_feats = torch.cat(text_feats, dim=0)
    text_atts = torch.cat(text_atts, dim=0)
    
    return text_feats, text_atts


def extract_vision_feats(data_loader, model, device, config):
    """Extract vision features"""
    model.eval()
    
    image_feats = []
    pooled_image_feats = []
    
    for data_batch in tqdm(data_loader, desc="Extracting vision features"):
        media = data_batch["image"].to(device, non_blocking=True)
        
        with torch.no_grad():
            image_feat, pooled_image_feat = model.encode_vision(media, test=True)
            
        if config.evaluation.eval_offload:
            image_feats.append(image_feat.cpu())
            pooled_image_feats.append(pooled_image_feat.cpu())
        else:
            image_feats.append(image_feat)
            pooled_image_feats.append(pooled_image_feat)
    
    image_feats = torch.cat(image_feats, dim=0)
    pooled_image_feats = torch.cat(pooled_image_feats, dim=0)
    
    return image_feats, pooled_image_feats


def compute_similarity_matrix(model, text_feats, pooled_image_feats, device, batch_size=64):
    """Compute full similarity matrix between all texts and videos"""
    
    num_texts = text_feats.shape[0]
    num_videos = pooled_image_feats.shape[0]
    
    # Project features
    text_proj_feats = model.text_proj(text_feats[:, 0])  # [num_texts, dim]
    vision_proj_feats = model.vision_proj(pooled_image_feats)  # [num_videos, dim]
    
    # Initialize similarity matrix
    similarity_matrix = torch.zeros(num_texts, num_videos)
    
    # Compute similarities in batches to avoid OOM
    with torch.no_grad():
        for i in tqdm(range(0, num_texts, batch_size), desc="Computing similarities"):
            end_i = min(i + batch_size, num_texts)
            text_batch = text_proj_feats[i:end_i].to(device)
            
            for j in range(0, num_videos, batch_size):
                end_j = min(j + batch_size, num_videos)
                vision_batch = vision_proj_feats[j:end_j].to(device)
                
                # Compute similarity using the same method as get_sim
                vision_norm = torch.nn.functional.normalize(vision_batch, dim=-1)
                text_norm = torch.nn.functional.normalize(text_batch, dim=-1)
                
                sim_batch = text_norm @ vision_norm.T  # [batch_texts, batch_videos]
                similarity_matrix[i:end_i, j:end_j] = sim_batch.cpu()
    
    return similarity_matrix


def main(config):
    init_distributed_mode(config)
    device = torch.device(config.device)
    
    # Setup logger
    os.makedirs(config.output_dir, exist_ok=True)
    logger = setup_logger("internvideo2_similarity", config.output_dir, get_rank())
    
    logger.info("Creating dataset")
    test_dataset = create_dataset("ret_eval", config)
    test_loader = create_loader(
        [test_dataset],
        [None],
        batch_size=[config.inputs.batch_size_test.video],
        num_workers=[config.num_workers],
        is_trains=[False],
        collate_fns=[None],
    )[0]
    
    logger.info("Creating model")
    model_cls = eval(config.model.model_cls)
    (model, model_without_ddp, optimizer, scheduler, scaler, tokenizer, start_epoch, global_step) = setup_model(
        config,
        model_cls,
        pretrain=False,
        find_unused_parameters=False
    )
    
    model.eval()
    logger.info("Model loaded successfully")
    
    # Extract text data
    logger.info("Preparing text data")
    texts = test_dataset.text
    max_txt_l = test_dataset.max_txt_l
    
    logger.info(f"Number of texts: {len(texts)}")
    logger.info(f"Number of videos: {len(test_dataset.image)}")
    
    # Extract features
    logger.info("Extracting text features...")
    text_feats, text_atts = extract_text_feats(texts, max_txt_l, tokenizer, model, device)
    
    logger.info("Extracting vision features...")
    image_feats, pooled_image_feats = extract_vision_feats(test_loader, model, device, config)
    
    # Compute similarity matrix
    logger.info("Computing similarity matrix...")
    similarity_matrix = compute_similarity_matrix(
        model, text_feats, pooled_image_feats, device, batch_size=32
    )
    
    # Save results
    output_file = os.path.join(config.output_dir, "msrvtt_similarity_matrix.pkl")
    results = {
        'similarity_matrix': similarity_matrix.numpy(),
        'texts': texts,
        'video_paths': test_dataset.image,
        'num_texts': len(texts),
        'num_videos': len(test_dataset.image),
        'config': config
    }
    
    logger.info(f"Saving similarity matrix to {output_file}")
    with open(output_file, 'wb') as f:
        pickle.dump(results, f)
    
    # Also save as JSON for easier inspection
    json_file = os.path.join(config.output_dir, "msrvtt_similarity_info.json")
    json_results = {
        'num_texts': len(texts),
        'num_videos': len(test_dataset.image),
        'texts': texts[:10],  # First 10 for inspection
        'video_paths': test_dataset.image[:10],  # First 10 for inspection
        'similarity_shape': list(similarity_matrix.shape),
        'similarity_stats': {
            'mean': float(similarity_matrix.mean()),
            'std': float(similarity_matrix.std()),
            'min': float(similarity_matrix.min()),
            'max': float(similarity_matrix.max())
        }
    }
    
    with open(json_file, 'w') as f:
        json.dump(json_results, f, indent=2)
    
    logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    logger.info(f"Similarity statistics - Mean: {similarity_matrix.mean():.4f}, Std: {similarity_matrix.std():.4f}")
    logger.info(f"Results saved to {output_file} and {json_file}")


if __name__ == "__main__":
    import sys
    
    # 기본 config 파일 설정
    if len(sys.argv) == 1:
        sys.argv.extend([
            'scripts/evaluation/stage2/zero_shot/6B/config_msrvtt.py',
            'output_dir', './outputs/msrvtt_similarities/test',
            'evaluate', 'True',
            'pretrained_path', '/home/work/smoretalk/seo/reranking/new_d/InternVideo2-Stage2_6B-224p-f4/internvideo2-s2_6b-224p-f4_with_audio_encoder.pt'
        ])
    
    config = setup_main()
    main(config)
