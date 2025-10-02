#!/usr/bin/env python3
"""
MSRVTT Multimodal Similarity Matrix Extraction Script (UNIQUE VERSION)
Extract similarities using unique captions only from train dataset
- Filter out captions that appear 2+ times across the dataset
- Random selection of 9,000 unique captions (1 per video)
- Only process train dataset videos
"""

import json
import logging
import os
import random
import pickle
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.distributed as dist
from torch.utils.data import DataLoader
from tqdm import tqdm

from utils.config_utils import setup_main
from utils.distributed import get_rank, get_world_size, init_distributed_mode, is_main_process
from utils.logger import setup_logger
from dataset import create_dataset, create_loader
from models import *
from tasks.shared_utils import setup_model
from models.criterions import get_sim
from tasks.retrieval_utils import extract_text_feats, extract_vision_feats

logger = logging.getLogger(__name__)

def load_msrvtt_unique_train_data():
    """Load train MSRVTT data with unique captions preferred (train data only)"""
    logger.info("Loading MSRVTT train annotation files...")
    
    # Load only train annotations for caption frequency analysis
    train_file = "/home/work/smoretalk/seo/reranking/dataset/MSR-VTT/msrvtt_train_9k.json"
    
    video_root = "/home/work/smoretalk/seo/reranking/new_d/MSR-VTT/video"
    
    # First pass: collect train captions only to find duplicates
    logger.info("First pass: analyzing train caption frequencies...")
    all_captions = []
    
    # Collect captions from train data only
    with open(train_file, 'r') as f:
        train_data = json.load(f)
    
    for video_info in train_data:
        captions = video_info['caption']  # List of ~20 captions
        for caption in captions:
            all_captions.append(caption.strip())
    
    # Count caption frequencies (train data only)
    caption_counts = Counter(all_captions)
    unique_captions = {caption for caption, count in caption_counts.items() if count == 1}
    
    logger.info(f"Total train captions: {len(all_captions)}")
    logger.info(f"Unique captions in train (appear only once): {len(unique_captions)}")
    logger.info(f"Duplicate captions in train (appear 2+ times): {len(caption_counts) - len(unique_captions)}")
    
    # Second pass: process all 9000 train videos, select unique caption for each
    logger.info("Second pass: processing all 9000 train videos...")
    all_videos = []
    all_texts = []
    video_text_mapping = []
    
    videos_with_unique_captions = 0
    videos_without_unique_captions = 0
    
    for video_info in train_data:
        video_id = video_info['video_id']
        video_path = os.path.join(video_root, f"{video_id}.mp4")
        
        if os.path.exists(video_path):
            # Find unique captions for this video
            captions = video_info['caption']
            unique_video_captions = [cap.strip() for cap in captions if cap.strip() in unique_captions]
            
            if unique_video_captions:
                # Randomly select one unique caption
                selected_caption = random.choice(unique_video_captions)
                videos_with_unique_captions += 1
            else:
                # If no unique captions, randomly select any caption (fallback)
                selected_caption = random.choice(captions).strip()
                videos_without_unique_captions += 1
                logger.warning(f"Video {video_id} has no unique captions, using fallback: '{selected_caption[:50]}...'")
            
            # Add all videos (9000 total)
            all_videos.append({
                'video_id': video_id,
                'video_path': video_path,
                'split': 'train'
            })
            all_texts.append({
                'caption': selected_caption,
                'video_id': video_id,
                'split': 'train'
            })
            video_text_mapping.append(len(all_videos) - 1)  # Video index
    
    logger.info(f"Videos with unique captions: {videos_with_unique_captions}")
    logger.info(f"Videos without unique captions (fallback used): {videos_without_unique_captions}")
    logger.info(f"Total train videos processed: {len(all_videos)}")
    
    logger.info(f"Final dataset: {len(all_videos)} videos and {len(all_texts)} texts (train only, unique captions preferred)")
    return all_videos, all_texts, video_text_mapping

def extract_video_features_batch(model, all_videos, device, config, cache_dir=None):
    """Extract video token sequences using encode_vision directly with caching"""
    from dataset import get_test_transform  
    import einops
    import os
    
    # Check for cached video tokens
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "video_tokens.pkl")
        
        if os.path.exists(cache_path):
            logger.info(f"Loading cached video tokens from {cache_path}...")
            try:
                with open(cache_path, 'rb') as f:
                    video_tokens, video_atts = pickle.load(f)
                logger.info(f"Loaded cached video tokens shape: {video_tokens.shape}")
                return video_tokens, video_atts
            except Exception as e:
                logger.warning(f"Failed to load cached video tokens: {e}")
                logger.info("Extracting video tokens from scratch...")
    
    logger.info(f"Extracting video token sequences from {len(all_videos)} videos...")
    
    # Get test transform for video processing
    class TempConfig:
        def __init__(self, media_type, data_root):
            self.media_type = media_type
            self.data_root = data_root
        def get(self, key, default=None):
            return getattr(self, key, default)
    
    temp_config = TempConfig('video', os.path.dirname(all_videos[0]['video_path']))
    test_transform = get_test_transform(config, temp_config)
    
    all_video_tokens = []
    
    # Process videos in very small batches to avoid OOM
    batch_size = 1
    for i in tqdm(range(0, len(all_videos), batch_size), desc="Processing videos"):
        batch_videos = all_videos[i:i+batch_size]
        
        # Load and transform videos in batch
        video_batch = []
        for video_info in batch_videos:
            # Load video using same method as dataset
            from dataset.video_utils import VIDEO_READER_FUNCS
            video_reader = VIDEO_READER_FUNCS[getattr(config, 'video_reader_type', 'decord')]
            
            video_path = video_info['video_path']
            num_frames = getattr(config, 'num_frames_test', 4)
            sample_type = getattr(config, 'sample_type_test', 'middle')
            
            # Read video frames
            video_tensor, frame_indices, duration = video_reader(
                video_path, num_frames, sample_type
            )
            
            # Apply transform
            if test_transform:
                video_tensor = test_transform(video_tensor)
            
            video_batch.append(video_tensor)
        
        # Stack batch
        video_batch = torch.stack(video_batch).to(device)  # [B, T, C, H, W]
        
        # Extract video features using model
        with torch.no_grad():
            # Use encode_vision like in evaluation code
            # Handle DistributedDataParallel wrapper
            if hasattr(model, 'module'):
                image_feat, pooled = model.module.encode_vision(video_batch, test=True)
            else:
                image_feat, pooled = model.encode_vision(video_batch, test=True)
            
            # Reshape to match evaluation format: (B, Li, D)
            if len(image_feat.shape) == 4:   # (B, T, L, D)
                image_feat = einops.rearrange(image_feat, 'b t l d -> b (t l) d').contiguous()
            elif len(image_feat.shape) == 3:  # (B, L, D) 
                pass  # Already correct format
            
            all_video_tokens.append(image_feat.cpu())
        
        # Clear memory aggressively
        del video_batch, image_feat, pooled
        torch.cuda.empty_cache()
        
        # Additional memory cleanup every 10 videos
        if i % 10 == 0:
            torch.cuda.empty_cache()
            import gc
            gc.collect()
    
    # Concatenate all video tokens
    video_tokens = torch.cat(all_video_tokens, dim=0)  # (N, Li, D) - already on CPU
    video_atts = torch.ones(video_tokens.size()[:2], dtype=torch.long)  # (N, Li) - CPU tensor
    
    logger.info(f"Extracted video tokens shape: {video_tokens.shape}")
    
    # Save video tokens to cache (ensure CPU tensors)
    if cache_dir:
        cache_path = os.path.join(cache_dir, "video_tokens.pkl")
        logger.info(f"Saving video tokens to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            # Ensure both tensors are on CPU before saving
            pickle.dump((video_tokens.cpu(), video_atts.cpu()), f)
        logger.info("Video tokens cached successfully!")
    
    return video_tokens, video_atts

def extract_text_features_batch(model, texts, tokenizer, device, max_txt_l=77, cache_dir=None):
    """Extract text token sequences using the same path as evaluation code with caching"""
    from tasks.retrieval_utils import extract_text_feats
    import os
    
    # Check for cached text tokens
    if cache_dir:
        os.makedirs(cache_dir, exist_ok=True)
        cache_path = os.path.join(cache_dir, "text_tokens.pkl")
        
        if os.path.exists(cache_path):
            logger.info(f"Loading cached text tokens from {cache_path}...")
            try:
                with open(cache_path, 'rb') as f:
                    text_feats, text_atts = pickle.load(f)
                logger.info(f"Loaded cached text tokens shape: {text_feats.shape}")
                return text_feats, text_atts
            except Exception as e:
                logger.warning(f"Failed to load cached text tokens: {e}")
                logger.info("Extracting text tokens from scratch...")
    
    logger.info(f"Extracting text token sequences from {len(texts)} texts...")
    
    # Process texts in small batches to avoid OOM
    actual_model = model.module if hasattr(model, 'module') else model
    actual_model.eval()  # Ensure eval mode
    text_feats_list = []
    text_atts_list = []
    
    batch_size = 1  # Process texts in batches
    for i in tqdm(range(0, len(texts), batch_size), desc="Processing text batches"):
        batch_texts = texts[i:i+batch_size]
        
        # Extract features for this batch
        with torch.no_grad():  # Ensure no gradients
            batch_text_feats, batch_text_atts = extract_text_feats(
                batch_texts, max_txt_l, tokenizer, actual_model, device
            )
        
        # Move to CPU immediately to save GPU memory
        text_feats_list.append(batch_text_feats.cpu())
        text_atts_list.append(batch_text_atts.cpu())
        
        # Clear GPU memory aggressively
        del batch_text_feats, batch_text_atts
        torch.cuda.empty_cache()
        
        # Force cleanup every single batch
        import gc
        gc.collect()
        torch.cuda.empty_cache()
    
    # Concatenate all text features
    text_feats = torch.cat(text_feats_list, dim=0)  # (N, L, D)
    text_atts = torch.cat(text_atts_list, dim=0)    # (N, L)
    
    logger.info(f"Extracted text tokens shape: {text_feats.shape}")
    
    # Save text tokens to cache (ensure CPU tensors)
    if cache_dir:
        cache_path = os.path.join(cache_dir, "text_tokens.pkl")
        logger.info(f"Saving text tokens to cache: {cache_path}")
        with open(cache_path, 'wb') as f:
            # Ensure both tensors are on CPU before saving
            pickle.dump((text_feats.cpu(), text_atts.cpu()), f)
        logger.info("Text tokens cached successfully!")
    
    return text_feats, text_atts


def compute_multimodal_similarity(model, video_tokens, video_atts, text_tokens, text_atts, device, batch_size=64, output_dir=None, checkpoint_interval=100, shared_cache_dir=None):
    """
    Compute multimodal similarity using cross-attention (OPTIMIZED version)
    Args:
        video_tokens: [num_videos, video_seq_len, d] - video token sequences
        video_atts: [num_videos, video_seq_len] - video attention masks
        text_tokens: [num_texts, text_seq_len, d] - text token sequences  
        text_atts: [num_texts, text_seq_len] - text attention masks
        batch_size: larger batch size for efficient processing
        output_dir: directory to save checkpoints
        checkpoint_interval: save checkpoint every N text items
    Returns:
        similarity_matrix: [num_videos, num_texts]
    """
    model.eval()
    num_videos = video_tokens.shape[0]
    num_texts = text_tokens.shape[0]
    
    # Check for existing checkpoint
    checkpoint_path = None
    start_t_idx = 0
    similarity_matrix = torch.zeros(num_videos, num_texts)
    
    if shared_cache_dir:
        os.makedirs(shared_cache_dir, exist_ok=True)
        checkpoint_path = os.path.join(shared_cache_dir, "similarity_checkpoint.pkl")
        if os.path.exists(checkpoint_path):
            logger.info(f"Found checkpoint at {checkpoint_path}, resuming...")
            with open(checkpoint_path, 'rb') as f:
                checkpoint = pickle.load(f)
                similarity_matrix = torch.tensor(checkpoint['similarity_matrix'])
                start_t_idx = checkpoint['last_completed_text_idx'] + 1
                logger.info(f"Resuming from text index {start_t_idx}/{num_texts}")
    
    logger.info(f"Computing multimodal similarity matrix: {num_videos} videos × {num_texts} texts")
    logger.info(f"Starting from text index: {start_t_idx}")
    logger.info(f"Using optimized batch processing with batch_size={batch_size}")
    
    actual_model = model.module if hasattr(model, 'module') else model
    
    with torch.no_grad():
        # 텍스트와 비디오 모두 배치로 처리 (메모리 안전)
        text_batch_size = min(10, batch_size // 96)  # 텍스트 배치 크기 (안전한 설정)
        
        for t_start in tqdm(range(start_t_idx, num_texts, text_batch_size), desc="Processing text batches"):
            t_end = min(t_start + text_batch_size, num_texts)
            current_text_batch_size = t_end - t_start
            
            # Get text batch
            text_batch = text_tokens[t_start:t_end].to(device)  # [t_batch, text_seq, d]
            text_att_batch = text_atts[t_start:t_end].to(device)  # [t_batch, text_seq]
            
            # Store similarities for current text batch
            batch_similarities_matrix = torch.zeros(num_videos, current_text_batch_size)
            
            for v_start in range(0, num_videos, batch_size):
                v_end = min(v_start + batch_size, num_videos)
                current_video_batch_size = v_end - v_start
                
                # Get video batch
                v_batch_tokens = video_tokens[v_start:v_end].to(device)  # [v_batch, video_seq, d]
                v_batch_atts = video_atts[v_start:v_end].to(device)      # [v_batch, video_seq]
                
                # 매트릭스 연산으로 모든 text-video 쌍을 한번에 처리
                # 각 텍스트와 각 비디오의 모든 조합을 한번에 계산
                
                # Expand texts and videos for all combinations
                # text_batch: [t_batch, text_seq, d] -> [t_batch * v_batch, text_seq, d]
                # v_batch_tokens: [v_batch, video_seq, d] -> [t_batch * v_batch, video_seq, d]
                
                expanded_texts = text_batch.unsqueeze(1).repeat(1, current_video_batch_size, 1, 1).reshape(-1, text_batch.shape[1], text_batch.shape[2])  # [t*v, text_seq, d]
                expanded_text_atts = text_att_batch.unsqueeze(1).repeat(1, current_video_batch_size, 1).reshape(-1, text_att_batch.shape[1])  # [t*v, text_seq]
                expanded_videos = v_batch_tokens.unsqueeze(0).repeat(current_text_batch_size, 1, 1, 1).reshape(-1, v_batch_tokens.shape[1], v_batch_tokens.shape[2])  # [t*v, video_seq, d]
                expanded_video_atts = v_batch_atts.unsqueeze(0).repeat(current_text_batch_size, 1, 1).reshape(-1, v_batch_atts.shape[1])  # [t*v, video_seq]
                
                # 한번에 모든 쌍 처리
                if hasattr(actual_model, 'text_encoder'):
                    fusion_output = actual_model.text_encoder(
                        encoder_embeds=expanded_texts,                    # [t*v, text_seq, d]
                        attention_mask=expanded_text_atts,                # [t*v, text_seq]
                        encoder_hidden_states=expanded_videos,            # [t*v, video_seq, d]
                        encoder_attention_mask=expanded_video_atts,       # [t*v, video_seq]
                        return_dict=True,
                        mode="fusion"
                    )
                    
                    # Get fusion CLS tokens
                    fusion_cls = fusion_output.last_hidden_state[:, 0]  # [t*v, d]
                    
                    # Compute ITM scores in batch
                    if hasattr(actual_model, 'itm_head') and actual_model.itm_head is not None:
                        itm_scores = actual_model.itm_head(fusion_cls)  # [t*v, 2]
                        all_similarities = itm_scores[:, 1]  # [t*v] - positive scores
                    else:
                        # Fallback: batch cosine similarity
                        v_proj = actual_model.vision_proj(expanded_videos[:, 0])  # [t*v, d]
                        t_proj = actual_model.text_proj(expanded_texts[:, 0])     # [t*v, d]
                        all_similarities = torch.cosine_similarity(v_proj, t_proj, dim=-1)  # [t*v]
                else:
                    # Fallback if no text_encoder
                    v_proj = actual_model.vision_proj(expanded_videos[:, 0])  # [t*v, d]
                    t_proj = actual_model.text_proj(expanded_texts[:, 0])     # [t*v, d]
                    all_similarities = torch.cosine_similarity(v_proj, t_proj, dim=-1)  # [t*v]
                
                # Reshape back to matrix form [v_batch, t_batch]
                similarities_matrix = all_similarities.reshape(current_text_batch_size, current_video_batch_size).T  # [v_batch, t_batch]
                
                # Store in global matrix
                batch_similarities_matrix[v_start:v_end, :] = similarities_matrix.cpu()
                
                # Clear all intermediate tensors
                del expanded_texts, expanded_text_atts, expanded_videos, expanded_video_atts
                del all_similarities, similarities_matrix
                if hasattr(actual_model, 'text_encoder'):
                    del fusion_output, fusion_cls
                
                # Clear video batch tensors
                del v_batch_tokens, v_batch_atts
                
                # Memory cleanup every few batches
                if v_start % (batch_size * 3) == 0:
                    torch.cuda.empty_cache()
            
            # Fill similarity matrix for current text batch
            similarity_matrix[:, t_start:t_end] = batch_similarities_matrix
            
            # Clear text batch tensors
            del text_batch, text_att_batch, batch_similarities_matrix
            
            # Memory cleanup less frequently
            if t_start % (text_batch_size * 5) == 0:
                torch.cuda.empty_cache()
            
            # Save checkpoint more frequently
            if shared_cache_dir and (t_end - start_t_idx) % checkpoint_interval == 0:
                checkpoint_data = {
                    'similarity_matrix': similarity_matrix.numpy(),
                    'last_completed_text_idx': t_end - 1,
                    'total_texts': num_texts,
                    'total_videos': num_videos,
                    'progress_percentage': (t_end / num_texts) * 100
                }
                with open(checkpoint_path, 'wb') as f:
                    pickle.dump(checkpoint_data, f)
                logger.info(f"💾 Checkpoint saved at text {t_end}/{num_texts} ({checkpoint_data['progress_percentage']:.1f}%)")
    
    # Remove checkpoint file when complete
    if shared_cache_dir and checkpoint_path and os.path.exists(checkpoint_path):
        os.remove(checkpoint_path)
        logger.info("🗑️ Checkpoint file removed (computation completed)")
    
    return similarity_matrix

def main(config):
    # Setup
    if not torch.distributed.is_initialized():
        init_distributed_mode(config)
    device = torch.device(config.device)
    
    # Setup logging
    setup_logger(config.output_dir)
    
    # Set random seed for reproducibility
    random.seed(42)
    torch.manual_seed(42)
    
    # Load model
    logger.info("Loading model...")
    model_cls = InternVideo2_Stage2_audiovisual
    model, model_without_ddp, optimizer, scheduler, scaler, tokenizer, start_epoch, global_step = setup_model(
        config, model_cls, pretrain=False, find_unused_parameters=config.model.get("find_unused_parameters", False)
    )
    model.eval()
    
    # Load unique train MSRVTT data
    logger.info("Loading unique train MSRVTT data...")
    all_videos, all_texts, video_text_mapping = load_msrvtt_unique_train_data()
    
    # Setup cache directory (unique version)
    cache_dir = "./outputs/msrvtt_features/token_cache_unique"
    
    # Extract text token sequences from raw texts
    text_captions = [text['caption'] for text in all_texts]
    text_tokens, text_atts = extract_text_features_batch(model, text_captions, tokenizer, device, cache_dir=cache_dir)
    
    # Extract video token sequences from raw videos
    video_tokens, video_atts = extract_video_features_batch(model, all_videos, device, config, cache_dir)
    
 
    # Compute multimodal similarity matrix using cross-attention (unique captions only)
    logger.info("Computing multimodal similarity matrix with cross-attention (unique captions only)...")
    output_dir = os.path.join(config.output_dir, "multimodal_similarities_unique")
    os.makedirs(output_dir, exist_ok=True)
    
    similarity_matrix = compute_multimodal_similarity(
        model, video_tokens, video_atts, text_tokens, text_atts, device, 
        batch_size=256,  # Conservative batch size for memory safety with text batching
        output_dir=output_dir,
        checkpoint_interval=getattr(config, 'checkpoint_interval', 50),  # More frequent checkpoints
        shared_cache_dir=cache_dir
    )
    
    # Save similarity matrix
    sim_path = os.path.join(output_dir, "similarity_matrix.pkl")
    with open(sim_path, 'wb') as f:
        pickle.dump({
            'similarity_matrix': similarity_matrix.numpy(),
            'video_data': all_videos,
            'text_data': all_texts,
            'video_text_mapping': video_text_mapping
        }, f)
    
    logger.info(f"Similarity matrix saved to {sim_path}")
    logger.info(f"Matrix shape: {similarity_matrix.shape}")
    
    # Save metadata
    meta_path = os.path.join(output_dir, "metadata.json")
    with open(meta_path, 'w') as f:
        json.dump({
            'num_videos': len(all_videos),
            'num_texts': len(all_texts),
            'matrix_shape': list(similarity_matrix.shape),
            'description': 'Multimodal similarity matrix using cross-attention with unique captions only (train dataset)',
            'unique_captions_only': True,
            'dataset_split': 'train'
        }, f, indent=2)
    
    logger.info(f"Metadata saved to {meta_path}")
    logger.info("✅ Unique caption multimodal similarity extraction completed!")

if __name__ == "__main__":
    config = setup_main()
    main(config)
