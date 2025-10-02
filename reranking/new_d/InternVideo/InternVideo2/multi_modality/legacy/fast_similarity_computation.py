"""
저장된 MSRVTT 특징으로 빠른 similarity 계산 스크립트
"""

import json
import logging
import os
import pickle
import numpy as np
import torch
from collections import defaultdict

logger = logging.getLogger(__name__)

def load_features(feature_dir):
    """저장된 특징 파일들 로드"""
    
    # 비디오 특징 로드
    video_feat_file = os.path.join(feature_dir, "video_features.pkl")
    with open(video_feat_file, 'rb') as f:
        video_features = pickle.load(f)
    
    # 비디오 메타데이터 로드
    video_meta_file = os.path.join(feature_dir, "video_metadata.pkl")
    with open(video_meta_file, 'rb') as f:
        video_metadata = pickle.load(f)
    
    # 텍스트 특징 로드
    text_feat_file = os.path.join(feature_dir, "text_features.pkl")
    with open(text_feat_file, 'rb') as f:
        text_features = pickle.load(f)
    
    # 텍스트 메타데이터 로드
    text_meta_file = os.path.join(feature_dir, "text_metadata.pkl")
    with open(text_meta_file, 'rb') as f:
        text_metadata = pickle.load(f)
    
    # 통계 정보 로드
    stats_file = os.path.join(feature_dir, "extraction_stats.json")
    with open(stats_file, 'r') as f:
        stats = json.load(f)
    
    return video_features, video_metadata, text_features, text_metadata, stats

def compute_similarity_matrices(video_features, text_features, normalize=True):
    """비디오-텍스트 similarity 매트릭스 계산"""
    
    # 모든 비디오 특징을 하나의 텐서로 결합
    video_ids = list(video_features.keys())
    video_feat_tensor = torch.cat([video_features[vid] for vid in video_ids], dim=0)
    
    # 모든 텍스트 특징을 하나의 텐서로 결합
    text_ids = []
    text_feat_tensor = None
    
    for video_id, text_feat in text_features.items():
        if text_feat is not None:
            if text_feat_tensor is None:
                text_feat_tensor = text_feat
            else:
                text_feat_tensor = torch.cat([text_feat_tensor, text_feat], dim=0)
            
            # 텍스트 ID 생성 (video_id + index)
            for i in range(text_feat.shape[0]):
                text_ids.append(f"{video_id}_{i}")
    
    # L2 정규화
    if normalize:
        video_feat_tensor = torch.nn.functional.normalize(video_feat_tensor, dim=-1)
        text_feat_tensor = torch.nn.functional.normalize(text_feat_tensor, dim=-1)
    
    # Similarity 매트릭스 계산
    # i2t_scores: 각 비디오에 대한 모든 텍스트의 similarity
    i2t_scores = video_feat_tensor @ text_feat_tensor.T
    
    # t2i_scores: 각 텍스트에 대한 모든 비디오의 similarity  
    t2i_scores = text_feat_tensor @ video_feat_tensor.T
    
    return i2t_scores, t2i_scores, video_ids, text_ids

def save_similarity_results(i2t_scores, t2i_scores, video_ids, text_ids, 
                          video_metadata, text_metadata, output_dir):
    """Similarity 결과 저장"""
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Similarity 매트릭스 저장
    similarity_data = {
        'i2t_scores': i2t_scores.detach().numpy(),
        't2i_scores': t2i_scores.detach().numpy(),
        'video_ids': video_ids,
        'text_ids': text_ids,
        'video_metadata': video_metadata,
        'text_metadata': text_metadata
    }
    
    similarity_file = os.path.join(output_dir, "msrvtt_similarity_matrix.pkl")
    with open(similarity_file, 'wb') as f:
        pickle.dump(similarity_data, f)
    
    # 통계 정보
    stats = {
        'num_videos': len(video_ids),
        'num_texts': len(text_ids),
        'i2t_shape': list(i2t_scores.shape),
        't2i_shape': list(t2i_scores.shape),
        'similarity_stats': {
            'i2t_mean': float(i2t_scores.mean()),
            'i2t_std': float(i2t_scores.std()),
            'i2t_min': float(i2t_scores.min()),
            'i2t_max': float(i2t_scores.max()),
            't2i_mean': float(t2i_scores.mean()),
            't2i_std': float(t2i_scores.std()),
            't2i_min': float(t2i_scores.min()),
            't2i_max': float(t2i_scores.max())
        }
    }
    
    stats_file = os.path.join(output_dir, "similarity_stats.json")
    with open(stats_file, 'w') as f:
        json.dump(stats, f, indent=2)
    
    logger.info(f"✅ Similarity matrix saved to {similarity_file}")
    logger.info(f"📊 Shape: {i2t_scores.shape[0]} videos × {i2t_scores.shape[1]} texts")
    logger.info(f"📈 Similarity range: [{stats['similarity_stats']['i2t_min']:.3f}, {stats['similarity_stats']['i2t_max']:.3f}]")
    
    return similarity_file, stats_file

def main():
    # 명령행 인자 처리
    import sys
    if len(sys.argv) < 2:
        print("Usage: python fast_similarity_computation.py <feature_dir> [output_dir]")
        print("Example: python fast_similarity_computation.py ./outputs/msrvtt_features/extract_all_features_20250920_203358/msrvtt_features/")
        sys.exit(1)
    
    # 특징 파일 디렉토리
    feature_dir = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "./outputs/msrvtt_similarities/fast_computation"
    
    logger.info("Loading extracted features...")
    video_features, video_metadata, text_features, text_metadata, stats = load_features(feature_dir)
    
    logger.info(f"Loaded features:")
    logger.info(f"  Videos: {len(video_features)}")
    logger.info(f"  Texts: {sum(len(metadata) for metadata in text_metadata.values())}")
    
    logger.info("Computing similarity matrices...")
    i2t_scores, t2i_scores, video_ids, text_ids = compute_similarity_matrices(
        video_features, text_features, normalize=True
    )
    
    logger.info("Saving results...")
    similarity_file, stats_file = save_similarity_results(
        i2t_scores, t2i_scores, video_ids, text_ids,
        video_metadata, text_metadata, output_dir
    )
    
    logger.info("🎉 Fast similarity computation completed!")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()
