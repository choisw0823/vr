#!/usr/bin/env python3
"""
MSR-VTT 비디오와 유사도 매트릭스를 사용해 ranking dataset을 생성합니다.

요구사항:
- 각 텍스트당 top 10 유사도 비디오 중 5개 선택
- 텍스트당 3개 조합 생성 
- 하나는 반드시 top1~top5에서 선택 (정답 포함)
- 정답이 아닌 비디오가 더 높은 유사도를 가지는 경우 제외
- 각 비디오에서 12프레임 균등 추출
"""

import os
import json
import pickle
import random
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Any
from collections import defaultdict
import argparse
from tqdm import tqdm


def load_similarity_matrix(pkl_path: str) -> Tuple[np.ndarray, List[str], List[str]]:
    """유사도 매트릭스와 메타데이터를 로드합니다."""
    print(f"Loading similarity matrix from {pkl_path}")
    with open(pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    print(f"Data type: {type(data)}")
    if isinstance(data, dict):
        print(f"Data keys: {list(data.keys())}")
    
    # 데이터 구조 확인
    if isinstance(data, dict):
        similarity_matrix = data.get('similarity_matrix', data.get('similarities', data))
        
        # text_data에서 caption 추출
        if 'text_data' in data:
            text_queries = [item['caption'] for item in data['text_data']]
        else:
            text_queries = data.get('text_queries', data.get('texts'))
            
        # video_data에서 video_id 추출  
        if 'video_data' in data:
            video_ids = [item['video_id'] for item in data['video_data']]
        else:
            video_ids = data.get('video_ids', data.get('videos'))
    else:
        # 단순히 매트릭스만 있는 경우
        similarity_matrix = data
        text_queries = None
        video_ids = None
    
    # numpy array인지 확인
    if not isinstance(similarity_matrix, np.ndarray):
        similarity_matrix = np.array(similarity_matrix)
    
    print(f"Similarity matrix shape: {similarity_matrix.shape}")
    
    # 메타데이터가 없으면 기본값 생성
    if text_queries is None:
        text_queries = [f"text_{i}" for i in range(similarity_matrix.shape[0])]
        print(f"Generated {len(text_queries)} text queries")
    else:
        print(f"Number of texts: {len(text_queries)}")
    
    if video_ids is None:
        video_ids = [f"video{i}" for i in range(similarity_matrix.shape[1])]
        print(f"Generated {len(video_ids)} video IDs")
    else:
        print(f"Number of videos: {len(video_ids)}")
    
    return similarity_matrix, text_queries, video_ids


def extract_frames_uniform(video_path: str, num_frames: int = 12, output_base_dir: str = "./data/frames") -> List[str]:
    """비디오에서 균등 간격으로 프레임을 추출하고 저장합니다."""
    if not os.path.exists(video_path):
        raise FileNotFoundError(f"Video not found: {video_path}")
    
    # 출력 디렉토리 설정 (r1/data/frames 안에 저장)
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    output_dir = os.path.join(output_base_dir, video_name)
    os.makedirs(output_dir, exist_ok=True)
    
    # 이미 추출된 프레임이 있는지 확인 (절대경로로 반환)
    expected_frames = [os.path.abspath(os.path.join(output_dir, f"frame_{i:02d}.jpg")) for i in range(num_frames)]
    if all(os.path.exists(f) for f in expected_frames):
        return expected_frames
    
    # 비디오 로드
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video: {video_path}")
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames == 0:
        raise RuntimeError(f"Video has no frames: {video_path}")
    
    # 균등 간격으로 프레임 인덱스 계산
    if total_frames >= num_frames:
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # 프레임이 부족하면 마지막 프레임 반복
        frame_indices = list(range(total_frames))
        frame_indices.extend([total_frames - 1] * (num_frames - total_frames))
    
    extracted_paths = []
    for i, frame_idx in enumerate(frame_indices):
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if ret:
            output_path = os.path.join(output_dir, f"frame_{i:02d}.jpg")
            cv2.imwrite(output_path, frame)
            # 절대경로로 저장
            extracted_paths.append(os.path.abspath(output_path))
    
    cap.release()
    return extracted_paths


def get_top_similar_videos(similarity_row: np.ndarray, video_ids: List[str], k: int = 10) -> List[Tuple[str, float]]:
    """유사도 기준 상위 k개 비디오를 반환합니다."""
    top_indices = np.argsort(similarity_row)[::-1][:k]
    return [(video_ids[idx], similarity_row[idx]) for idx in top_indices]


def sample_video_combinations(
    top_videos: List[Tuple[str, float]], 
    ground_truth_video: str,
    num_combinations: int = 3,
    videos_per_combination: int = 5
) -> List[List[Tuple[str, float]]]:
    """비디오 조합을 샘플링합니다."""
    combinations = []
    
    # ground truth의 유사도 점수 찾기
    gt_score = None
    for video_id, score in top_videos:
        if video_id == ground_truth_video:
            gt_score = score
            break
    
    if gt_score is None:
        # print(f"Warning: Ground truth video {ground_truth_video} not in top videos")
        return []
    
    # 정답보다 유사도가 높은 비디오들 필터링 (학습 안정성)
    valid_videos = [(vid, score) for vid, score in top_videos if score <= gt_score or vid == ground_truth_video]
    
    if len(valid_videos) < videos_per_combination:
        # print(f"Warning: Not enough valid videos ({len(valid_videos)} < {videos_per_combination})")
        return []
    
    for i in range(num_combinations):
        # 첫 번째 조합은 top1~top5에서 선택
        if i == 0:
            top5_videos = valid_videos[:5]
            if ground_truth_video not in [v[0] for v in top5_videos]:
                # 정답이 top5에 없으면 강제로 포함
                combination = top5_videos[:4] + [(ground_truth_video, gt_score)]
            else:
                combination = top5_videos
        else:
            # 나머지는 랜덤 샘플링 (정답 포함)
            other_videos = [v for v in valid_videos if v[0] != ground_truth_video]
            if len(other_videos) >= videos_per_combination - 1:
                sampled = random.sample(other_videos, videos_per_combination - 1)
                combination = sampled + [(ground_truth_video, gt_score)]
            else:
                combination = other_videos + [(ground_truth_video, gt_score)]
        
        # 정확히 5개가 되도록 조정
        if len(combination) > videos_per_combination:
            combination = combination[:videos_per_combination]
        elif len(combination) < videos_per_combination:
            # 부족하면 valid_videos에서 추가
            needed = videos_per_combination - len(combination)
            existing_ids = {v[0] for v in combination}
            additional = [v for v in valid_videos if v[0] not in existing_ids][:needed]
            combination.extend(additional)
        
        # 유사도 내림차순 정렬
        combination.sort(key=lambda x: x[1], reverse=True)
        combinations.append(combination)
    
    return combinations


def create_dataset_sample(
    text_query: str,
    text_idx: int,
    combination_idx: int,
    video_combination: List[Tuple[str, float]],
    ground_truth_video: str,
    video_dir: str,
    frames_per_candidate: int = 12,
    output_base_dir: str = "./data/frames"
) -> Dict[str, Any]:
    """단일 샘플을 생성합니다."""
    
    candidates = []
    gt_video_id = None
    
    for cand_id, (video_id, sim_score) in enumerate(video_combination, 1):
        # 비디오 파일 경로
        video_path = os.path.join(video_dir, f"{video_id}.mp4")
        
        try:
            # 프레임 추출
            frame_paths = extract_frames_uniform(video_path, frames_per_candidate, output_base_dir)
            
            # 정답 비디오 확인
            is_correct = (video_id == ground_truth_video)
            if is_correct:
                gt_video_id = cand_id
            
            candidate = {
                "candidate_id": cand_id,
                "video_id": video_id,
                "is_correct": is_correct,
                "sim_score": float(sim_score),
                "frames": frame_paths
            }
            candidates.append(candidate)
            
        except Exception as e:
            print(f"Error processing video {video_id}: {e}")
            continue
    
    if gt_video_id is None:
        raise ValueError(f"Ground truth video {ground_truth_video} not found in combination")
    
    # candidates를 랜덤으로 섞기
    random.shuffle(candidates)
    
    # 섞인 후의 candidate_id를 다시 할당 (1부터 시작)
    for i, candidate in enumerate(candidates, 1):
        if candidate["is_correct"]:
            gt_video_id = i  # 정답의 새로운 candidate_id
        candidate["candidate_id"] = i
    
    # 섞인 candidates를 유사도 기준으로 정렬해서 order_sim_desc 생성
    sorted_candidates = sorted(candidates, key=lambda x: x["sim_score"], reverse=True)
    order_sim_desc = [c["candidate_id"] for c in sorted_candidates]
    
    sample = {
        "sample_id": f"{text_idx}_{combination_idx}",
        "query": text_query,
        "num_candidates": len(candidates),
        "frames_per_candidate": frames_per_candidate,
        "candidates": candidates,  # 랜덤으로 섞인 순서
        "ground_truth": {
            "order_sim_desc": order_sim_desc,  # 유사도 기준 내림차순 순서
            "correct_cand_id": gt_video_id     # 섞인 후의 정답 ID
        },
        "case_type": f"combination_{combination_idx}",
        "seed": random.randint(1000, 99999)
    }
    
    return sample


def create_ranking_dataset(
    similarity_pkl_path: str,
    video_dir: str,
    output_path: str,
    max_texts: int = None,
    frames_per_candidate: int = 12,
    combinations_per_text: int = 3,
    videos_per_combination: int = 5,
    output_base_dir: str = "./data/frames"
):
    """전체 ranking dataset을 생성합니다."""
    
    # 유사도 매트릭스 로드
    similarity_matrix, text_queries, video_ids = load_similarity_matrix(similarity_pkl_path)
    
    # Ground truth 매핑 - video_text_mapping 사용
    with open(similarity_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    video_text_mapping = None
    if isinstance(data, dict) and 'video_text_mapping' in data:
        video_text_mapping = data['video_text_mapping']
        print(f"Using video_text_mapping with {len(video_text_mapping)} entries")
    
    def get_ground_truth_video(text_idx: int) -> str:
        if video_text_mapping and text_idx < len(video_text_mapping):
            video_idx = video_text_mapping[text_idx]
            if video_idx < len(video_ids):
                return video_ids[video_idx]
        # 기본값: 인덱스 매핑
        return video_ids[text_idx % len(video_ids)]
    
    samples = []
    
    # 처리할 텍스트 수 제한
    num_texts = min(len(text_queries), max_texts) if max_texts else len(text_queries)
    
    print(f"Processing {num_texts} texts...")
    
    for text_idx in tqdm(range(num_texts), desc="Creating samples"):
        text_query = text_queries[text_idx]
        ground_truth_video = get_ground_truth_video(text_idx)
        
        # 상위 10개 유사 비디오 추출
        top_videos = get_top_similar_videos(similarity_matrix[text_idx], video_ids, k=20)
        
        # 비디오 조합 샘플링
        combinations = sample_video_combinations(
            top_videos, 
            ground_truth_video,
            num_combinations=combinations_per_text,
            videos_per_combination=videos_per_combination
        )
        
        if not combinations:
            # print(f"Skipping text {text_idx}: no valid combinations")
            continue
        
        # 각 조합에 대해 샘플 생성
        for comb_idx, combination in enumerate(combinations):
            try:
                sample = create_dataset_sample(
                    text_query,
                    text_idx,
                    comb_idx,
                    combination,
                    ground_truth_video,
                    video_dir,
                    frames_per_candidate,
                    output_base_dir
                )
                samples.append(sample)
            except Exception as e:
                print(f"Error creating sample {text_idx}_{comb_idx}: {e}")
                continue
    
    # JSON 저장
    dataset = {
        "metadata": {
            "frames_per_candidate": frames_per_candidate,
            "default_num_candidates": videos_per_combination,
            "total_samples": len(samples),
            "source_similarity_matrix": similarity_pkl_path,
            "source_video_dir": video_dir
        },
        "samples": samples
    }
    
    print(f"Generated {len(samples)} samples")
    print(f"Saving dataset to {output_path}")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(dataset, f, indent=2, ensure_ascii=False)
    
    print("Dataset creation completed!")


def main():
    parser = argparse.ArgumentParser(description="Create ranking dataset from MSR-VTT videos and similarity matrix")
    parser.add_argument("--similarity_pkl", 
                       default="/home/work/smoretalk/seo/reranking/new_d/InternVideo/InternVideo2/multi_modality/outputs/msrvtt_features/multimodal_similarities_unique_20250927_094405/multimodal_similarities_unique/similarity_matrix.pkl",
                       help="Path to similarity matrix pickle file")
    parser.add_argument("--video_dir",
                       default="/home/work/smoretalk/seo/reranking/new_d/MSR-VTT/video",
                       help="Directory containing MSR-VTT videos")
    parser.add_argument("--output",
                       default="./data/msrvtt_ranking_dataset.json",
                       help="Output JSON file path")
    parser.add_argument("--max_texts", type=int, default=9000,
                       help="Maximum number of texts to process (for testing)")
    parser.add_argument("--frames_per_candidate", type=int, default=12,
                       help="Number of frames to extract per video")
    parser.add_argument("--combinations_per_text", type=int, default=1,
                       help="Number of combinations per text")
    parser.add_argument("--videos_per_combination", type=int, default=5,
                       help="Number of videos per combination")
    
    args = parser.parse_args()
    
    # 경로 확인
    if not os.path.exists(args.similarity_pkl):
        print(f"Error: Similarity matrix file not found: {args.similarity_pkl}")
        return
    
    if not os.path.exists(args.video_dir):
        print(f"Error: Video directory not found: {args.video_dir}")
        return
    
    # 출력 디렉토리 생성
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    
    # 랜덤 시드 설정
    random.seed(42)
    np.random.seed(42)
    
    # 데이터셋 생성
    create_ranking_dataset(
        similarity_pkl_path=args.similarity_pkl,
        video_dir=args.video_dir,
        output_path=args.output,
        max_texts=args.max_texts,
        frames_per_candidate=args.frames_per_candidate,
        combinations_per_text=args.combinations_per_text,
        videos_per_combination=args.videos_per_combination,
        output_base_dir="./data/frames"
    )


if __name__ == "__main__":
    main()
