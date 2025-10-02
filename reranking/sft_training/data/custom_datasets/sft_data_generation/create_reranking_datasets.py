#!/usr/bin/env python3
"""
Reranking을 위한 학습 데이터 생성
9000 x 9000 트레인 데이터에 대해 Easy, Medium, Hard 모드로 데이터셋 생성
"""

import pickle
import json
import numpy as np
import random
from tqdm import tqdm

def load_data():
    """데이터 로드 및 트레인 데이터 필터링"""
    print("similarity_matrix.pkl 파일을 로드하는 중...")
    with open('/home/work/smoretalk/seo/reranking/new_d/InternVideo/InternVideo2/multi_modality/outputs/msrvtt_features/multimodal_similarities_unique_20250927_094405/multimodal_similarities_unique/similarity_matrix.pkl', 'rb') as f:
        data = pickle.load(f)
    
    similarity_matrix = data['similarity_matrix']
    video_data = data['video_data']
    text_data = data['text_data']
    
    # 트레인 데이터 인덱스 필터링
    # split 키가 있으면 train만 사용, 없으면 전체 사용
    if 'split' in text_data[0]:
        train_indices = [i for i, text in enumerate(text_data) if text['split'] == 'train']
        train_similarity_matrix = similarity_matrix[np.ix_(train_indices, train_indices)]
    else:
        # split 키가 없으면 전체 데이터 사용 (unique 데이터의 경우)
        train_indices = list(range(len(text_data)))
        train_similarity_matrix = similarity_matrix
    
    print(f"데이터 로드 완료:")
    print(f"- 전체 유사도 매트릭스 크기: {similarity_matrix.shape}")
    print(f"- 트레인 데이터 개수: {len(train_indices)}")
    print(f"- 트레인 유사도 매트릭스 크기: {train_similarity_matrix.shape}")
    
    return train_similarity_matrix, video_data, text_data, train_indices

def create_easy_dataset(train_similarity_matrix, video_data, text_data, train_indices):
    """
    Easy 모드: 분포를 넓게 해서 텍스트별로 5개의 비디오 추출 (음수 포함)
    - 정답 1개 + 랜덤 4개 (전체 분포에서)
    """
    print("\n=== Easy 모드 데이터셋 생성 ===")
    
    easy_data = {
        "metadata": {
            "mode": "easy",
            "description": "넓은 분포에서 5개 비디오 추출 (음수 유사도 포함)",
            "videos_per_text": 5,
            "total_texts": len(train_indices),
            "sampling_strategy": "전체 분포에서 랜덤 샘플링"
        },
        "training_samples": []
    }
    
    random.seed(42)  # 재현 가능한 결과를 위해
    
    for local_idx, text_idx in enumerate(tqdm(train_indices, desc="Easy 모드 처리 중")):
        text_info = text_data[text_idx]
        text_similarities = train_similarity_matrix[local_idx, :]
        
        # 정답 비디오 (자기 자신)
        correct_video_idx = local_idx
        
        # 정답이 아닌 비디오들 중에서 랜덤 4개 선택
        other_indices = [i for i in range(len(train_indices)) if i != correct_video_idx]
        random_indices = random.sample(other_indices, 4)
        
        # 5개 비디오 조합 (정답 1개 + 랜덤 4개)
        selected_indices = [correct_video_idx] + random_indices
        random.shuffle(selected_indices)  # 순서 섞기
        
        videos = []
        for video_local_idx in selected_indices:
            video_global_idx = train_indices[video_local_idx]
            video_info = video_data[video_global_idx]
            similarity_score = float(text_similarities[video_local_idx])
            
            videos.append({
                "video_id": video_info['video_id'],
                "video_path": video_info['video_path'],
                "similarity_score": similarity_score,
                "is_correct": video_info['video_id'] == text_info['video_id']
            })
        
        sample = {
            "text_id": text_info['video_id'],
            "caption": text_info['caption'],
            "videos": videos
        }
        
        easy_data["training_samples"].append(sample)
    
    return easy_data

def create_medium_dataset(train_similarity_matrix, video_data, text_data, train_indices):
    """
    Medium 모드: 텍스트와 관련 있는 비디오들 포함
    - 정답 1개 + 상위 2개 + 중간 1개 + 하위 1개
    """
    print("\n=== Medium 모드 데이터셋 생성 ===")
    
    medium_data = {
        "metadata": {
            "mode": "medium",
            "description": "텍스트와 관련성이 있는 비디오들로 구성",
            "videos_per_text": 5,
            "total_texts": len(train_indices),
            "sampling_strategy": "정답 1개 + 상위 2개 + 중간 1개 + 하위 1개"
        },
        "training_samples": []
    }
    
    for local_idx, text_idx in enumerate(tqdm(train_indices, desc="Medium 모드 처리 중")):
        text_info = text_data[text_idx]
        text_similarities = train_similarity_matrix[local_idx, :]
        
        # 유사도 순으로 정렬
        sorted_indices = np.argsort(text_similarities)[::-1]  # 내림차순
        
        # 정답 비디오 위치 찾기
        correct_video_idx = local_idx
        correct_rank = np.where(sorted_indices == correct_video_idx)[0][0]
        
        # 비디오 선택 전략
        selected_indices = [correct_video_idx]  # 정답 포함
        
        # 상위 2개 (정답 제외)
        top_candidates = [idx for idx in sorted_indices[:20] if idx != correct_video_idx]
        selected_indices.extend(top_candidates[:2])
        
        # 중간 범위에서 1개
        mid_start = len(train_indices) // 3
        mid_end = 2 * len(train_indices) // 3
        mid_candidates = [idx for idx in sorted_indices[mid_start:mid_end] if idx not in selected_indices]
        if mid_candidates:
            selected_indices.append(random.choice(mid_candidates))
        
        # 하위 범위에서 1개
        low_candidates = [idx for idx in sorted_indices[-1000:] if idx not in selected_indices]
        if low_candidates:
            selected_indices.append(random.choice(low_candidates))
        
        # 부족한 경우 추가
        while len(selected_indices) < 5:
            remaining = [idx for idx in range(len(train_indices)) if idx not in selected_indices]
            if remaining:
                selected_indices.append(random.choice(remaining))
            else:
                break
        
        random.shuffle(selected_indices)  # 순서 섞기
        
        videos = []
        for video_local_idx in selected_indices[:5]:
            video_global_idx = train_indices[video_local_idx]
            video_info = video_data[video_global_idx]
            similarity_score = float(text_similarities[video_local_idx])
            
            videos.append({
                "video_id": video_info['video_id'],
                "video_path": video_info['video_path'],
                "similarity_score": similarity_score,
                "is_correct": video_info['video_id'] == text_info['video_id']
            })
        
        sample = {
            "text_id": text_info['video_id'],
            "caption": text_info['caption'],
            "videos": videos
        }
        
        medium_data["training_samples"].append(sample)
    
    return medium_data

def create_hard_dataset(train_similarity_matrix, video_data, text_data, train_indices):
    """
    Hard 모드: 상위 20개에서 5개 선택
    - 텍스트와 유사도가 높은 상위 10개 비디오에서 5개 추출
    """
    print("\n=== Hard 모드 데이터셋 생성 ===")
    
    hard_data = {
        "metadata": {
            "mode": "hard",
            "description": "상위 20개 유사한 비디오에서 5개 선택",
            "videos_per_text": 5,
            "total_texts": len(train_indices),
            "sampling_strategy": "상위 20개에서 5개 선택"
        },
        "training_samples": []
    }
    
    for local_idx, text_idx in enumerate(tqdm(train_indices, desc="Hard 모드 처리 중")):
        text_info = text_data[text_idx]
        text_similarities = train_similarity_matrix[local_idx, :]
        
        # 상위 20개 비디오 인덱스
        top20_indices = np.argsort(text_similarities)[::-1][:10]
        
        # 상위 20개에서 5개 선택 (정답 포함 보장)
        correct_video_idx = local_idx
        if correct_video_idx in top20_indices:
            # 정답이 상위 20개에 있는 경우
            other_indices = [idx for idx in top20_indices if idx != correct_video_idx]
            selected_indices = [correct_video_idx] + random.sample(other_indices, min(4, len(other_indices)))
        else:
            # 정답이 상위 20개에 없는 경우 (정답 + 상위 4개)
            selected_indices = [correct_video_idx] + list(top20_indices[:4])
        
        # 부족한 경우 상위 20개에서 추가
        while len(selected_indices) < 5 and len(selected_indices) < len(top20_indices):
            remaining = [idx for idx in top20_indices if idx not in selected_indices]
            if remaining:
                selected_indices.append(remaining[0])
            else:
                break
        
        random.shuffle(selected_indices)  # 순서 섞기
        
        videos = []
        for video_local_idx in selected_indices[:5]:
            video_global_idx = train_indices[video_local_idx]
            video_info = video_data[video_global_idx]
            similarity_score = float(text_similarities[video_local_idx])
            
            videos.append({
                "video_id": video_info['video_id'],
                "video_path": video_info['video_path'],
                "similarity_score": similarity_score,
                "is_correct": video_info['video_id'] == text_info['video_id']
            })
        
        sample = {
            "text_id": text_info['video_id'],
            "caption": text_info['caption'],
            "videos": videos
        }
        
        hard_data["training_samples"].append(sample)
    
    return hard_data

def save_dataset(data, filename):
    """데이터셋을 JSON 파일로 저장"""
    print(f"\n{filename}에 저장하는 중...")
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"저장 완료! 파일 크기: {len(json.dumps(data)) / 1024 / 1024:.2f} MB")

def analyze_dataset(data, mode_name):
    """데이터셋 분석"""
    print(f"\n=== {mode_name} 모드 분석 ===")
    
    total_samples = len(data["training_samples"])
    correct_counts = []
    similarity_ranges = []
    
    for sample in data["training_samples"]:
        correct_count = sum(1 for video in sample["videos"] if video["is_correct"])
        correct_counts.append(correct_count)
        
        similarities = [video["similarity_score"] for video in sample["videos"]]
        similarity_ranges.append({
            "min": min(similarities),
            "max": max(similarities),
            "mean": np.mean(similarities)
        })
    
    print(f"총 샘플 수: {total_samples}")
    print(f"샘플당 평균 정답 개수: {np.mean(correct_counts):.2f}")
    print(f"유사도 범위:")
    print(f"  최소값: {np.mean([r['min'] for r in similarity_ranges]):.4f}")
    print(f"  최대값: {np.mean([r['max'] for r in similarity_ranges]):.4f}")
    print(f"  평균값: {np.mean([r['mean'] for r in similarity_ranges]):.4f}")

if __name__ == "__main__":
    # 데이터 로드
    train_similarity_matrix, video_data, text_data, train_indices = load_data()
    
    # Easy 모드 생성
    easy_data = create_easy_dataset(train_similarity_matrix, video_data, text_data, train_indices)
    save_dataset(easy_data, "reranking_train_easy.json")
    analyze_dataset(easy_data, "Easy")
    
    # Medium 모드 생성
    medium_data = create_medium_dataset(train_similarity_matrix, video_data, text_data, train_indices)
    save_dataset(medium_data, "reranking_train_medium.json")
    analyze_dataset(medium_data, "Medium")
    
    # Hard 모드 생성
    hard_data = create_hard_dataset(train_similarity_matrix, video_data, text_data, train_indices)
    save_dataset(hard_data, "reranking_train_hard.json")
    analyze_dataset(hard_data, "Hard")
    
    print("\n=== 모든 reranking 데이터셋 생성 완료 ===")
    print("생성된 파일:")
    print("- reranking_train_easy.json")
    print("- reranking_train_medium.json") 
    print("- reranking_train_hard.json")
