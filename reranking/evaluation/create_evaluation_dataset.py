#!/usr/bin/env python3
"""
평가 데이터셋 생성 코드
유사도 기반 상위 10개에서 5개 후보를 선택해 reranking 평가 데이터 생성
"""

import os
import json
import pickle
import numpy as np
from typing import List, Dict, Any
from tqdm import tqdm
import argparse


class EvaluationDatasetCreator:
    def __init__(self, similarity_data_path: str, video_base_path: str):
        """
        평가 데이터셋 생성기 초기화
        
        Args:
            similarity_data_path: 유사도 행렬 pickle 파일 경로
            video_base_path: 비디오 파일들이 있는 기본 경로
        """
        self.similarity_data_path = similarity_data_path
        self.video_base_path = video_base_path
        self.load_data()
    
    def load_data(self):
        """유사도 데이터 로드"""
        print(f"Loading similarity data from: {self.similarity_data_path}")
        
        with open(self.similarity_data_path, 'rb') as f:
            data = pickle.load(f)
        
        self.similarity_matrix = data['similarity_matrix']
        self.text_data = data['text_data']
        self.video_data = data['video_data']
        
        print(f"Loaded {len(self.text_data)} texts, {len(self.video_data)} videos")
        print(f"Similarity matrix shape: {self.similarity_matrix.shape}")
    
    def create_evaluation_samples(self, max_samples: int = None) -> List[Dict[str, Any]]:
        """평가 샘플 생성"""
        if max_samples:
            text_data = self.text_data[:max_samples]
        else:
            text_data = self.text_data
        
        print(f"Creating evaluation samples for {len(text_data)} texts...")
        
        evaluation_samples = []
        
        for i, text_info in enumerate(tqdm(text_data, desc="Creating samples")):
            try:
                # 쿼리와 정답 비디오
                query = text_info['caption']
                correct_video_id = text_info['video_id']
                
                # 정답 비디오 인덱스 찾기
                correct_video_idx = None
                for j, video_info in enumerate(self.video_data):
                    if video_info['video_id'] == correct_video_id:
                        correct_video_idx = j
                        break
                
                if correct_video_idx is None:
                    print(f"Warning: Correct video not found for text {i}")
                    continue
                
                # 유사도 기반 상위 10개 후보 선택 (정답 포함 보장)
                text_similarities = self.similarity_matrix[:, i]  # i번째 텍스트에 대한 모든 비디오 유사도
                top10_indices = np.argsort(text_similarities)[-10:][::-1]  # 상위 10개
                
                # 정답이 상위 10개에 없으면 추가
                if correct_video_idx not in top10_indices:
                    # 가장 낮은 점수 비디오를 정답으로 교체
                    top10_indices[-1] = correct_video_idx
                
                # 유사도 점수 추출
                top10_similarities = [float(text_similarities[idx]) for idx in top10_indices]
                
                # 3가지 케이스 생성

                for case_idx in range(3):
                    rng = np.random.default_rng(seed=i * 3 + case_idx)  # 재현성 보장

                    if case_idx == 0:
                        # 1) top5 뽑고, 정답이 없으면 5번째를 정답으로 교체
                        candidate_indices = top10_indices[:5].copy()
                        if correct_video_idx not in candidate_indices:
                            candidate_indices[-1] = correct_video_idx
                            # 혹시 모를 중복 방지 + 길이 5 보장
                            seen, dedup = set(), []
                            for x in candidate_indices:
                                if x not in seen:
                                    dedup.append(x); seen.add(x)
                            for x in top10_indices:
                                if len(dedup) == 5: break
                                if x not in seen:
                                    dedup.append(x); seen.add(x)
                            candidate_indices = np.array(dedup[:5], dtype=top10_indices.dtype)

                        # 2) ★★★ 모델 입력용 순서는 '항상 랜덤' ★★★
                        rng.shuffle(candidate_indices)

                        case_type = "top5"
                    else:
                        other_indices = [idx for idx in top10_indices if idx != correct_video_idx]
                        selected_others = rng.choice(other_indices, 4, replace=False)
                        candidate_indices = np.array([correct_video_idx] + list(selected_others))
                        rng.shuffle(candidate_indices)  # 이미 랜덤 유지
                        case_type = f"random_{case_idx}"

                    
                    top10_rank_map = {int(idx): int(np.where(top10_indices == idx)[0][0]) + 1
                                    for idx in candidate_indices}

                    # 비디오 정보 수집 (모델/평가에 쓰는 '표시 순서' = candidate_id 1..5)
                    videos = []
                    for j, video_idx in enumerate(candidate_indices):
                        video_info = self.video_data[video_idx]
                        video_path = os.path.join(self.video_base_path, f"{video_info['video_id']}.mp4")
                        videos.append({
                            "candidate_id": j + 1,  # [1]..[5] 와 1:1 매칭
                            "video_id": video_info['video_id'],
                            "video_path": video_path,
                            "similarity_score": float(text_similarities[video_idx]),
                            "is_correct": video_info['video_id'] == correct_video_id,
                            "gt_rank": top10_rank_map[int(video_idx)]  # top-10 내 랭크(메타)
                        })

                    # ==== 여기서부터 '평가 기준' GT 순열 생성 ====
                    # 정답의 candidate_id (1..5)
                    correct_pos = next(v["candidate_id"] for v in videos if v["is_correct"])

                    # 나머지 후보를 gt_rank로 정렬
                    others = [v for v in videos if v["candidate_id"] != correct_pos]
                    others_sorted_by_true = sorted(others, key=lambda x: x["gt_rank"])  # 1(최상) -> 크게(하위)

                    # 최종 GT 순열: 정답 먼저 + (진짜 랭크 좋은 순서)
                    ground_truth_order = [correct_pos] + [v["candidate_id"] for v in others_sorted_by_true]

                    # ============================================

                    sample = {
                        "sample_id": f"{i}_{case_idx}",
                        "text_idx": i,
                        "case_idx": case_idx,
                        "case_type": case_type,       # "top5" / "random_1" / "random_2"
                        "query": query,
                        "correct_video_id": correct_video_id,

                        # ★ 평가 기준 GT: [1]..[5] 공간의 순열 (예: [4,1,2,5,3])
                        "ground_truth_ranking": ground_truth_order,

                        "videos": videos,
                        "metadata": {
                            "top10_video_ids": [self.video_data[idx]['video_id'] for idx in top10_indices],
                            "top10_similarities": [float(text_similarities[idx]) for idx in top10_indices]
                        }
                    }
                    evaluation_samples.append(sample)

            except Exception as e:
                print(f"Error processing text {i}: {e}")
                continue
        
        return evaluation_samples
    
    def save_evaluation_dataset(self, samples: List[Dict[str, Any]], output_path: str):
        """평가 데이터셋 저장"""
        print(f"Saving {len(samples)} evaluation samples to {output_path}")
        
        # 메타데이터 생성
        dataset = {
            "metadata": {
                "total_samples": len(samples),
                "texts_count": len(set(sample["text_idx"] for sample in samples)),
                "cases_per_text": 3,
                "candidates_per_case": 5,
                "similarity_data_path": self.similarity_data_path,
                "video_base_path": self.video_base_path,
                "description": "Evaluation dataset for video-text reranking with similarity-based candidate selection"
            },
            "samples": samples
        }
        
        # JSON 저장
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(dataset, f, ensure_ascii=False, indent=2)
        
        print(f"Dataset saved successfully!")
        print(f"File size: {os.path.getsize(output_path) / 1024 / 1024:.2f} MB")
    
    def analyze_dataset(self, samples: List[Dict[str, Any]]):
        """데이터셋 분석"""
        print("\n=== Dataset Analysis ===")
        
        total_samples = len(samples)
        texts_count = len(set(sample["text_idx"] for sample in samples))
        
        print(f"Total samples: {total_samples}")
        print(f"Unique texts: {texts_count}")
        print(f"Cases per text: {total_samples // texts_count}")
        
        # 케이스 타입별 분석
        case_types = {}
        for sample in samples:
            case_type = sample["case_type"]
            case_types[case_type] = case_types.get(case_type, 0) + 1
        
        print(f"Case type distribution: {case_types}")
        
        # 정답 위치 분석
        correct_positions = []
        for sample in samples:
            for i, video in enumerate(sample["videos"]):
                if video["is_correct"]:
                    correct_positions.append(i + 1)  # 1-based position
                    break
        
        print(f"Correct answer positions:")
        for pos in range(1, 6):
            count = correct_positions.count(pos)
            print(f"  Position {pos}: {count} ({count/len(correct_positions)*100:.1f}%)")
        
        # 유사도 점수 분석
        similarities = []
        for sample in samples:
            for video in sample["videos"]:
                similarities.append(video["similarity_score"])
        
        print(f"Similarity scores:")
        print(f"  Min: {np.min(similarities):.4f}")
        print(f"  Max: {np.max(similarities):.4f}")
        print(f"  Mean: {np.mean(similarities):.4f}")
        print(f"  Std: {np.std(similarities):.4f}")


def main():
    parser = argparse.ArgumentParser(description='Create evaluation dataset for reranking')
    parser.add_argument('--similarity_data', type=str, required=True, 
                       help='Path to similarity matrix pickle file')
    parser.add_argument('--video_base_path', type=str, required=True,
                       help='Base path to video files')
    parser.add_argument('--output_path', type=str, default='evaluation_dataset.json',
                       help='Output path for evaluation dataset')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of texts to process')
    
    args = parser.parse_args()
    
    # 데이터셋 생성기 초기화
    creator = EvaluationDatasetCreator(args.similarity_data, args.video_base_path)
    
    # 평가 샘플 생성
    samples = creator.create_evaluation_samples(args.max_samples)
    
    # 데이터셋 분석
    creator.analyze_dataset(samples)
    
    # 데이터셋 저장
    creator.save_evaluation_dataset(samples, args.output_path)
    
    print(f"\nEvaluation dataset creation completed!")
    print(f"Dataset saved to: {args.output_path}")


if __name__ == "__main__":
    main()
