#!/usr/bin/env python3
"""
생성된 평가 데이터셋을 사용한 Qwen VL 2.5 reranking 평가 코드
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any, Tuple
from tqdm import tqdm
import argparse
from PIL import Image
import re
import cv2
import time

# LLaMA-Factory ChatModel 사용
sys.path.append('/home/work/smoretalk/seo/LLaMA-Factory/src')
from llamafactory.chat.chat_model import ChatModel


class QwenVLDatasetEvaluator:
    def __init__(self, model_path: str, device: str = "cuda"):
        """
        Qwen VL 2.5 모델 초기화 (LLaMA-Factory ChatModel 사용)
        
        Args:
            model_path: 모델 경로
            device: 사용할 디바이스 (사용되지 않음, ChatModel이 자동 처리)
        """
        self.model_path = model_path
        
        print(f"Loading Qwen VL model from: {model_path}")
        
        # ChatModel 설정
        args = {
            "model_name_or_path": model_path,
            "template": "qwen2_vl",
            "infer_backend": "huggingface",
            "trust_remote_code": True
        }
        
        # ChatModel 초기화
        self.chat_model = ChatModel(args)
        
        # 시스템 프롬프트
#         self.system_prompt = """You are RankLLM, a vision-only reranker. You will receive a Query and N candidates. Each candidate is composed of 12 sampled frames.

# Decision rules (concise):
# - Judge relevance ONLY from what is visible in the grids.
# - Down-rank clear mismatches (wrong domain/scene/action). Consider temporal coherence.
# - Tie-breakers (in order): action visibility/close-up > temporal coherence > low occlusion/clutter > overall coverage.

# ABSOLUTE OUTPUT CONTRACT (exactly two blocks; no extra text/markdown):
# <think>
# <content>
# [1] short description of visible content only.
# [2] short description of visible content only.
# [3] ...
# [4] ...
# [5] ...
# </content>

# <notes>
# [1] brief note on alignment/conflict with the query.
# [2] brief note on alignment/conflict with the query.
# [3] ...
# [4] ...
# [5] ...
# </notes>

# <contrast>
# - One-liner: why a higher-ranked item beats a lower-ranked item (because/whereas style).
# - Add a few more one-liners as needed.
# </contrast>

# <rationale>
# 1st=[i] (reason)
# 2nd=[j] (reason)
# 3rd=[k] (reason)
# 4th=[m] (reason)
# 5th=[n] (reason)
# </rationale>
# </think>
# <answer> [i] > [j] > [k] > [m] > [n] </answer>

# Constraints:
# - Inside <think>, use ONLY the section tags <content>, <notes>, <contrast>, <rationale>.
# - Do NOT use any per-candidate XML-like tags (e.g., no <cand .../>); write lines as [N] text...
# - Keep all statements evidence-based from the grids; no speculation.
# - Always produce a total order in <answer> using indices [1]..[N].

# Query: "{query}"

# Candidates:
# {candidates}"""
    system_prompt = """You are RankLLM, a vision-only reranker. You will receive a Query and N candidates. Each candidate is composed of 12 uniform sampled frames from video.

Decision rules (concise):
- Judge relevance ONLY from what is visible in the frames.
- Down-rank clear mismatches (wrong domain/scene/action). Consider temporal coherence.
- Tie-breakers (in order): action visibility/close-up > temporal coherence > low occlusion/clutter > overall coverage.

ABSOLUTE OUTPUT CONTRACT :
<think>
<content>
[1] Describe the content of the video and its relation to the query, aware of temporal sequence and spatial information.
[2] ...
[3] ...
[4] ...
[5] ...
</content>

<contrast>
- One-liner: why a higher-ranked item beats a lower-ranked item (because/whereas style). 
- Add a few more one-liners as needed.
</contrast>

<summary>
1st=[i] (sufficient reasoning)
2nd=[j] (sufficient reasoning)
3rd=[k] (sufficient reasoning)
4th=[m] (sufficient reasoning)
5th=[n] (sufficient reasoning)
</summary>
</think>
<answer> [i] > [j] > [k] > [m] > [n] </answer>

Constraints:
- Inside <think>, use ONLY the section tags <content>, <contrast>, <summary>.
- Do NOT use any per-candidate XML-like tags (e.g., no <cand .../>); write lines as [N] text...
- Keep all statements evidence-based from the grids; no speculation.
- Always produce a total order in <answer> using indices [1]..[N].

Query: "{query}"

Candidates:
{candidates}
"""

    def load_evaluation_dataset(self, dataset_path: str) -> Dict[str, Any]:
        """평가 데이터셋 로드"""
        print(f"Loading evaluation dataset from: {dataset_path}")
        
        with open(dataset_path, 'r', encoding='utf-8') as f:
            dataset = json.load(f)
        
        print(f"Loaded {dataset['metadata']['total_samples']} samples")
        return dataset

    def load_video_frames(self, video_path: str, num_frames: int = 12) -> List[str]:
        """비디오에서 프레임 추출 후 경로 리스트 반환 (캐싱 활용)"""
        try:
            # 캐시 디렉토리 설정
            video_name = os.path.splitext(os.path.basename(video_path))[0]
            cache_dir = f"/home/work/smoretalk/seo/reranking/evaluation/frames/{video_name}"
            
            # 캐시된 프레임 확인
            cached_frames = []
            for i in range(num_frames):
                frame_path = os.path.join(cache_dir, f"frame_{i:02d}.jpg")
                if os.path.exists(frame_path):
                    cached_frames.append(frame_path)
            
            # 모든 프레임이 캐시되어 있으면 반환
            if len(cached_frames) == num_frames:
                return cached_frames
            
            # 캐시 디렉토리 생성
            os.makedirs(cache_dir, exist_ok=True)
            
            # OpenCV로 비디오 읽기
            cap = cv2.VideoCapture(video_path)
            
            # 총 프레임 수
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            if total_frames == 0:
                raise ValueError(f"No frames found in video: {video_path}")
            
            # 프레임 인덱스 계산
            if total_frames >= num_frames:
                indices = np.linspace(0, total_frames-1, num_frames, dtype=int)
            else:
                # 프레임이 부족하면 반복
                indices = list(range(total_frames)) * ((num_frames // total_frames) + 1)
                indices = indices[:num_frames]
            
            frame_paths = []
            
            # 프레임 추출 및 저장
            for i, frame_idx in enumerate(indices):
                frame_path = os.path.join(cache_dir, f"frame_{i:02d}.jpg")
                
                # 이미 캐시된 프레임이 있으면 스킵
                if os.path.exists(frame_path):
                    frame_paths.append(frame_path)
                    continue
                
                # 새로 추출
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                
                if ret:
                    cv2.imwrite(frame_path, frame)
                    frame_paths.append(frame_path)
                else:
                    # 프레임 읽기 실패 시 이전 프레임 복사
                    if frame_paths:
                        import shutil
                        shutil.copy2(frame_paths[-1], frame_path)
                        frame_paths.append(frame_path)
            
            cap.release()
            return frame_paths
            
        except Exception as e:
            print(f"Error loading video {video_path}: {e}")
            # 더미 이미지 생성
            cache_dir = f"/home/work/smoretalk/seo/reranking/evaluation/frames/dummy"
            os.makedirs(cache_dir, exist_ok=True)
            
            dummy_paths = []
            dummy_img = np.zeros((224, 224, 3), dtype=np.uint8)
            
            for i in range(num_frames):
                frame_path = os.path.join(cache_dir, f"dummy_frame_{i:02d}.jpg")
                if not os.path.exists(frame_path):
                    cv2.imwrite(frame_path, dummy_img)
                dummy_paths.append(frame_path)
            
            return dummy_paths

    def create_prompt(self, query: str, num_candidates: int = 5) -> str:
        """프롬프트 생성"""
        candidates = []
        for i in range(num_candidates):
            candidates.append(f"[{i+1}] video: <|vision_start|><image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <|vision_end|>")
        
        candidates_str = "\n".join(candidates)
        return self.system_prompt.format(query=query, candidates=candidates_str)

    def extract_ranking_from_response(self, response: str) -> List[int]:
        """응답에서 순위 추출"""
        try:
            # <answer> 태그에서 순위 추출
            answer_match = re.search(r'<answer>(.*?)</answer>', response, re.DOTALL)
            if answer_match:
                answer_content = answer_match.group(1).strip()
                
                # [숫자] 패턴 찾기
                ranking_matches = re.findall(r'\[(\d+)\]', answer_content)
                if ranking_matches:
                    return [int(x) for x in ranking_matches]
            
            # <answer> 태그가 없으면 전체 응답에서 찾기
            ranking_matches = re.findall(r'\[(\d+)\]', response)
            if ranking_matches:
                return [int(x) for x in ranking_matches]
                
        except Exception as e:
            print(f"Error extracting ranking: {e}")
        
        return []

    def predict_ranking(self, query: str, video_paths: List[str]) -> List[int]:
        """단일 쿼리에 대한 순위 예측"""
        try:
            # 프롬프트 생성
            prompt = self.create_prompt(query, len(video_paths))
            
            # 비디오 프레임 로드
            all_frame_paths = []
            for video_path in video_paths:
                frame_paths = self.load_video_frames(video_path)
                all_frame_paths.extend(frame_paths)
            
            # 메시지 구성
            messages = [
                {
                    "role": "user",
                    "content": prompt
                }
            ]
            
            # ChatModel로 추론
            responses = self.chat_model.chat(
                messages=messages,
                images=all_frame_paths,
                temperature=0.1,
                max_new_tokens=1024
            )
            
            # 응답 추출
            response = responses[0].response_text if responses else ""
            
            # 순위 추출
            ranking = self.extract_ranking_from_response(response)
            
            return ranking
            
        except Exception as e:
            print(f"Error in prediction: {e}")
            return list(range(1, len(video_paths) + 1))  # 기본 순서 반환

    def calculate_ndcg(self, predicted_ranking: List[int], ground_truth_ranking: List[int], k: int = 5) -> float:
        """nDCG@k 계산
        Args:
            predicted_ranking: 모델이 예측한 순위 [3, 1, 2, 4, 5]
            ground_truth_ranking: 실제 순위 [4, 1, 2, 5, 3] (4가 1위)
        """
        def get_relevance_scores(pred_ranking, gt_ranking):
            """각 예측 위치의 관련도 점수 계산"""
            relevance = []
            # GT에서 위치별 점수: 1위=5점, 2위=4점, 3위=3점, 4위=2점, 5위=1점
            for pred_item in pred_ranking:
                if pred_item in gt_ranking:
                    gt_position = gt_ranking.index(pred_item)  # 0-based position in GT
                    relevance_score = max(0, len(gt_ranking) - gt_position)  # 5, 4, 3, 2, 1
                else:
                    relevance_score = 0
                relevance.append(relevance_score)
            return relevance
        
        def dcg_at_k(relevance_scores, k):
            """DCG@k 계산"""
            dcg = 0.0
            for i in range(min(k, len(relevance_scores))):
                if relevance_scores[i] > 0:
                    dcg += relevance_scores[i] / np.log2(i + 2)
            return dcg
        
        # 예측 순위의 관련도 점수
        pred_relevance = get_relevance_scores(predicted_ranking, ground_truth_ranking)
        
        # DCG 계산
        dcg = dcg_at_k(pred_relevance, k)
        
        # IDCG 계산 (이상적인 순위: ground_truth 순서대로)
        ideal_relevance = [len(ground_truth_ranking) - i for i in range(len(ground_truth_ranking))]  # [5, 4, 3, 2, 1]
        idcg = dcg_at_k(ideal_relevance, k)
        
        return dcg / idcg if idcg > 0 else 0.0

    def evaluate(self, dataset_path: str, output_path: str = None, max_samples: int = None):
        """전체 평가 실행"""
        print("Starting evaluation...")
        
        # 데이터셋 로드
        dataset = self.load_evaluation_dataset(dataset_path)
        samples = dataset['samples']
        
        if max_samples:
            samples = samples[:max_samples]
        
        print(f"Evaluating {len(samples)} samples...")
        
        # 예측 및 정답 수집
        predicted_rankings = []
        ground_truth_rankings = []
        results = []
        ndcg_scores = []
        
        # 실시간 결과 저장을 위한 초기 파일 생성
        initial_results = {
            "metadata": {
                "total_samples": len(samples),
                "evaluation_time": time.strftime("%Y-%m-%d %H:%M:%S")
            },
            "results": []
        }
        
        with open(output_path, 'w') as f:
            json.dump(initial_results, f, indent=2)
        
        for sample in tqdm(samples, desc="Evaluating"):
            try:
                # 쿼리와 비디오 경로 추출
                query = sample['query']
                ground_truth_ranking = sample['ground_truth_ranking']
                
                # 비디오 경로들 (candidate_id 순서대로)
                videos = sorted(sample['videos'], key=lambda x: x['candidate_id'])
                video_paths = [video['video_path'] for video in videos]
                
                # 모델 예측
                pred_ranking = self.predict_ranking(query, video_paths)
                
                # 예측이 부족하면 나머지 채우기
                if len(pred_ranking) < 5:
                    missing = [i for i in range(1, 6) if i not in pred_ranking]
                    pred_ranking.extend(missing)
                
                predicted_rankings.append(pred_ranking[:5])
                ground_truth_rankings.append(ground_truth_ranking)
                
                # nDCG 계산
                ndcg = self.calculate_ndcg(pred_ranking[:5], ground_truth_ranking, 5)
                ndcg_scores.append(ndcg)
                
                # 결과 저장
                result = {
                    'sample_id': sample['sample_id'],
                    'text_idx': sample['text_idx'],
                    'case_idx': sample['case_idx'],
                    'case_type': sample['case_type'],
                    'query': query,
                    'correct_video_id': sample['correct_video_id'],
                    'predicted_ranking': pred_ranking[:5],
                    'ground_truth_ranking': ground_truth_ranking,
                    'ndcg': ndcg,
                    'videos': videos
                }
                results.append(result)
                
                # 실시간 결과 저장 (매 샘플마다)
                with open(output_path, 'r') as f:
                    current_data = json.load(f)
                
                current_data['results'].append(result)
                
                with open(output_path, 'w') as f:
                    json.dump(current_data, f, indent=2)
                
            except Exception as e:
                print(f"Error processing sample {sample['sample_id']}: {e}")
                continue
        
        # R@1 계산 - 두 가지 지표
        r1_sim_correct = 0  # 유사도 GT 1위와 일치
        r1_truth_correct = 0  # 실제 정답 비디오를 1위로 예측
        
        for i, (pred_ranking, gt_ranking) in enumerate(zip(predicted_rankings, ground_truth_rankings)):
            if len(pred_ranking) > 0 and len(gt_ranking) > 0:
                # 1) 유사도 GT 1위와 일치
                gt_first = gt_ranking[0]
                if pred_ranking[0] == gt_first:
                    r1_sim_correct += 1
                
                # 2) 실제 정답을 1위로 예측
                # 실제 정답의 candidate_id 찾기
                correct_candidate_id = None
                for video in results[i]['videos']:
                    if video['is_correct']:
                        correct_candidate_id = video['candidate_id']
                        break
                
                if correct_candidate_id and pred_ranking[0] == correct_candidate_id:
                    r1_truth_correct += 1
        
        r1_sim_score = r1_sim_correct / len(predicted_rankings) if predicted_rankings else 0.0
        r1_truth_score = r1_truth_correct / len(predicted_rankings) if predicted_rankings else 0.0
        avg_ndcg = np.mean(ndcg_scores) if ndcg_scores else 0.0
        
        # 케이스별 분석
        case_analysis = {}
        for result in results:
            case_type = result['case_type']
            if case_type not in case_analysis:
                case_analysis[case_type] = {'r1_sim': [], 'r1_truth': [], 'ndcg': []}
            
            # 이 샘플의 R@1 (유사도 GT)
            gt_first = result['ground_truth_ranking'][0]
            pred_r1_sim = 1 if (len(result['predicted_ranking']) > 0 and 
                               result['predicted_ranking'][0] == gt_first) else 0
            
            # 이 샘플의 R@1 (실제 정답)
            correct_candidate_id = None
            for video in result['videos']:
                if video['is_correct']:
                    correct_candidate_id = video['candidate_id']
                    break
            
            pred_r1_truth = 1 if (correct_candidate_id and 
                                 len(result['predicted_ranking']) > 0 and 
                                 result['predicted_ranking'][0] == correct_candidate_id) else 0
            
            case_analysis[case_type]['r1_sim'].append(pred_r1_sim)
            case_analysis[case_type]['r1_truth'].append(pred_r1_truth)
            case_analysis[case_type]['ndcg'].append(result['ndcg'])
        
        # 결과 출력
        print("\n=== Evaluation Results ===")
        print(f"R@1 (Sim GT): {r1_sim_score:.4f} ({r1_sim_score*100:.2f}%)")
        print(f"R@1 (Truth): {r1_truth_score:.4f} ({r1_truth_score*100:.2f}%)")
        print(f"nDCG@5: {avg_ndcg:.4f}")
        print(f"Total samples: {len(results)}")
        
        print("\n=== Case-wise Analysis ===")
        for case_type, metrics in case_analysis.items():
            case_r1_sim = np.mean(metrics['r1_sim'])
            case_r1_truth = np.mean(metrics['r1_truth'])
            case_ndcg = np.mean(metrics['ndcg'])
            print(f"{case_type}: R@1(Sim)={case_r1_sim:.4f}, R@1(Truth)={case_r1_truth:.4f}, nDCG@5={case_ndcg:.4f}")
        
        # 결과 저장
        if output_path:
            evaluation_results = {
                'overall_metrics': {
                    'r1_sim_score': r1_sim_score,
                    'r1_truth_score': r1_truth_score,
                    'ndcg_score': avg_ndcg,
                    'total_samples': len(results)
                },
                'case_analysis': {
                    case_type: {
                        'r1_sim_score': float(np.mean(metrics['r1_sim'])),
                        'r1_truth_score': float(np.mean(metrics['r1_truth'])),
                        'ndcg_score': float(np.mean(metrics['ndcg'])),
                        'sample_count': len(metrics['r1_sim'])
                    }
                    for case_type, metrics in case_analysis.items()
                },
                'detailed_results': results,
                'model_path': self.model_path
            }
            
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(evaluation_results, f, ensure_ascii=False, indent=2)
            
            print(f"\nResults saved to: {output_path}")
        
        return {'r1_sim': r1_sim_score, 'r1_truth': r1_truth_score, 'ndcg': avg_ndcg}, results


def main():
    parser = argparse.ArgumentParser(description='Evaluate Qwen VL 2.5 using pre-generated dataset')
    parser.add_argument('--model_path', type=str, required=True, help='Path to Qwen VL model')
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to evaluation dataset JSON')
    parser.add_argument('--output_path', type=str, default='evaluation_results.json', help='Output path for results')
    parser.add_argument('--max_samples', type=int, default=None, help='Maximum number of samples to evaluate')
    parser.add_argument('--device', type=str, default='cuda', help='Device to use')
    
    args = parser.parse_args()
    
    # 평가기 초기화
    evaluator = QwenVLDatasetEvaluator(args.model_path, args.device)
    
    # 평가 실행
    metrics, results = evaluator.evaluate(
        dataset_path=args.dataset_path,
        output_path=args.output_path,
        max_samples=args.max_samples
    )
    
    print("Evaluation completed!")


if __name__ == "__main__":
    main()
