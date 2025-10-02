#!/usr/bin/env python3
"""
SFT 학습용 데이터 생성
1. reranking 데이터에서 16개 샘플 선택
2. 각 비디오를 12프레임으로 샘플링
3. GPT-4o에 보내서 SFT 학습용 데이터 생성
"""

import json
import cv2
import numpy as np
import random
import os
import base64
import time
from pathlib import Path
from typing import List, Dict, Any
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

class SFTDataGenerator:
    def __init__(self, 
                 reranking_data_path: str = "../reranking_train_hard.json",
                 video_base_path: str = "../video",
                 output_dir: str = "./sft_data",
                 openai_api_key: str = None):
        """
        SFT 데이터 생성기 초기화
        
        Args:
            reranking_data_path: reranking 데이터 파일 경로
            video_base_path: 비디오 파일들이 있는 기본 경로
            output_dir: 출력 디렉토리
            openai_api_key: OpenAI API 키
        """
        self.reranking_data_path = reranking_data_path
        self.video_base_path = video_base_path
        self.output_dir = output_dir
        self.openai_api_key = openai_api_key or os.getenv("OPENAI_API_KEY")
        
        # OpenAI 클라이언트 초기화
        self.client = OpenAI(api_key=self.openai_api_key)
        
        # 출력 디렉토리 생성
        os.makedirs(output_dir, exist_ok=True)
        os.makedirs(f"{output_dir}/frames", exist_ok=True)
        
        # 랜덤 시드 설정
        random.seed(42)
        
    def load_reranking_data(self) -> Dict[str, Any]:
        """reranking 데이터 로드"""
        print("reranking 데이터를 로드하는 중...")
        with open(self.reranking_data_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        print(f"로드 완료: {len(data['training_samples'])}개 샘플")
        return data
    
    def select_samples(self, data: Dict[str, Any], num_samples: int = 16) -> List[Dict]:
        """16개 샘플 선택"""
        print(f"{num_samples}개 샘플을 선택하는 중...")
        
        # 다양한 유사도 범위에서 샘플 선택
        samples = data['training_samples']
        
        # 유사도 범위별로 샘플 분류
        high_similarity = []
        medium_similarity = []
        low_similarity = []
        
        for sample in samples:
            similarities = [video['similarity_score'] for video in sample['videos']]
            max_sim = max(similarities)
            
            if max_sim > 2.5:
                high_similarity.append(sample)
            elif max_sim > 1.5:
                medium_similarity.append(sample)
            else:
                low_similarity.append(sample)
        
        # 각 범위에서 균등하게 선택
        selected = []
        ranges = [high_similarity, medium_similarity, low_similarity]
        samples_per_range = num_samples // len(ranges)
        
        for i, range_samples in enumerate(ranges):
            if i == len(ranges) - 1:  # 마지막 범위는 나머지 모두
                samples_per_range = num_samples - len(selected)
            
            if len(range_samples) >= samples_per_range:
                selected.extend(random.sample(range_samples, samples_per_range))
            else:
                selected.extend(range_samples)
        
        # 부족한 경우 추가 선택
        while len(selected) < num_samples:
            remaining = [s for s in samples if s not in selected]
            if remaining:
                selected.append(random.choice(remaining))
            else:
                break
        
        print(f"선택 완료: {len(selected)}개 샘플")
        return selected[:num_samples]
    
    def extract_frames(self, video_path: str, num_frames: int = 12) -> List[np.ndarray]:
        """비디오에서 12프레임 추출"""
        if not os.path.exists(video_path):
            print(f"비디오 파일이 존재하지 않습니다: {video_path}")
            return []
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"비디오를 열 수 없습니다: {video_path}")
            return []
        
        # 총 프레임 수 확인
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        if total_frames == 0:
            print(f"비디오에 프레임이 없습니다: {video_path}")
            cap.release()
            return []
        
        # 균등하게 프레임 선택
        frame_indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
        
        frames = []
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if ret:
                # BGR to RGB 변환
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frames.append(frame_rgb)
            else:
                print(f"프레임 {frame_idx}를 읽을 수 없습니다")
        
        cap.release()
        print(f"프레임 추출 완료: {len(frames)}개 프레임")
        return frames
    
    def combine_frames_with_labels(self, frames: List[np.ndarray]) -> np.ndarray:
        """12개 프레임을 하나의 이미지로 합치고 각 프레임에 라벨 추가"""
        if len(frames) != 12:
            print(f"프레임 개수가 12개가 아닙니다: {len(frames)}개")
            return None
        
        # 프레임 크기 통일 (가장 작은 크기로 맞춤)
        min_height = min(frame.shape[0] for frame in frames)
        min_width = min(frame.shape[1] for frame in frames)
        
        resized_frames = []
        for frame in frames:
            resized_frame = cv2.resize(frame, (min_width, min_height))
            resized_frames.append(resized_frame)
        
        # 3x4 그리드로 배치 (3행 4열)
        rows = []
        for i in range(3):  # 3행
            row_frames = resized_frames[i*4:(i+1)*4]  # 각 행에 4개씩
            row = np.hstack(row_frames)
            rows.append(row)
        
        combined = np.vstack(rows)
        
        # 각 프레임에 라벨 추가
        label_height = 40
        label_width = min_width
        
        # 라벨을 위한 공간 추가
        final_height = combined.shape[0] + label_height * 3
        final_width = combined.shape[1]
        final_image = np.ones((final_height, final_width, 3), dtype=np.uint8) * 255
        
        # 프레임 배치
        final_image[label_height:label_height+combined.shape[0], :] = combined
        
        # 라벨 추가
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.8
        font_color = (0, 0, 255)  # 빨간색
        thickness = 2
        
        # 각 행의 라벨 추가
        for i in range(3):  # 3행
            y_pos = label_height - 10 if i == 0 else label_height + i * (combined.shape[0] // 3) + 30
            for j in range(4):  # 4열
                frame_num = i * 4 + j + 1
                x_pos = j * min_width
                cv2.putText(final_image, f"Frame {frame_num}", (x_pos, y_pos), 
                           font, font_scale, font_color, thickness)
        
        return final_image
    
    def frames_to_base64(self, frames: List[np.ndarray]) -> str:
        """12개 프레임을 하나의 이미지로 합쳐서 base64로 인코딩"""
        if len(frames) != 12:
            print(f"프레임 개수가 12개가 아닙니다: {len(frames)}개")
            return ""
        
        # 프레임 합치기
        combined_image = self.combine_frames_with_labels(frames)
        if combined_image is None:
            return ""
        
        # 이미지 크기 조정 (너무 크면 압축)
        height, width = combined_image.shape[:2]
        if width > 1200:
            scale = 1200 / width
            new_width = 1200
            new_height = int(height * scale)
            combined_image = cv2.resize(combined_image, (new_width, new_height))
        
        # JPEG로 인코딩
        _, buffer = cv2.imencode('.jpg', cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
        base64_str = base64.b64encode(buffer).decode('utf-8')
        
        return base64_str
   

    def call_gpt4o(self, text: str, video_images: List[str], video_info: List[Dict]) -> str:
        """
        o3 API 호출 (비전 전용 reranking)
        - 입력 이미지는 후보당 1장(3x4 grid, frame 1..12 라벨 포함)이며 메타데이터는 사용하지 않는다.
        - 출력 형식은 반드시 아래 두 줄만:
            <think> </think>
            <answer> [i] > [j] > [k] > ... </answer>
        """
        if not self.openai_api_key:
            print("OPENAI_API_KEY가 설정되지 않았습니다.")
            return ""

        # 1) 시스템 프롬프트: 형식 강제 (영어)
        system_prompt = """
        System Prompt: You are RankLLM, a vision-only reranker. You will receive a Query and N candidates. Each candidate is a single composite image arranged in a 3×4 grid (12 tiles) labeled “frame 1 … frame 12” in temporal order (left→right, top→bottom). There is NO metadata.

        Decision rules (concise):
        - Judge relevance ONLY from what is visible in the grids.
        - Down-rank clear mismatches (wrong domain/scene/action). Consider temporal coherence.
        - Tie-breakers (in order): action visibility/close-up > temporal coherence > low occlusion/clutter > overall coverage.

        ABSOLUTE OUTPUT CONTRACT (exactly two blocks; no extra text/markdown):
        <think>
        <content>
        [1] short description of visible content only.
        [2] short description of visible content only.
        [3] ...
        [4] ...
        [5] ...
        </content>

        <notes>
        [1] brief note on alignment/conflict with the query.
        [2] brief note on alignment/conflict with the query.
        [3] ...
        [4] ...
        [5] ...
        </notes>

        <contrast>
        - One-liner: why a higher-ranked item beats a lower-ranked item (because/whereas style).
        - Add a few more one-liners as needed.
        </contrast>

        <rationale>
        1st=[i] (reason)
        2nd=[j] (reason)
        3rd=[k] (reason)
        4th=[m] (reason)
        5th=[n] (reason)
        </rationale>
        </think>
        <answer> [i] > [j] > [k] > [m] > [n] </answer>

        Input you will receive:
        Query: "<QUERY TEXT>"

        Candidates (each provided as one 3×4 grid image, no metadata):
        [1] image: <grid-image-1>
        [2] image: <grid-image-2>
        [3] image: <grid-image-3>
        [4] image: <grid-image-4>
        [5] image: <grid-image-5>

        Constraints:
        - Inside <think>, use ONLY the section tags <content>, <notes>, <contrast>, <rationale>.
        - Do NOT use any per-candidate XML-like tags (e.g., no <cand .../>); write lines as [N] text...
        - Keep all statements evidence-based from the grids; no speculation.
        - Always produce a total order in <answer> using indices [1]..[N].
        """



        # 2) 유저 프롬프트: 쿼리 + 후보 인덱스만 텍스트로 표기
        #    (메타데이터/유사도/정답 등은 포함하지 않음)
        user_header = f'Query: "{text}"\n\nCandidates:\n'
        for i in range(len(video_images)):
            user_header += f"[{i+1}] (3×4 grid; tiles labeled frame 1…12)\n"

        messages = [
            {
                "role": "system",
                "content": [
                    {"type": "text", "text": system_prompt, "cache_control": {"type": "ephemeral"}}
                    ],
                    },
            {"role": "user", "content": [{"type": "text", "text": user_header}]},
        ]

        # 3) 후보별 이미지 첨부 (각 인덱스 안내 텍스트 뒤에 해당 이미지 추가)
        for i, image_base64 in enumerate(video_images):
            idx = i + 1
            messages[1]["content"].append({"type": "text", "text": f"[{idx}] image:"})
            messages[1]["content"].append({
                "type": "image_url",
                "image_url": {"url": f"data:image/jpeg;base64,{image_base64}"}
            })

        # 4) API 호출 (결정성↑를 위해 temperature 낮춤, 토큰은 한 줄 출력에 충분)
        try:
            response = self.client.chat.completions.create(
                model="o3-2025-04-16",
                messages=messages,
            )
            return response.choices[0].message.content
        except Exception as e:
            print(f"API 호출 중 오류 발생: {e}")
            return ""

    def parse_answer_to_order(self, answer_text: str) -> List[int]:
        """
        '<answer> [1] > [3] > [2] > [4] > [5] </answer>' -> [1,3,2,4,5]
        """
        if not answer_text:
            return []
        
        # <answer> 태그 내용 추출
        s = answer_text.strip()
        if "<answer>" in s and "</answer>" in s:
            s = s.split("<answer>")[1].split("</answer>")[0]
        elif "<answer" in s:
            s = s.split("<answer", 1)[-1]
            if ">" in s:
                s = s.split(">", 1)[-1]
        
        # 대괄호 인덱스 파싱
        parts = [p.strip() for p in s.split(">")]
        order = []
        for p in parts:
            if "[" in p and "]" in p:
                try:
                    n = int(p[p.index("[")+1:p.index("]")])
                    order.append(n)
                except Exception:
                    pass
        return order

    def build_ground_truth_order(self, videos: List[Dict]) -> List[int]:
        """
        정답 랭킹 생성: is_correct가 True인 것 먼저, 그 다음 similarity_score 내림차순
        """
        enriched = []
        for i, v in enumerate(videos, start=1):
            is_correct = v.get("is_correct", False)
            sim_score = v.get("similarity_score", -1e9)
            enriched.append((i, bool(is_correct), float(sim_score)))
        
        # is_correct=True 먼저, similarity_score 내림차순
        enriched.sort(key=lambda x: (not x[1], -x[2], x[0]))
        return [i for (i, _, _) in enriched]

    def count_position_differences(self, pred_order: List[int], gt_order: List[int]) -> int:
        """
        위치가 다른 아이템 수 계산
        """
        if len(pred_order) != len(gt_order):
            return len(pred_order)  # 길이가 다르면 모두 다른 것으로 처리
        
        gt_positions = {item: pos for pos, item in enumerate(gt_order)}
        diff_count = 0
        
        for pos, item in enumerate(pred_order):
            if item not in gt_positions or gt_positions[item] != pos:
                diff_count += 1
        
        return diff_count

    def evaluate_ranking(self, pred_order: List[int], gt_order: List[int]) -> Dict[str, Any]:
        """
        랭킹 평가: 1위가 다르거나 위치가 다른 아이템이 2개 초과면 버림
        """
        if not pred_order or not gt_order:
            return {"keep": False, "reason": "invalid_order", "pos_diff": None}
        
        # 1위 확인
        top1_match = len(pred_order) > 0 and len(gt_order) > 0 and pred_order[0] == gt_order[0]
        
        # 위치 차이 계산
        pos_diff = self.count_position_differences(pred_order, gt_order)
        
        # 필터링 규칙
        keep = top1_match and (pos_diff <= 2)
        
        if not top1_match:
            reason = "top1_mismatch"
        elif pos_diff > 2:
            reason = "pos_diff_gt_2"
        else:
            reason = "ok"
        
        return {
            "keep": keep,
            "reason": reason,
            "pos_diff": pos_diff,
            "top1_match": top1_match
        }

    def process_sample(self, sample: Dict, sample_idx: int) -> Dict[str, Any]:
        """단일 샘플 처리"""
        print(f"\n=== 샘플 {sample_idx + 1} 처리 중 ===")
        
        text = sample['caption']
        videos = sample['videos']
        
        print(f"텍스트: {text}")
        
        # 각 비디오의 프레임 추출
        video_frames_data = []
        for i, video in enumerate(videos):
            video_id = video['video_id']
            video_path = os.path.join(self.video_base_path, f"{video_id}.mp4")
            
            print(f"비디오 {i+1}: {video_id}")
            frames = self.extract_frames(video_path)
            
            if frames:
                base64_image = self.frames_to_base64(frames)
                video_frames_data.append({
                    'video_id': video_id,
                    'frames': base64_image,
                    'video_info': video
                })
                
                # 프레임 저장
                frame_dir = f"{self.output_dir}/frames/sample_{sample_idx}_video_{i}"
                os.makedirs(frame_dir, exist_ok=True)
                
                # 개별 프레임 저장
                for j, frame in enumerate(frames):
                    cv2.imwrite(f"{frame_dir}/frame_{j:02d}.jpg", 
                               cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                
                # 합쳐진 이미지도 저장
                combined_image = self.combine_frames_with_labels(frames)
                if combined_image is not None:
                    cv2.imwrite(f"{frame_dir}/combined_12frames.jpg", 
                               cv2.cvtColor(combined_image, cv2.COLOR_RGB2BGR))
            else:
                print(f"비디오 {video_id}에서 프레임을 추출할 수 없습니다.")
        
        # GPT-4o 호출
        if video_frames_data:
            video_images = []
            video_info = []
            
            for vfd in video_frames_data:
                video_images.append(vfd['frames'])
                video_info.append(vfd['video_info'])
            
            print("o3에 요청 중...")
            gpt_response = self.call_gpt4o(text, video_images, video_info)
            
            if gpt_response:
                print("o3 응답 받음")
                
                # 정답 랭킹 생성
                gt_order = self.build_ground_truth_order(videos)
                
                # o3 응답에서 랭킹 파싱
                pred_order = self.parse_answer_to_order(gpt_response)
                
                # 랭킹 평가
                ranking_eval = self.evaluate_ranking(pred_order, gt_order)
                
                print(f"정답 순위: {gt_order}")
                print(f"예측 순위: {pred_order}")
                print(f"평가 결과: {ranking_eval}")
                
            else:
                print("o3 응답 실패")
                gpt_response = "응답을 받을 수 없었습니다."
                gt_order = self.build_ground_truth_order(videos)
                pred_order = []
                ranking_eval = {"keep": False, "reason": "api_failure", "pos_diff": None}
        else:
            gpt_response = "비디오 프레임을 추출할 수 없어 분석할 수 없습니다."
            gt_order = self.build_ground_truth_order(videos)
            pred_order = []
            ranking_eval = {"keep": False, "reason": "frame_extraction_failure", "pos_diff": None}
        
        # 결과 저장
        result = {
            "sample_index": sample_idx,
            "text": text,
            "videos": videos,
            "o3_analysis": gpt_response,
            "gt_order": gt_order,
            "pred_order": pred_order,
            "ranking_evaluation": ranking_eval,
            "video_frames_count": len(video_frames_data)
        }
        
        return result
    
    def generate_sft_data(self, num_samples: int = 2):
        """SFT 데이터 생성 메인 함수"""
        print("=== SFT 학습용 데이터 생성 시작 ===")
        
        # 데이터 로드
        reranking_data = self.load_reranking_data()
        
        # 샘플 선택
        selected_samples = self.select_samples(reranking_data, num_samples)
        
        # 각 샘플 처리
        all_results = []
        kept_samples = []
        discarded_samples = []
        
        for i, sample in enumerate(selected_samples):
            try:
                result = self.process_sample(sample, i)
                all_results.append(result)
                
                # 필터링: keep 여부에 따라 분류
                if result.get("ranking_evaluation", {}).get("keep", False):
                    kept_samples.append(result)
                else:
                    discarded_samples.append(result)
                
                # API 호출 간격 (rate limiting 방지)
                if i < len(selected_samples) - 1:
                    print("API 호출 간격 대기 중...")
                    time.sleep(2)
                    
            except Exception as e:
                print(f"샘플 {i} 처리 중 오류: {e}")
                continue
        
        # 메타데이터 생성
        metadata = {
            "total_samples": len(selected_samples),
            "processed_samples": len(all_results),
            "kept_samples": len(kept_samples),
            "discarded_samples": len(discarded_samples),
            "generation_time": time.strftime("%Y-%m-%d %H:%M:%S"),
            "model": "o3-2025-04-16",
            "description": "SFT 학습용 비디오-텍스트 매칭 데이터",
            "filter_criteria": {
                "discard_if_top1_different": True,
                "discard_if_position_diff_gt": 2
            }
        }
        
        # 전체, kept, discarded 데이터 구성
        all_data = {"metadata": metadata, "samples": all_results}
        kept_data = {"metadata": metadata, "samples": kept_samples}
        discarded_data = {"metadata": metadata, "samples": discarded_samples}
        
        # 결과 저장
        all_file = f"{self.output_dir}/sft_training_data_all.json"
        kept_file = f"{self.output_dir}/sft_training_data_kept.json"
        discarded_file = f"{self.output_dir}/sft_training_data_discarded.json"
        
        with open(all_file, 'w', encoding='utf-8') as f:
            json.dump(all_data, f, ensure_ascii=False, indent=2)
        with open(kept_file, 'w', encoding='utf-8') as f:
            json.dump(kept_data, f, ensure_ascii=False, indent=2)
        with open(discarded_file, 'w', encoding='utf-8') as f:
            json.dump(discarded_data, f, ensure_ascii=False, indent=2)
        
        print(f"\n=== SFT 데이터 생성 완료 ===")
        print(f"전체 결과: {all_file} ({len(all_results)}개)")
        print(f"채택된 샘플: {kept_file} ({len(kept_samples)}개)")
        print(f"버려진 샘플: {discarded_file} ({len(discarded_samples)}개)")
        
        return {"all": all_data, "kept": kept_data, "discarded": discarded_data}

def main():
    """메인 함수"""
    # API 키 확인
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY 환경변수를 설정해주세요.")
        print("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # SFT 데이터 생성기 초기화
    generator = SFTDataGenerator(
        reranking_data_path="../reranking_train_hard.json",
        video_base_path="../video",
        output_dir="./sft_data_contrastive_reasoning",
        openai_api_key=api_key
    )
    
    # SFT 데이터 생성
    sft_data = generator.generate_sft_data(num_samples=32)
    
    print("\n생성된 파일들:")
    print("- sft_data_contrastive_reasoning/sft_training_data.json")
    print("- sft_data_contrastive_reasoning/frames/ (프레임 이미지들)")

if __name__ == "__main__":
    main()
