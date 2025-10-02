#!/usr/bin/env python3
"""
reranking 데이터를 Easy-r1 형식으로 변환하는 스크립트
"""

import json
import os
from typing import List, Dict, Any

def convert_reranking_to_easy_r1(input_file: str, output_file: str, mode: str = "easy"):
    """
    reranking JSON 데이터를 Easy-r1 형식으로 변환
    
    Args:
        input_file: reranking JSON 파일 경로
        output_file: 출력 파일 경로
        mode: easy/medium/hard 모드
    """
    
    print(f"Converting {input_file} to Easy-r1 format...")
    
    with open(input_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    converted_data = []
    
    for sample in data["training_samples"]:
        # 텍스트와 비디오 정보 추출
        caption = sample["caption"]
        videos = sample["videos"]
        
        # 정답 비디오 찾기
        correct_video = None
        for video in videos:
            if video["is_correct"]:
                correct_video = video
                break
        
        if correct_video is None:
            print(f"Warning: No correct video found for caption: {caption[:50]}...")
            continue
        
        # Easy-r1 형식으로 변환
        # 프롬프트: 시스템 프롬프트 + 쿼리 + 후보들
        system_prompt = """You are RankLLM, a vision-only reranker. You will receive a Query and N candidates. Each candidate is composed of 12 sampled frames.

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

Constraints:
- Inside <think>, use ONLY the section tags <content>, <notes>, <contrast>, <rationale>.
- Do NOT use any per-candidate XML-like tags (e.g., no <cand .../>); write lines as [N] text...
- Keep all statements evidence-based from the grids; no speculation.
- Always produce a total order in <answer> using indices [1]..[N].


Query: "{query}"

Candidates:
{candidates}"""

        # 후보 비디오들 문자열 생성
        candidates = []
        for i, video in enumerate(videos):
            candidates.append(f"[{i+1}] video: <|vision_start|><image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <|vision_end|>")
        
        candidates_str = "\n".join(candidates)
        
        # 프롬프트 생성
        prompt = system_prompt.format(
            query=caption,
            candidates=candidates_str
        )
        
        # 정답 생성 (정답 비디오의 인덱스 찾기)
        correct_idx = None
        for i, video in enumerate(videos):
            if video["is_correct"]:
                correct_idx = i + 1  # 1-based indexing
                break
        
        if correct_idx is None:
            continue
            
        # 나머지 비디오들의 인덱스
        other_indices = [i+1 for i in range(len(videos)) if i+1 != correct_idx]
        
        # 정답 형식: [correct_idx] > [other1] > [other2] > [other3] > [other4]
        answer = f"[{correct_idx}] > " + " > ".join([f"[{idx}]" for idx in other_indices])
        
        # 이미지 경로들 (각 비디오당 12프레임)
        image_paths = []
        for video in videos:
            video_id = video["video_id"]
            # 프레임 경로 생성 (실제 경로에 맞게 수정 필요)
            for frame_idx in range(12):
                frame_path = f"/path/to/frames/{video_id}/frame_{frame_idx:02d}.jpg"
                image_paths.append(frame_path)
        
        # Easy-r1 형식으로 변환
        converted_sample = {
            "problem": prompt,
            "answer": answer,
            "images": image_paths
        }
        
        converted_data.append(converted_sample)
    
    # JSON 파일로 저장
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(converted_data, f, ensure_ascii=False, indent=2)
    
    print(f"Conversion completed! {len(converted_data)} samples converted.")
    print(f"Output saved to: {output_file}")

def main():
    """메인 함수"""
    
    # 입력 파일들
    input_files = [
        "/home/work/smoretalk/seo/reranking/sft_training/data/custom_datasets/sft_data_generation/reranking_train_easy.json",
        "/home/work/smoretalk/seo/reranking/sft_training/data/custom_datasets/sft_data_generation/reranking_train_medium.json", 
        "/home/work/smoretalk/seo/reranking/sft_training/data/custom_datasets/sft_data_generation/reranking_train_hard.json"
    ]
    
    # 출력 파일들
    output_files = [
        "/home/work/smoretalk/seo/reranking/sft_training/data/custom_datasets/easy_r1_reranking_easy.json",
        "/home/work/smoretalk/seo/reranking/sft_training/data/custom_datasets/easy_r1_reranking_medium.json",
        "/home/work/smoretalk/seo/reranking/sft_training/data/custom_datasets/easy_r1_reranking_hard.json"
    ]
    
    modes = ["easy", "medium", "hard"]
    
    for input_file, output_file, mode in zip(input_files, output_files, modes):
        if os.path.exists(input_file):
            convert_reranking_to_easy_r1(input_file, output_file, mode)
        else:
            print(f"Warning: Input file not found: {input_file}")

if __name__ == "__main__":
    main()
