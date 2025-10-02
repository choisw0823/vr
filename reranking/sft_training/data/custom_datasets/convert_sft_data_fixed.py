import json
import os
from typing import List, Dict, Any

def convert_sft_data_to_llamafactory_format():
    """SFT 데이터를 LLaMA-Factory 형식으로 변환"""
    
    # 기존 데이터 읽기 (절대 경로 사용)
    with open('/home/work/smoretalk/seo/reranking/sft_training/data/custom_datasets/sft_data_generation/sft_data_contrastive_reasoning_unique_v1/sft_training_data_kept.json', 'r', encoding='utf-8') as f:
        sft_data = json.load(f)
    
    # 변환된 데이터 저장할 리스트
    converted_samples = []
    
    # 프레임 기본 경로 (절대 경로 사용)
    frames_base_path = '/home/work/smoretalk/seo/reranking/sft_training/data/custom_datasets/sft_data_generation/sft_data_contrastive_reasoning_unique_v1/frames'
    
    print(f'총 {len(sft_data["samples"])}개 샘플 변환 시작...')
    
    for sample in sft_data['samples']:
        sample_index = sample['sample_index']
        text = sample['text']
        videos = sample['videos']
        gt_order = sample['gt_order']
        
        # 이미지 경로 생성 
        image_paths = []
        for video_idx in [1, 2, 3, 4, 5]:
            video_idx  # 1-based를 0-based로 변환
            video_folder = f'sample_{sample_index}_video_{video_idx}'
            # 12개 프레임 추가
            for frame_num in range(0, 12):
                frame_path = f'{frames_base_path}/{video_folder}/frame_{frame_num:02d}.jpg'
                image_paths.append(frame_path)
            # frame_path = f'{frames_base_path}/{video_folder}/combined_12frames.jpg'
            # image_paths.append(frame_path)
        
        # RankLLM 시스템 프롬프트와 입력 형식으로 사용자 메시지 생성
        system_prompt = """System Prompt: You are RankLLM, a vision-only reranker. You will receive a Query and N candidates. Each candidate is composed of 12 uniform sampled frames from video. 

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
        """


        user_content = f'{system_prompt}\n\nQuery: "{text}"\n\nCandidates:\n'
        
        for i, video_idx in enumerate([1, 2, 3, 4, 5]):
            # user_content += f'[{i+1}] video:' + "<|vision_start|>" + "<image>"*12 + "<|vision_end|>"+ "\n"
            user_content += f'[{video_idx}] video: <|vision_start|>' + '<image> ' * 12 + '<|vision_end|>\n'
            # user_content += f'[{i+1}] video: <|vision_start|><image><|vision_end|>\n'
        # 기존 O3 분석 결과 사용
        o3_analysis = sample.get('o3_analysis', '')
        
        if o3_analysis:
            # O3 분석 결과를 그대로 사용
            assistant_content = o3_analysis
        else:
            continue
        
        
        # 변환된 샘플 생성
        converted_sample = {
            'messages': [
                {
                    'role': 'user',
                    'content': user_content
                },
                {
                    'role': 'assistant',
                    'content': assistant_content.strip()
                }
            ],
            'images': image_paths
        }
        
        converted_samples.append(converted_sample)
        print(f'샘플 {sample_index} 변환 완료')
    
    # 변환된 데이터 저장
    output_path = '/home/work/smoretalk/seo/reranking/sft_training/data/custom_datasets/sft_rankllm_converted.json'
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(converted_samples, f, ensure_ascii=False, indent=2)
    
    print(f'\n✓ RankLLM 형식으로 변환 완료!')
    print(f'출력 파일: {output_path}')
    print(f'총 샘플 수: {len(converted_samples)}')
    print(f'각 샘플당 이미지 수: {len(image_paths)}개 (5개 비디오 × 12프레임)')
    print(f'형식: RankLLM 시스템 프롬프트 기반')
    
    return converted_samples

if __name__ == '__main__':
    convert_sft_data_to_llamafactory_format()
