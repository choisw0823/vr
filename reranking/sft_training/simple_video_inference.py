#!/usr/bin/env python3
"""
간단한 비디오 inference 예제
"""

import sys
sys.path.append('/home/work/smoretalk/seo/LLaMA-Factory/src')

from llamafactory.chat.chat_model import ChatModel


def main():
    # 1. 모델 설정
    args = {
        "model_name_or_path": "/home/work/smoretalk/seo/reranking/sft_training/outputs",
        "template": "qwen2_vl",
        "infer_backend": "huggingface",
        "trust_remote_code": True
    }
    
    # 2. ChatModel 초기화
    chat_model = ChatModel(args)
    
    messages = [
     {
        "role": "user",
        "content": "System Prompt: You are RankLLM, a vision-only reranker. You will receive a Query and N candidates. Each candidate is composed of 12 uniform sampled frames from video. \n\n        Decision rules (concise):\n        - Judge relevance ONLY from what is visible in the frames.\n        - Down-rank clear mismatches (wrong domain/scene/action). Consider temporal coherence.\n        - Tie-breakers (in order): action visibility/close-up > temporal coherence > low occlusion/clutter > overall coverage.\n\n        ABSOLUTE OUTPUT CONTRACT :\n        <think>\n        <content>\n        [1] Describe the content of the video and its relation to the query, aware of temporal sequence and spatial information.\n        [2] ...\n        [3] ... \n        [4] ...\n        [5] ...\n        </content>\n\n        <contrast>\n        - One-liner: why a higher-ranked item beats a lower-ranked item (because/whereas style). \n        - Add a few more one-liners as needed.\n        </contrast>\n\n        <summary>\n        1st=[i] (sufficient reasoning)\n        2nd=[j] (sufficient reasoning)\n        3rd=[k] (sufficient reasoning)\n        4th=[m] (sufficient reasoning)\n        5th=[n] (sufficient reasoning)\n        </summary>\n        </think>\n        <answer> [i] > [j] > [k] > [m] > [n] </answer>\n\n        Constraints:\n        - Inside <think>, use ONLY the section tags <content>, <contrast>, <summary>.\n        - Do NOT use any per-candidate XML-like tags (e.g., no <cand .../>); write lines as [N] text...\n        - Keep all statements evidence-based from the grids; no speculation.\n        - Always produce a total order in <answer> using indices [1]..[N].\n        \n\nQuery: \"demonstration on how to prepare something using a microwave\"\n\nCandidates:\n[1] video: <|vision_start|><image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <|vision_end|>\n[2] video: <|vision_start|><image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <|vision_end|>\n[3] video: <|vision_start|><image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <|vision_end|>\n[4] video: <|vision_start|><image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <|vision_end|>\n[5] video: <|vision_start|><image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <image> <|vision_end|>\n"
      }
    ]
    
    # 비디오 파일 경로 (실제 경로로 변경하세요)
    video_path = [
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_00.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_01.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_02.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_03.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_04.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_05.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_06.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_07.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_08.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_09.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_10.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7202/frame_11.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_00.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_01.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_02.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_03.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_04.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_05.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_06.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_07.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_08.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_09.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_10.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7572/frame_11.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_00.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_01.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_02.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_03.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_04.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_05.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_06.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_07.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_08.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_09.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_10.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7154/frame_11.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_00.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_01.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_02.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_03.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_04.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_05.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_06.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_07.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_08.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_09.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_10.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video8330/frame_11.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_00.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_01.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_02.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_03.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_04.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_05.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_06.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_07.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_08.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_09.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_10.jpg",
            "/home/work/smoretalk/seo/reranking/evaluation/frames/video7837/frame_11.jpg"
    ]
    
    # 4. Inference 실행
    responses = chat_model.chat(
        messages=messages,
        images=video_path,
        temperature=0.7,
        max_new_tokens=8192
    )
    
    # 5. 결과 출력
    for response in responses:
        print("응답:", response.response_text)


if __name__ == "__main__":
    main()
