#!/usr/bin/env python3
"""
Easy R1 체크포인트(.pt)를 HuggingFace 형식(.safetensors)으로 변환
"""

import torch
import os
import argparse
from safetensors.torch import save_file
import json
from pathlib import Path


def convert_r1_checkpoint_to_hf(checkpoint_dir, output_dir=None):
    """
    R1 체크포인트를 HuggingFace 형식으로 변환
    
    Args:
        checkpoint_dir: R1 체크포인트 디렉토리 (global_step_XXX)
        output_dir: 출력 디렉토리 (None이면 checkpoint_dir/actor/huggingface 사용)
    """
    
    checkpoint_dir = Path(checkpoint_dir)
    
    # 입력 파일 경로
    pt_file = checkpoint_dir / "actor" / "model_world_size_1_rank_0.pt"
    hf_config_dir = checkpoint_dir / "actor" / "huggingface"
    
    if not pt_file.exists():
        raise FileNotFoundError(f"모델 파일을 찾을 수 없습니다: {pt_file}")
    
    if not hf_config_dir.exists():
        raise FileNotFoundError(f"HuggingFace 설정 디렉토리를 찾을 수 없습니다: {hf_config_dir}")
    
    # 출력 디렉토리 설정
    if output_dir is None:
        output_dir = hf_config_dir
    else:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"📁 입력 파일: {pt_file}")
    print(f"📁 출력 디렉토리: {output_dir}")
    print(f"📊 파일 크기: {pt_file.stat().st_size / 1024**3:.2f} GB")
    
    # PyTorch 체크포인트 로드
    print("\n🔄 PyTorch 체크포인트 로딩 중...")
    try:
        checkpoint = torch.load(pt_file, map_location='cpu')
        print("✅ 체크포인트 로드 완료")
    except Exception as e:
        print(f"❌ 체크포인트 로드 실패: {e}")
        raise
    
    # 체크포인트 구조 확인
    print(f"\n📦 체크포인트 키: {list(checkpoint.keys())}")
    
    # state_dict 추출
    if 'module' in checkpoint:
        state_dict = checkpoint['module']
        print("✅ 'module' 키에서 state_dict 추출")
    elif 'model' in checkpoint:
        state_dict = checkpoint['model']
        print("✅ 'model' 키에서 state_dict 추출")
    elif isinstance(checkpoint, dict) and any(k.startswith('model.') for k in checkpoint.keys()):
        state_dict = checkpoint
        print("✅ 직접 state_dict 사용")
    else:
        state_dict = checkpoint
        print("⚠️  체크포인트를 state_dict로 직접 사용")
    
    # 키 이름 정리 (module. 접두사 제거 등)
    cleaned_state_dict = {}
    for key, value in state_dict.items():
        # module. 접두사 제거
        clean_key = key.replace('module.', '')
        # _orig_mod. 접두사 제거 (torch.compile 사용시)
        clean_key = clean_key.replace('_orig_mod.', '')
        cleaned_state_dict[clean_key] = value
    
    print(f"\n📊 모델 파라미터 수: {len(cleaned_state_dict)}")
    print(f"📊 샘플 키들:")
    for i, key in enumerate(list(cleaned_state_dict.keys())[:5]):
        print(f"  - {key}: {cleaned_state_dict[key].shape}")
    
    # safetensors 형식으로 저장
    print(f"\n💾 safetensors 형식으로 변환 중...")
    
    # 모델 크기 확인하여 분할 필요 여부 판단 (5GB 이상이면 분할)
    total_size = sum(param.numel() * param.element_size() for param in cleaned_state_dict.values())
    total_size_gb = total_size / 1024**3
    
    print(f"📊 총 모델 크기: {total_size_gb:.2f} GB")
    
    if total_size_gb > 4.5:  # 5GB보다 약간 작게 설정
        print("⚠️  모델이 너무 큽니다. 3개 파일로 분할합니다...")
        save_sharded_model(cleaned_state_dict, output_dir)
    else:
        print("📦 단일 파일로 저장합니다...")
        safetensors_path = output_dir / "model.safetensors"
        save_file(cleaned_state_dict, safetensors_path)
        print(f"✅ 저장 완료: {safetensors_path}")
    
    print(f"\n✨ 변환 완료!")
    print(f"📁 HuggingFace 체크포인트 위치: {output_dir}")
    print(f"\n사용법:")
    print(f"  model_path = '{output_dir}'")


def save_sharded_model(state_dict, output_dir):
    """큰 모델을 여러 파일로 분할 저장"""
    
    # 파라미터를 3개 그룹으로 나누기
    keys = list(state_dict.keys())
    num_shards = 3
    shard_size = len(keys) // num_shards
    
    weight_map = {}
    
    for shard_idx in range(num_shards):
        start_idx = shard_idx * shard_size
        if shard_idx == num_shards - 1:
            end_idx = len(keys)
        else:
            end_idx = (shard_idx + 1) * shard_size
        
        shard_keys = keys[start_idx:end_idx]
        shard_dict = {k: state_dict[k] for k in shard_keys}
        
        # 파일 이름
        shard_filename = f"model-{shard_idx+1:05d}-of-{num_shards:05d}.safetensors"
        shard_path = output_dir / shard_filename
        
        # 저장
        save_file(shard_dict, shard_path)
        print(f"✅ 저장 완료: {shard_filename} ({len(shard_keys)} 파라미터)")
        
        # weight_map에 추가
        for key in shard_keys:
            weight_map[key] = shard_filename
    
    # index.json 생성
    index = {
        "metadata": {
            "total_size": sum(param.numel() * param.element_size() for param in state_dict.values())
        },
        "weight_map": weight_map
    }
    
    index_path = output_dir / "model.safetensors.index.json"
    with open(index_path, 'w') as f:
        json.dump(index, f, indent=2)
    
    print(f"✅ 인덱스 파일 생성: {index_path}")


def main():
    parser = argparse.ArgumentParser(description='R1 체크포인트를 HuggingFace 형식으로 변환')
    parser.add_argument('--checkpoint_dir', 
                       default='/home/work/smoretalk/seo/reranking/r1/checkpoints/easy_r1_msrvtt/qwen2_5_vl_3b_msrvtt_grpo_exp2/global_step_1000',
                       help='R1 체크포인트 디렉토리 (global_step_XXX)')
    parser.add_argument('--output_dir', 
                       default=None,
                       help='출력 디렉토리 (기본값: checkpoint_dir/actor/huggingface)')
    
    args = parser.parse_args()
    
    print("🚀 R1 체크포인트 변환 시작\n")
    
    try:
        convert_r1_checkpoint_to_hf(args.checkpoint_dir, args.output_dir)
    except Exception as e:
        print(f"\n❌ 변환 실패: {e}")
        import traceback
        traceback.print_exc()
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())

