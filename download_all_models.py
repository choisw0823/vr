#!/usr/bin/env python3
"""
모든 모델과 데이터를 한번에 다운로드하는 Python 스크립트
Usage: python download_all_models.py
"""

import os
import sys
import subprocess
from pathlib import Path

def run_download_script(script_path, description):
    """다운로드 스크립트를 실행하는 함수"""
    print(f"\n{description} 다운로드 중...")
    print("-" * 50)
    
    if not os.path.exists(script_path):
        print(f"❌ {description} 다운로드 스크립트를 찾을 수 없습니다: {script_path}")
        return False
    
    try:
        # Python 스크립트 실행
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, check=True)
        print(f"✅ {description} 다운로드 완료")
        if result.stdout:
            print(result.stdout)
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} 다운로드 실패: {e}")
        if e.stdout:
            print("STDOUT:", e.stdout)
        if e.stderr:
            print("STDERR:", e.stderr)
        return False

def main():
    print("=" * 60)
    print("모든 모델과 데이터 다운로드 시작")
    print("=" * 60)
    
    # 현재 스크립트의 디렉토리
    script_dir = Path(__file__).parent.absolute()
    
    # 다운로드할 항목들
    download_items = [
        {
            "script": script_dir / "reranking/sft_training/outputs/download_models.py",
            "description": "SFT 모델"
        },
        {
            "script": script_dir / "reranking/r1/checkpoints/download_checkpoints.py", 
            "description": "Easy-r1 체크포인트"
        },
        {
            "script": script_dir / "reranking/new_d/InternVideo2-Stage2_6B-224p-f4/download_internvideo2.py",
            "description": "InternVideo2 모델"
        },
        {
            "script": script_dir / "reranking/new_d/InternVideo/InternVideo2/multi_modality/outputs/msrvtt_features/download_msrvtt_features.py",
            "description": "MSRVTT features"
        }
    ]
    
    success_count = 0
    total_count = len(download_items)
    
    for item in download_items:
        if run_download_script(item["script"], item["description"]):
            success_count += 1
    
    print("\n" + "=" * 60)
    print("다운로드 완료!")
    print("=" * 60)
    print(f"성공: {success_count}/{total_count}")
    
    if success_count == total_count:
        print("🎉 모든 다운로드가 성공적으로 완료되었습니다!")
    else:
        print("⚠️  일부 다운로드가 실패했습니다. 위의 에러 메시지를 확인해주세요.")
    
    print("\n다운로드된 파일들:")
    print("- SFT 모델: reranking/sft_training/outputs/")
    print("- Easy-r1 체크포인트: reranking/r1/checkpoints/")
    print("- InternVideo2 모델: reranking/new_d/InternVideo2-Stage2_6B-224p-f4/")
    print("- MSRVTT features: reranking/new_d/InternVideo/InternVideo2/multi_modality/outputs/msrvtt_features/")
    print("\n총 다운로드 크기: 약 100GB+")
    print("다운로드 시간: 네트워크 속도에 따라 30분~2시간 소요")

if __name__ == "__main__":
    main()
