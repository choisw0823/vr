from huggingface_hub import snapshot_download
import os

def download_sft_models(repo_id="ccchhhoi/qwen2_5_vl_3b_sft_msrvtt", local_dir="./"):
    print(f"Downloading SFT models from {repo_id} to {local_dir}...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print("SFT models download complete.")

if __name__ == "__main__":
    # 현재 스크립트가 있는 디렉토리를 기준으로 outputs 폴더를 찾습니다.
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_dir = os.path.join(script_dir)
    download_sft_models(local_dir=output_dir)