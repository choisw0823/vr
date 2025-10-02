from huggingface_hub import snapshot_download
import os

def download_msrvtt_features(repo_id="ccchhhoi/qwen2_5_vl_3b_msrvtt_features", local_dir="./"):
    print(f"Downloading MSRVTT features from {repo_id} to {local_dir}...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print("MSRVTT features download complete.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    features_dir = os.path.join(script_dir)
    download_msrvtt_features(local_dir=features_dir)
