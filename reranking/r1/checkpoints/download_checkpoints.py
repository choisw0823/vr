from huggingface_hub import snapshot_download
import os

def download_checkpoints(repo_id="ccchhhoi/qwen2_5_vl_3b_easy_r1_checkpoints", local_dir="./"):
    print(f"Downloading Easy-r1 checkpoints from {repo_id} to {local_dir}...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print("Easy-r1 checkpoints download complete.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    checkpoint_dir = os.path.join(script_dir)
    download_checkpoints(local_dir=checkpoint_dir)
