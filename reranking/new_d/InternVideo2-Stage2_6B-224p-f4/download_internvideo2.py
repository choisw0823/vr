from huggingface_hub import snapshot_download
import os

def download_internvideo2_model(repo_id="ccchhhoi/qwen2_5_vl_3b_internvideo2", local_dir="./"):
    print(f"Downloading InternVideo2 model from {repo_id} to {local_dir}...")
    snapshot_download(repo_id=repo_id, local_dir=local_dir, local_dir_use_symlinks=False)
    print("InternVideo2 model download complete.")

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    model_dir = os.path.join(script_dir)
    download_internvideo2_model(local_dir=model_dir)
