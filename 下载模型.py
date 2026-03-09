# download_from_hf.py
import os

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

from huggingface_hub import snapshot_download

model_id = "Qwen/Qwen2.5-1.5B-Instruct"
local_dir = "./qwen2.5-1.5b-instruct"

print(f"从HuggingFace镜像下载: {model_id}")
print(f"保存到: {local_dir}")

try:
    model_path = snapshot_download(
        repo_id=model_id,
        local_dir=local_dir,
        local_dir_use_symlinks=False,
        resume_download=True,
        max_workers=4
    )
    print(f"✅ 下载完成: {model_path}")

    # 检查文件
    print("\n下载的文件:")
    for file in os.listdir(local_dir):
        if os.path.isfile(os.path.join(local_dir, file)):
            size = os.path.getsize(os.path.join(local_dir, file)) / (1024 * 1024)
            print(f"  {file}: {size:.2f} MB")

except Exception as e:
    print(f"❌ 下载失败: {e}")