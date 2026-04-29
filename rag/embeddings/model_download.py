from huggingface_hub import snapshot_download

# 下载 BGE-M3 到 D:\Aprojet\Model\bge-m3
model_dir = snapshot_download(
    repo_id="BAAI/bge-m3",
    local_dir="D:/Aprojet/Model/bge-m3",  # 你的路径
    local_dir_use_symlinks=False,         # 不使用软链接，直接保存
    resume_download=True,                 # 断点续传
    max_workers=8                         # 加速下载
)

print(f"模型已下载到：{model_dir}")