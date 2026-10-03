
from huggingface_hub import snapshot_download

print("Downloading person detection model...")

snapshot_download(
    repo_id="hustvl/yolos-tiny",
    local_dir="models/yolos-tiny"
)

print("Downloading image embedding model...")

snapshot_download(
    repo_id="facebook/dinov2-small",
    local_dir="models/dinov2-small"
)

print("Both models downloaded successfully.")
