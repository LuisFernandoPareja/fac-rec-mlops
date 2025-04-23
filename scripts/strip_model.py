import sys
import os
import torch
from pathlib import Path
# Add yolov5 folder to the system path
yolo_path = os.path.join(os.path.dirname(__file__), '../yolov5')
sys.path.append(os.path.abspath(yolo_path))

from models.yolo import Model as DetectionModel
from torch.serialization import safe_globals

# Force all paths to Posix (Linux-style)
def to_posix_paths(obj):
    if isinstance(obj, dict):
        return {k: to_posix_paths(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [to_posix_paths(i) for i in obj]
    elif isinstance(obj, Path):
        return Path(str(obj).replace("\\", "/"))
    return obj


with safe_globals([DetectionModel]):
    checkpoint = torch.load("models/best.pt", map_location="cpu", weights_only=False)
    cleaned_ckpt = to_posix_paths(checkpoint)
    torch.save(cleaned_ckpt, "models/best_linux_clean.pt")
