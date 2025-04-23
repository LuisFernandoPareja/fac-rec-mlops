import torch
from pathlib import Path
from ultralytics.nn.tasks import DetectionModel
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
    ckpt = torch.load("models/best.pt", map_location="cpu", weights_only=False)
    cleaned_ckpt = to_posix_paths(ckpt)
    torch.save(cleaned_ckpt, "models/best2_linux.pt")