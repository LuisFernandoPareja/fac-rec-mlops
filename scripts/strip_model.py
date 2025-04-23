import sys
import os
import torch

# Add yolov5 folder to the system path
yolo_path = os.path.join(os.path.dirname(__file__), '../yolov5')
sys.path.append(os.path.abspath(yolo_path))

from models.yolo import Model as DetectionModel
from torch.serialization import safe_globals

with safe_globals([DetectionModel]):
    checkpoint = torch.load("models/best.pt", map_location="cpu", weights_only=False)
    torch.save(checkpoint, "models/best_clean.pt")
