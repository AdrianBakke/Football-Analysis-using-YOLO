#!/usr/bin/env python

import yaml
from pathlib import Path
from ultralytics import YOLO
p = Path(__file__).parent.parent

# Define configuration settings
model = p/"runs"/"detect"/"train6"/"weights"/"best.pt"
model = p/"training"/"yolo11n.pt"
#data_path = p/"training"/"data.yaml"

# Define paths
data = p/"data"/"football-players-detection-1"/"data.yaml"
args_path = Path(__file__).parent / "args.yaml"

# Load training arguments
with open(args_path, "r") as f:
    args = yaml.safe_load(f)

# Initialize YOLO model with optional predefined weights
model = YOLO(model=model)

# Freeze last layers (typically the head) for fine-tuning
nparams = 0
nheads = 0

for name, param in model.model.named_parameters():
    # Detection head components in YOLOv8 architecture
    nparams += 1
    if any(key in name for key in ['cv2', 'cv3', 'dfl']):
        nheads += 1
        param.requires_grad = False

nparams += 1
# Verify frozen layers
print("Frozen layers:")
for name, param in model.model.named_parameters():
    if not param.requires_grad:
        print(f" - {name}")

print(nparams, nheads)

model.train(data=data)

