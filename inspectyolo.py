import torch

# Load the YOLOv5 model
model = torch.load('training/models/yolov5lu.pt', map_location=torch.device('cpu'))

# Print model architecture
print(model)

# Access specific model attributes
if 'model' in model:
   print("Model configuration:", model['model'])
if 'names' in model:
   print("Class names:", model['names'])
