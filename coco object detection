import numpy as np
import torch
import torchvision
from PIL import Image, ImageDraw, ImageFont
import matplotlib.pyplot as plt
import requests
from io import BytesIO

def load_image(image_path_or_url):
    if image_path_or_url.startswith("http"):
        response = requests.get(image_path_or_url)
        image = Image.open(BytesIO(response.content)).convert("RGB")
    else:
        image = Image.open(image_path_or_url).convert("RGB")
    return image


def get_coco_instance_category_names():
    """
    Returns the list of class names used in COCO dataset.
    These correspond to the class labels predicted by torchvision object detection models.
    """
    return [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
        'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
        'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl', 'banana',
        'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table',
        'toilet', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
        'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]

# simple object detection with Pytorch

# 1. Load Pretrained Faster R-CNN model
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# 2. Load an image
image = load_image("lion_hunts_zebra.jpg")

# 3. Preprocess image
input_tensor = torchvision.transforms.functional.to_tensor(image)
input_tensor = input_tensor.unsqueeze(0)  # Add batch dimension

# 4. Run object detection
with torch.no_grad():
    predictions = model(input_tensor)[0]

# 5. Draw results
draw = ImageDraw.Draw(image)
font = ImageFont.load_default()

# Only show boxes with high confidence
threshold = 0.8

COCO_INSTANCE_CATEGORY_NAMES = get_coco_instance_category_names()

for box, label, score in zip(predictions['boxes'], predictions['labels'], predictions['scores']):
    if score > threshold:
        box = box.tolist()
        label_name = COCO_INSTANCE_CATEGORY_NAMES[label]
        draw.rectangle(box, outline="red", width=3)
        draw.text((box[0], box[1]), f"{label_name}: {score:.2f}", fill="white", font=font)

# 6. Show result
plt.imshow(image)
plt.axis("off")
plt.title("Object Detection Result")
plt.show()
