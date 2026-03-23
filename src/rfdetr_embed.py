from PIL import Image
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES

import torch

print("MODEL LOADING\n")
model = RFDETRMedium()
print("MODEL LOADED")
core_model = model.model

image = Image.open("../imgs/image1.jpg")

# convert image to tensor
image_tensor = model.preprocess(image)

with torch.no_grad():
    outputs = core_model(image_tensor)

print(outputs.keys())

embeddings = outputs["hs"]
obj_embeddings = embeddings[-1]

print(obj_embeddings.shape)