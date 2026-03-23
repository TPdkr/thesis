import requests
import supervision as sv
from PIL import Image
from rfdetr import RFDETRMedium
from rfdetr.util.coco_classes import COCO_CLASSES

import torch

print("MODEL LOADING\n")
model = RFDETRMedium()
print("MODEL LOADED\n")

image = Image.open("../imgs/image1.jpg")
detections = model.predict(image, threshold=0.5)
print("IMAGE PREDICTED\n")
labels = [f"{COCO_CLASSES[class_id]}" for class_id in detections.class_id]

annotated_image = sv.BoxAnnotator().annotate(image, detections)
annotated_image = sv.LabelAnnotator().annotate(annotated_image, detections, labels)

annotated_image.show()