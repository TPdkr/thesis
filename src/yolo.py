from ultralytics import YOLO
from ultralytics import solutions
from PIL import Image
import numpy as np
import cv2
import os

# Load a pretrained YOLO model
MODEL_YOLO = YOLO("../build/yolo26n.pt", verbose=False)
CONFIDENCE = 0.5
CLASSES_YOLO=[2]

print("YOLO26n MODEL LOADED\n")

#OPTION 1
def getEmbeds1(imgs, conf=0, classes=[], verbose=False):
    """
    Getting embeddings of the objects in YOLO model using crops in RAM. Here all classes
    that are specified are detected and predicted using the model. 

    Args:
        imgs: image list to be processed
        conf: confidence threshold 0 be default
        classes: list of classes for detection ints, default []
        verbose: whether to print debugging info
    Returns:
        embedding: the resulting embedding vector
    """

    batch_size = 300
    #image is processed
    raw = MODEL_YOLO.predict(imgs, conf=conf, classes=classes, embed=None, stream=True)
    print(type(raw), type(raw[0]) if isinstance(raw, list) else "not a list")

    embeds = []
    objects_batch = []

    for i, result in enumerate(raw):
        
        img_path=imgs[i]
        img = np.array(Image.open(img_path), dtype=np.uint8)
    
        #result = raw[0] if isinstance(raw, list) else raw
        boxes = result.boxes #boxes are extracted from results
        xyxy = boxes.xyxy #boundaries for each

        #for each box the image is cropped
        for i, box in enumerate(xyxy):
            x1,y1,x2,y2 = map(int, box.tolist())
            crop = img[y1:y2, x1:x2]

            if (verbose):
                print("detected box: {0}; x1: {1} x2: {2} y1: {3} y2: {4}".format(box,x1,x2,y1,y2))
            objects_batch.append(crop)

        
        if len(objects_batch) >= batch_size:
            print("Processing batch")
            embeds.append(getEmbedFromCrops(objects_batch, verbose))
            objects_batch = []  # Clear the batch
        
    if objects_batch:  # Process any remaining objects
        embeds.append(getEmbedFromCrops(objects_batch, verbose))
        
    return embeds


#OPTION 2
cropper = solutions.ObjectCropper(
    model="../build/yolo26n.pt",  
    classes=CLASSES_YOLO,  #classes to crop
    conf=CONFIDENCE,   #confidence threshold
    verbose=True,
    crop_dir="../imgs/crops/"
)

print("CROPPER LOADED\n")

def getEmbeds2(imgs, verbose=False):
    """
    Getting embeddings using built in YOLO cropper object. This method, while possible
    is slower due to disk IO operations.

    Args:
        imgs: list of images to be processed
        verbose: whether to print debugging info
    Returns:
        embedding: the resulting embedding vector
    """
    #store objects for detection together
    embeds = []
    objects_batch = []
    batch_size = 100
    
    for i, img_path in enumerate(imgs):
        # Clear crops dir before each image
        for f in os.listdir("../imgs/crops/"):
            os.remove(f"../imgs/crops/{f}")

        img = cv2.imread(img_path)
        cropper(img)  ##passing a single image

        for crop_file in os.listdir("../imgs/crops/"):
            obj = cv2.imread(f"../imgs/crops/{crop_file}")
            objects_batch.append(obj)

        if len(objects_batch) >= batch_size:
            embedding = MODEL_YOLO.embed(objects_batch)
            embeds.append(embedding)
            objects_batch = []  # Clear the batch

    #batch processing
    if objects_batch:  # Process any remaining objects
        embedding = MODEL_YOLO.embed(objects_batch)
        embeds.append(embedding)

    if verbose:
        print("Embedding is a tensor of length : {0}".format(len(embedding)))
        print("Embedding vector is of length: {0}".format(len(embedding[0])))

    return embedding

# OPTION 3
def getEmbedFromCrops(crops, verbose=False):
    """
    Get a list of embedding from YOLO26n model based on a list of crops in an image.

    Args:
        crops: list of crops to be processed
        verbose: whether to print debugging info
    Returns:
        embedding: the resulting embedding vector
    """
    #batch embedding speeds up time for processing
    embedding = MODEL_YOLO.embed(crops)
    if verbose:
        print("Embedding is a tensor of length : {0}".format(len(embedding)))
        print("Embedding vector is of length: {0}".format(len(embedding[0])))

    return embedding
