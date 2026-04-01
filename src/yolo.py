from ultralytics import YOLO
from ultralytics import solutions
from PIL import Image
import numpy as np
import cv2
import os
import torch

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

    batch_size_embed = 300
    batch_size_pred = 1
    embeds = []
    objects_batch = []
    i=0
    skipped=0

    paths_chunked = [imgs[i:i + batch_size_pred] for i in range(0, len(imgs), batch_size_pred)]

    # I encountered an issue when a tensor was returned instead of results.
    # Having a batch size should help with this. 

    for j, batch in enumerate(paths_chunked):
        #all images are predicted on
        for result in MODEL_YOLO.predict(batch, conf=conf, classes=classes, stream=True):
            # progress is tracked
            i+=1
            if i%50==0:
                print(f"Progress: {i}/{len(imgs)}; {i/len(imgs)*100:.2f}%")

            # Error handling here in case batch too big
            # Guard against unexpected tensor returns
            if not hasattr(result, 'boxes') or not hasattr(result, 'orig_img'):
                print(f"Unexpected result type {type(result)} for {i-1}, skipping")
                skipped += 1
                continue

            #og image is retrieved as well as the boxes
            img = result.orig_img
            boxes = result.boxes #boxes are extracted from results
    
            #do we actually have any boes to go through?
            if boxes is None or len(boxes) == 0:
                print("Boxes not found for an image")
                continue

            #for each object find detections and add them to the list
            for box in boxes.xyxy:
                x1, y1, x2, y2 = map(int, box.tolist())
                crop = img[y1:y2, x1:x2]

                #checking if results are valid
                if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
                    continue

                if (verbose):
                    print("detected box: {0}; x1: {1} x2: {2} y1: {3} y2: {4}".format(box,x1,x2,y1,y2))
                #add final results
                objects_batch.append(crop)           

            #flush the batch
            if len(objects_batch) >= batch_size_embed:
                print("Processing batch")
                embeds.extend(getEmbedFromCrops(objects_batch, verbose))
                objects_batch = []  # Clear the batch
        
    if objects_batch:
        embeds.extend(getEmbedFromCrops(objects_batch, verbose))
        objects_batch = []  # Clear the batch

    if verbose:
        print(f"Toral embeddings found : {len(embeds)}")
    print(f"Skipped {skipped} images due to invalid results out of {i}")
        
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
