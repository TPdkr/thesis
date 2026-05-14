from ultralytics import YOLO
from ultralytics import solutions
from PIL import Image
import numpy as np
import cv2
import os
import torch
from utils import getDevice

# Load a pretrained YOLO model
MODEL_YOLO = YOLO("../build/yolo26n.pt", verbose=False).to(getDevice())
MODEL_YOLO_EMBED = YOLO("../build/yolo26n.pt", verbose=False)
CONFIDENCE_YOLO = 0.5
CLASSES_YOLO=[2]#car
print("YOLO26n MODEL LOADED\n")

def getCropsFromResult(result, verbose=False):
    crops=[]

    # Error handling here in case batch too big
    # Guard against unexpected tensor returns
    if not hasattr(result, 'boxes') or not hasattr(result, 'orig_img'):
        print(f"Unexpected result type {type(result)}, no boxes or orig_img available")
        return None

    #og image is retrieved as well as the boxes
    img = result.orig_img
    boxes = result.boxes #boxes are extracted from results

    #do we actually have any boes to go through?
    if boxes is None or len(boxes) == 0:
        print("Boxes not found for an image")
        return None
  
    #for each object find detections and add them to the list
    for box in boxes.xyxy:
        x1, y1, x2, y2 = map(int, box.tolist())
        crop = img[y1:y2, x1:x2]

        #checking if results are valid
        if crop.size == 0 or crop.shape[0] < 2 or crop.shape[1] < 2:
            return None
        #add final results
        crops.append(crop)           

    return crops

def getCropsFromResults(results, verbose=False):
    """
    Extrtact all crops from results array and return as a list. 

    Args:
        results: list of YOLO results objects
        verbose: to print or not debug info
    Returns:
        crops: list of numpy ndarray of crops of images
    """
    crops = []
    for result in results:
        crops.extend(getCropsFromResult(result, verbose))
    return crops

def getEmbedFromCrops(crops):
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
    return embedding

# OPTION 3
def getEmbedFromResults(results, verbose=False):

    #KEY VARIABLES
    batch_size_embed = 300
    embeds = []#final embeddings list
    objects_batch = []#crops of object to embed
    i=0#progress counter

    #this is a midifed embed1 that takes results as inputs
    for result in results:
        # progress is tracked
        if i%50==0:
            print(f"Progress: {i}/{len(results)}; {i/len(results)*100:.2f}%")

        #crops are added and checked for validity
        crops = getCropsFromResult(result, verbose)
        if crops is None:
            print("Error getting crops from result")
            continue
        objects_batch.extend(crops)     

        #flush the batch when a certain size is reached 
        if len(objects_batch) >= batch_size_embed:
            print("Processing batch")
            embeds.extend(getEmbedFromCrops(objects_batch))
            objects_batch = []  # Clear the batch

        #update progress
        i+=1
        
    #if something remains we flush it
    if objects_batch:
        embeds.extend(getEmbedFromCrops(objects_batch))
        objects_batch = []  # Clear the batch

    if verbose:
        print(f"Total embeddings found : {len(embeds)}")
        
    return embeds


