from ultralytics import YOLO
from ultralytics import solutions
import cv2
import os
import time

# Load a pretrained YOLO model
model = YOLO("../build/yolo26n.pt", verbose=False)
CONFIDENCE = 0.5
CLASSES_YOLO=[0, 1, 2, 3, 15]

print("MODEL LOADED\n")

#OPTION 1
def getEmbeds1(img, confidence=0, classes=[], verbose=False):
    #image is processed
    result = model(img)[0]
    boxes = result.boxes #boxes are extracted from results
    confs = boxes.conf #confidence score for each
    xyxy = boxes.xyxy #boundaries for each

    #for each box the image is cropped
    crops = []
    for i, box in enumerate(xyxy):
        if confs[i]>=confidence:
            x1,y1,x2,y2 = map(int, box.tolist())
            crop = img[y1:y2, x1:x2]

            if (verbose):
                print("detected box: {0}; x1: {1} x2: {2} y1: {3} y2: {4}".format(box,x1,x2,y1,y2))
            crops.append(crop)
        
    #batch embedding speeds up time for processing
    embedding = model.embed(crops)
    if verbose:
        print("Embedding is a tensor of length : {0}".format(len(embedding)))
        print("Embedding vector is of length: {0}".format(len(embedding[0])))

    return embedding


#OPTION 2
cropper = solutions.ObjectCropper(
    model="../build/yolo26n.pt",  
    #classes=[0, 1, 2, 3, 15],  #classes to crop
    conf=CONFIDENCE,   #confidence threshold
    verbose=True,
    crop_dir="../imgs/crops/"
)

print("CROPPER LOADED\n")

def getEmbeds2(img, classes=[], verbose=False):
    #crop images using built in function
    results = cropper(img)
    #extract crops from files
    crops = os.listdir("../imgs/crops/")

    if verbose:
        print("Results of cropper:")
        print(results)
        print(crops)

    #store objects for detection together
    objects = []
    for crop in crops:
        object = cv2.imread(f"../imgs/crops/{crop}")
        objects.append(object)
    #batch processing
    embedding = model.embed(objects)

    if verbose:
        print("Embedding is a tensor of length : {0}".format(len(embedding)))
        print("Embedding vector is of length: {0}".format(len(embedding[0])))

    return embedding

def getEmbed3(img, verbose=False):
    """
    get embeddings of objects in an image as an array back
    
    Args:
        img image to process
    Returns:
        embeddings of the objects

    """
    if (verbose):
        print("RF DETR FEATURE EXTRACTION IN PROGRESS")


print("\nRESULTS FOR PICTURE: ")
#opening the image
img = cv2.imread("../imgs/image1.jpg")
#custom function is called
start = time.time()
getEmbeds1(img, CONFIDENCE)
end = time.time()
print("TIME: get crops took {0} to run".format(end-start))

start = time.time()
getEmbeds2(img)
end = time.time()
print("TIME: get crops 2 took {0} to run".format(end-start))

embed = model.embed("../imgs/image1.jpg")

#clean the directory
files = os.listdir("../imgs/crops/")
print("Deleting files: {0}".format(files))
for f in files:
    os.remove("../imgs/crops/{0}".format(f))
