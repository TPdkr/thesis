"""
KITTY PREPROCESSING

This file preprocesses the KITTY dataset from (kitty source)[https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction]
selected validation data and test data sets.

The goal is being able to integrate this into the torch workflow and only have data we care about with clear grouond truth values
for depth instead of a cloud of points. 

YOLO is used to filter for pictures that contain cars and to identify their bounding boxes. 
"""

from ultralytics import YOLO
import os
from PIL import Image
import numpy as np
import sklearn

# Load a pretrained YOLO model
model = YOLO("../build/yolo26n.pt")

# find all files in a given folder
ds_path = "../datasets/depth_selection/val_selection_cropped/image/"
search_for = ["car"]




def depth_read(filename):
    """
    Convert an image to a depth map in the form of a numpy array. This is copied
    from the KITTY dev kit. 
    """
    # loads depth map D from png file
    # and returns it as a numpy array,
    # for details see readme.txt

    depth_png = np.array(Image.open(filename), dtype=int)
    # make sure we have a proper 16bit depth map here.. not 8bit!
    assert(np.max(depth_png) > 255)

    depth = depth_png.astype(float) / 256.
    depth[depth_png == 0] = -1.
    return depth


def listPicsWith(dir, classes, verbose=False):
    """
    Return a list of pictures that contain specific classes listed in classes.

    Args:
        Str: dir - directory to search
        List of strings: classes - classes to search for
        Bool: verbose=False - debug mode

    Returns:
        List of strings: files that match the condition
    """
    # find all files in a given folder and covert their paths to be full
    files = os.listdir(dir)
    files_full = [dir+file for file in files]


    # debugging if needed
    if verbose:
        print(len(files))
        print(files[0])
        print(files_full[0])

    # compute results for all given files 
    results = model(files_full)
    # create a counter for the files that contains desired classes and a list
    usable=0
    usable_files=[]

    #find all files we want to use
    for i,result in enumerate(results):

        # class name of each box (copied from the docs)
        names = [result.names[cls.item()] for cls in result.boxes.cls.int()] 

        # conditions are met
        if set(classes).issubset(set(names)):
            usable+=1
            usable_files.append(files_full[i])

    #return final results
    print(f"{usable} images have {", ".join(classes)} in them")
    return usable_files
    
def togglePath(filepath):
    """
    Toggle between the depth map and the image path. This is done by replacing the relevant parts of the path. 
    It checks which path is presented. 

    Args:
        Str: filepath - path to toggle
    Returns:
        Str: path to the other file 
    """
    if "groundtruth_depth" in filepath:
        filepath = filepath.replace("/groundtruth_depth/","/image/")
        filepath = filepath.replace("_sync_groundtruth_depth_","_sync_image_")
    else:
        filepath = filepath.replace("/image/","/groundtruth_depth/")
        filepath = filepath.replace("_sync_image_","_sync_groundtruth_depth_")

    return filepath 