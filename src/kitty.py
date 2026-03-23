from ultralytics import YOLO
import os

# Load a pretrained YOLO model
model = YOLO("../build/yolo26n.pt")

# find all files in a given folder
ds_path = "../datasets/depth_selection/val_selection_cropped/image/"
search_for = ["car"]

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
    

listPicsWith(ds_path, search_for, True)
