from ultralytics import YOLO

# Load a pretrained YOLO model
model = YOLO("../build/yolo26n.pt")

# Run batched inference on a list of images
results = model(["../imgs/image1.jpg", "../imgs/image2.jpg"])  # return a list of Results objects

# Process results list
for result in results:
    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    obb = result.obb  # Oriented boxes object for OBB outputs
    #result.show()  # display to screen
    result.save(filename="../imgs/result.jpg")  # save to disk

embed = model.embed("../imgs/image1.jpg")

print("Embedding of an image is of length: {0}, element 0 of shape {1} and content: {2}".format(len(embed), embed[0].shape, embed))