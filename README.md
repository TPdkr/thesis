# About

Bachelor's thesis repo. Depth estimation with computer vision for autonomous race cars. 

## Resources and sources

- ultralytics YOLO26n model (downloaded via pip)
- DINOv3 developed by Meta(hugging face)
- KITTI dataset((kitty source)[https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction]
selected validation data and test data sets)

## Set Up

First create a virtual environment and enter it in order to run the code there. 

Then run the code below in order to load all the libraries into the machine. If the code 
is running into memory issues it might help to purge pip cache or recreate the venv.

Also, I encountered memory issues when running in the built in VS code terminal. However, switching to the 
system terminal helpd and no problems were encountered further.

Using not the latest version of python is recommended at the time. I used 3.12 as 3.14 did not have
builds of the libraries available yet. This can be chosen when creating venv. 

**Install all libraries:**

```
pip install -r libraries.txt
```

**Clear cache if full**
```
pip cache purge
```

### GPU acceleration

I used an intel iGPU for this project so by default the libraries file contains xpu dependenices that can
be adjusted based on user needs. 

### Hugging face and DINOv3 access

In order to access DINOv3 for this project I made a Hugging face account and requested access at [https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m).

It took INSERT TIME for the request to get approved and afterwards I was able to access the model via the API and successfully make the code
work with extractign image embeddings.
