# About

Bachelor's thesis repo. Object specific depth estimation with computer vision for autonomous cars. 

## Resources and sources

- ultralytics YOLO26n model (downloaded via pip)
- DINOv3 developed by Meta(hugging face)
- KITTI dataset([kitty source](https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction))
selected validation data and test data sets)
- libraries found in [libraries.txt](./libraries.txt) file
- Constructor racing data

## Repository structure

The tree structure was achieved by running the command below in the main directory of the repo.
```
tree --dirsfirst --gitignore -L 2
```

```
.
├── build
├── data
│   └── README.md
├── imgs
│   ├── crops
│   ├── image1.jpg
│   ├── image2.jpg
│   ├── result0.jpg
│   └── result1.jpg
├── src
│   ├── da_model.ipynb
│   ├── dino.ipynb
│   ├── dino_yolo_embed.ipynb
│   ├── dino_yolo_embed_og.ipynb
│   ├── embed_model.ipynb
│   ├── kitty.py
│   ├── rfdetr_embed.py
│   ├── rfdetr_test.py
│   ├── utils.py
│   ├── visuals.py
│   ├── yolo.ipynb
│   ├── yolo.py
│   ├── yolo_test.py
│   └── zhu_model.ipynb
├── visualizations
│   ├── ....
├── libraries.txt
└── README.md

7 directories, 38 files
```

### Main dir: repository info
1. libraries.txt file containing libraries list
2. README.md
3. .gitignore to prevent datasets and such going into the repo

### build: contains pre trained models like yolo when running code

It also contains the models trained using this project by default.

### data: contains data produced when running the code like embeddings arrays

Also, test results are there in form of csv files as this allows to easily recreate all graphs and adjust them as needed.

### imgs: a set of test images

### src: python source code
1. ***yolo.py & yolo.ipynb*** functions utilizing YOLO and example use cases
2. ***kitty.py*** functions using KITTI dataset
3. ***utils.py*** useful function that fall out of scope of other files
4. ***visuals.py*** functions used for visualizations of the data
4. ***dino.ipynb testing dinov3*** to make sure it works locally

Other:
5. ***yolo_test.py*** just a file making sure the model works and data can be read

Core code and main body of code:
1. ***dino_yolo_embed_og.ipynb*** extract embeddings based on ground truth bounding boxes
2. ***dino_yolo_embed.ipynb*** file to extract embeddings from dino and yolo based on yolo bounding boxes
3. ***embed_model.ipynb*** models based on embeddings training and visualizations
4. ***zhu_model.ipynb*** model that utilizes vgg16 and requires longer training to predict depths
5. ***da_model.ipynb*** the depth anythign v3 based model

### visualizations: contains the graphics and visualizations made during the project

## Running the code to get results

The ASCII text below shows how the code should be executed kinda like a pipeline that should take shorter time
for embeddings based approach and over several hours for the vgg16 based approach.

```
                                                  
1 dino_yolo_embed_og.ipynb───>embed_model.ipynb   
                             (USE_OG_BOXES=True)  
                                                  
2 dino_yolo_embed.ipynb ─────>embed_model.ipynb   
                             (USE_OG_BOXES=False) 
                                                  
3 zhu_model.ipynb       

4 da_model.ipynb
                                                  
```

1. Embeddings based approach with ground truth boxes used
2. Embedddings based approach with YOLO recognized boxes
3. approach based on vgg16 and a depth regressor
4. depth anything v3 based approach


## Set Up

**This setup is written for VS code and might not work fully on conda or colab due to differences in how these platforms work
and manage files and libraries.**

First create a virtual environment and enter it in order to run the code there. 

Then run the code below in order to load all the libraries into the machine. If the code 
is running into memory issues it might help to purge pip cache or recreate the venv.

Also, I encountered memory issues when running flatpak version of VS code. Your system package manager version is recommened to avoid them.

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

It took less than an hour for the request to get approved and afterwards I was able to access the model via the API and successfully make the code
work with extractign image embeddings.

In order to make the thing work it is necessary to give the CLI a valid token which can be obtained from the website which can be done
during install or with the command below.

```
huggingface-cli login
```

### Depth anything installation

You can go to the source of these instructions on [depth-anything/DA3METRIC-LARGE](https://huggingface.co/depth-anything/DA3METRIC-LARGE) or see the steps below that replicate the instructions on the website for an install. I recommened running them in src dir.

```
git clone https://github.com/ByteDance-Seed/depth-anything-3
cd depth-anything-3
pip install -e .
```
