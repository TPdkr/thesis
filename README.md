# About

Bachelor's thesis repo. Depth estimation with computer vision for autonomous race cars. 

## Resources and sources

- ultralytics YOLO26n model (downloaded via pip)
- DINOv3 developed by Meta(hugging face)
- KITTI dataset((kitty source)[https://www.cvlibs.net/datasets/kitti/eval_depth.php?benchmark=depth_prediction]
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
│   ├── dino.ipynb
│   ├── kitty.ipynb
│   ├── kitty.py
│   ├── model.ipynb
│   ├── rfdetr_embed.py
│   ├── rfdetr_test.py
│   ├── utils.py
│   ├── yolo.ipynb
│   ├── yolo.py
│   └── yolo_test.py
├── visualizations
│   ├── depthmap_vs_image_kitty.png
│   ├── loss_plot.png
│   └── ransac_vs_avg.png
├── libraries.txt
└── README.md

7 directories, 20 files
```

**Main dir:**
1. libraries.txt file containing libraries list
2. README.md

**build:** contains pre trained models like yolo when running code

**data:** contains data produced when running the code like embeddings arrays

**imgs:** a set of test images

**src:** python source code
1. yolo.py & yolo.ipynb
2. kitty.py & kitty.ipynb

**visualizations:** contains the graphics and visualizations made during the project


## Set Up

**This setup is written for VS code and might not work fully on conda or colab due to differences in how these platforms work
and manage files and libraries.**

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

It took less than an hour for the request to get approved and afterwards I was able to access the model via the API and successfully make the code
work with extractign image embeddings.

In order to make the thing work it is necessary to give the CLI a valid token which can be obtained from the website which can be done
during install or with the command below.

```
huggingface-cli login
```
