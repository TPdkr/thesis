# About
Bachelor's thesis repo. Depth estimation with computer vision

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
