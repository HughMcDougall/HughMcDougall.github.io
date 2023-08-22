Next Entry: [WebGuide](.\..\02_mcmcsamplers\page.html)  
  
  
Go Back: [Comfortably NumPyro](.\..\blog_numpyrohome.html)  
Return to [blog home](.\..\..\bloghome.html)  
  
This is the default header  
  
  
# Getting Started
  
  This tutorial covers the bare minimum basics of working with NumPyro: how to install, run and how to apply to a simple model.
  - Aimed at brand new users
  - Mention [DFM tutorial](https://dfm.io/posts/intro-to-numpyro/)
  - Mention [numpyro tutes](https://num.pyro.ai/en/stable/)
  
  
  ## Installing NumPyro
  
  - Best to work on some sort of linux system or mac
  - If working on a windows machine, try using the [Windows Subsystem for Linux](https://learn.microsoft.com/en-us/windows/wsl/install) which will let you run all sorts of
  - Also strongly advise using a conda environment or something similar (e.g. virtual environment). NumPyro can have significant changes between versions, particularly when using external modules like JAXNS's nested sampling, and so good version control is a must
  
  
  **Setting Up an Environment**
  
  ```
      conda create -n numpyroenv"
      conda install python
      conda install pip
  ```
  
  And then activate this environment with:
  
  ```
      conda activate numpyroenv
  ```
  
  You now have a conda environment running which safely contains all of the packages we'll need. `conda list` or `pip list` will show you everything you've got installed at the moment, including version numbers. If something is going wrong, this is a good place to start.
  
  **Installing NumPyro & Its Dependencies**
  
  First, install basic python packages that you'll need to do most anything with NumPyro. Note that many tutorials use the packages `corner` or `arviz` instead of `chainconsumer`. 
  
  ```
      pip install numpy, matplotlib, chainconsumer
  ```
  
  Now, install JAX, numpyro and their associated packages
  
  ```
      pip install jax, jaxlib, jaxopt
      pip install numpyro
  ```
  
  If you're planning on using nested sampling with numpyro, you'll need the following. Note that we're installing a *specific version of jaxns*. At time of writing, NumPyro is not compatible with the versions of jaxns 2.1.0 or greater.
  
  ```
      pip install etils, tensorflow_probability
      pip install jaxns==2.0.1
  ```
  
  ## Your First Model
  - Show imports & Data
  - In this example, show a simple linear regression
  
  
  ```python
  import jax, numpyro, chainconsumer
  import matplotlib.pyplot as plt
  ```
  
  
  ```python
  
  ```
    
For more detailed information, feel free to check my [GitHub repos](https://github.com/HughMcDougall/) or [contact me directly](hughmcdougallemail@gmail.com).  
  
