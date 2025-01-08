'''
Render all .bmp files in from graph to this folder and generate
'''

import imageio
from glob import glob

import subprocess
import sys

filenames = glob("*.bmp")
images = []

for filename in filenames:
    images.append(imageio.imread(filename))
    
imageio.mimsave('./_movie.gif', images, fps=48, loop=0)
