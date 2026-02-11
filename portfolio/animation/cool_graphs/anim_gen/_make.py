'''
Render all .bmp files in from graph to this folder and generate
'''

import imageio
from glob import glob

import subprocess
import sys

filenames = glob("*.bmp")
print("Found %i files. Making..." %len(filenames))
images = []

for i, filename in enumerate(filenames):
    if i % (len(filenames)//10)==0: print(i, end=', ')
    images.append(imageio.imread(filename))

print("Making")    
imageio.mimsave('./_movie.gif', images, fps=48, loop=0)
print("Done.")
