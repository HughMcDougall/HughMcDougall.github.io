'''
For any subfolders in the talks/ folder this will take all
.jpgs and write them as element in a markdown file. You
can get the .jpgs by exporting a .pptx file in Powerpoint.
'''

from glob import glob
import numpy as np
import os

for folder in glob("*/"):
    print("Looking in %s" %folder)
    out = open(folder + "_page.md",'w')
    print("\t writing to %s" %(folder + "_page.md"))

    os.chdir(folder+"/slides/")

    files = glob("*.jpg")
        
    numbers = [int(filename.replace("Slide","").replace(".JPG","")) for filename in files]

    files = np.array(files)
    indices = np.argsort(numbers)
    files = files[indices]
    for file in files:
        out.write("![jpg](%s)  \n\n" %("./slides/"+file))

    out.close()
    os.chdir("..")
    os.chdir("..")  

print("Done")
