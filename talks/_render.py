from glob import glob
import numpy as np

for folder in glob("*/"):
    out = open(folder + "_page.md",'w')
    files = glob(folder+"*.jpg")
    numbers = [int(filename.replace(folder+"Slide","").replace(".JPG","")) for filename in files]

    files = np.array(files)
    indices = np.argsort(numbers)
    files = files[indices]
    for file in files:
        out.write("![jpg](%s) \\  \n\n" %file.replace(folder, "./" ))

    out.close()
