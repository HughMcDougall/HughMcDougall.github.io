

# Importing Image class from PIL module
from PIL import Image
from glob import glob

frame_to_extract  = 0.25
im_width = 370
im_height = 217
r_true = im_height / im_width


for targ in glob("*.gif"):
    with Image.open(targ) as im:
        im.seek(int(frame_to_extract*im.n_frames))
         
        # Size of the image in pixels (size of original image)
        width, height = im.size
        r = height/width
        if r<r_true:
            im_out = im.resize([im_width, int(im_width * r)])
        else:
            im_out = im.resize([int(im_height*r**-1), im_height])
         
        # Setting the points for cropped image
        width, height = im_out.size
        left = int(width/2-im_width/2)
        top = int(height/2-im_height/2)
        right = left + im_width
        bottom = top + im_height
         
        # Cropped image of above dimension
        # (It will not change original image)
        im_out = im_out.crop((left, top, right, bottom))
         
        # Shows the image in image viewer
        im_out.save(targ.replace(".gif","_thumb.png"))
        im_out.close()
