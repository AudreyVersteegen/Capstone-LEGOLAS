import os

from PIL import Image
from os import listdir

directory = "C:/Users/audre/OneDrive/Desktop/rotated/"
destination = "C:/Users/audre/OneDrive/Desktop/cropped/"
for img in os.listdir(directory):
    if img.endswith(".jpg"):
        im = Image.open(str(directory + img))

        width, height = im.size

        left = width*0.3
        right = width-(width*0.4)
        top = height * 0.2
        bottom = height - (height * 0.2)

        im1 = im.crop((left, top, right, bottom))
        im1.save(str(destination) + "cropped_"+ str(img))