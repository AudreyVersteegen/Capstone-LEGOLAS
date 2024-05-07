import os

from PIL import Image
from os import listdir

directory = "C:/Users/audre/OneDrive/Desktop/training images/"
destination = "C:/Users/audre/OneDrive/Desktop/rotated/"


for img in os.listdir(directory):
    if img.endswith(".jpg"):
        im = Image.open(str(directory + img))

        im1 = im.rotate(-15)

        im1.save(str(destination) + "rot_neg_15_"+ str(img))