import torch
import torchvision
from torchvision.io import read_image
import os

directory = "C:/Users/audre/OneDrive/Desktop/cropped/"
destination = "C:/Users/audre/OneDrive/Desktop/colorized/"

for img in os.listdir(directory):
    if img.endswith(".jpg"):
        im = read_image(str(directory+img))

        img1 = torchvision.transforms.functional.adjust_hue(im, -0.5)
        img2 = torchvision.transforms.functional.adjust_hue(im, 0.1)
        img3 = torchvision.transforms.functional.adjust_hue(im, -0.1)


        torchvision.io.write_jpeg(img1, str(destination)+"red_"+str(img))
        torchvision.io.write_jpeg(img2, str(destination)+"blue_"+str(img))
        torchvision.io.write_jpeg(img3, str(destination)+"green_"+str(img))

