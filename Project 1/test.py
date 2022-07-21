# -*- coding: utf-8 -*-
"""
Created on Tue Oct 20 09:57:10 2020

@author: HP
"""
import cv2
from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
img_cat1 = Image.open('cat.jpg')
img_cat2 = np.array(Image.open('cat.jpg'))
img_cat3 = cv2.imread('cat.jpg')

plt.subplot(131)
plt.imshow(img_cat1)
plt.title('Image')

plt.subplot(132)
plt.imshow(img_cat2)
plt.title('Image read')

plt.subplot(133)
plt.imshow(img_cat3)
plt.title('cv read')