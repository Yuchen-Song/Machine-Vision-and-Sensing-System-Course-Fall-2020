# -*- coding: utf-8 -*-
"""
Created on Tue Oct 13 14:58:26 2020

@author: Aven
"""

import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

img = cv2.imread('cat.jpg', 1)

import cv2
import numpy as np


def cross_correlation_2d(image, kernel):
    '''Given a kernel of arbitrary m x n dimensions, with both m and n being
    odd, compute the cross correlation of the given image with the given
    kernel, such that the output is of the same dimensions as the image and that
    you assume the pixels out of the bounds of the image to be zero. Note that
    you need to apply the kernel to each channel separately, if the given image
    is an RGB image.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN 
    image = np.array(image)
    kernel = np.array(kernel)  
    dimenssion = len(image.shape)
    m, n= kernel.shape
    if (((m%2) == 0) | ((n%2) == 0)):
        raise Exception("Kernel(m x n), with m and n both odd!")
    if dimenssion == 2:
        a, b = image.shape
        result = np.zeros([a-m+1, b-n+1])
        for i in range(a-m+1):
            for j in range(b-n+1):
                result [i , j]= np.sum(image[i:i+m, j:j+n] * kernel)
    else:
        a, b, c = image.shape
        result = np.zeros([a-m+1, b-n+1, c])
        for o in range(c):
            for i in range(a-m+1):
                for j in range(b-n+1):
                    result [i , j, o]= np.sum(image[i:i+m, j:j+n, o] * kernel)            
    return result
    # TODO-BLOCK-END

def convolve_2d(img, kernel):
    '''Use cross_correlation_2d() to carry out a 2D convolution.

    Inputs:
        img:    Either an RGB image (height x width x 3) or a grayscale image
                (height x width) as a numpy array.
        kernel: A 2D numpy array (m x n), with m and n both odd (but may not be
                equal).

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    img = np.array(img)
    kernel = np.array(kernel)  
    m, n= kernel.shape
    # kernel center 
    p = float(m - 1)/2.0
    q = float(n - 1)/2.0
    # Rotate the convolution kernel 180 degrees clockwise
    for i in range(m):
        for j in range(n):
            kernel[i, j] = kernel[int(2*p-1), int(2*q-j)]
    return cross_correlation_2d(img, kernel)
    # TODO-BLOCK-END

def gaussian_blur_kernel_2d(sigma, height, width):
    '''Return a Gaussian blur kernel of the given dimensions and with the given
    sigma. Note that width and height are different.

    Input:
        sigma:  The parameter that controls the radius of the Gaussian blur.
                Note that, in our case, it is a circular Gaussian (symmetric
                across height and width).
        width:  The width of the kernel.
        height: The height of the kernel.

    Output:
        Return a kernel of dimensions height x width such that convolving it
        with an image results in a Gaussian-blurred image.
    '''
    # TODO-BLOCK-BEGIN
    if (((height%2) == 0) | ((width%2) == 0)):
        raise Exception("Kernel(m x n) of gaussian blur, with m and n both odd!")
    kernel = np.zeros([height, width])
    p = (height-1)/2
    q = (width-1)/2
    # Calculate the weight matrix
    for i in range(height):
        for j in range(width):
            kernel[i, j] = 1/(2*np.pi*np.square(sigma)*np.exp((np.square(i-p)+np.square(q-j))/(2*np.square(sigma))))
    # The final weight matrix is obtained by normalization
    kernel = kernel / np.sum(kernel)
    return kernel
    # TODO-BLOCK-END

def low_pass(img, sigma, size):
    '''Filter the image as if its filtered with a low pass filter of the given
    sigma and a square kernel of the given size. A low pass filter supresses
    the higher frequency components (finer details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    img = np.array(img)
    dimenssion = len(img.shape)
    height = width = size
    kernel = gaussian_blur_kernel_2d(sigma, height, width)
    # padding, keep the dimensions of the image
    if dimenssion == 2:
        a, b = img.shape
        p = (height-1) + a
        q = (width-1) + b
        new_img = np.zeros([p, q], dtype=np.float)
        new_img[int((height-1)/2):int((height-1)/2 + a),int((width-1)/2):int((width-1)/2 + b)] = img[:,:].astype(np.uint8)
    else:
        a, b, c = img.shape
        p = (height-1) + a
        q = (width-1) + b
        new_img = np.zeros([p, q, c])
        new_img[int((height-1)/2):int((height-1)/2 + a),int((width-1)/2):int((width-1)/2 + b), :] = img[:,:,:].astype(np.uint8)
    return cross_correlation_2d(new_img, kernel).astype(np.uint8)
    # TODO-BLOCK-END

def high_pass(img, sigma, size):
    '''Filter the image as if its filtered with a high pass filter of the given
    sigma and a square kernel of the given size. A high pass filter suppresses
    the lower frequency components (coarse details) of the image.

    Output:
        Return an image of the same dimensions as the input image (same width,
        height and the number of color channels)
    '''
    # TODO-BLOCK-BEGIN
    return ((img - low_pass(img, sigma, size))+135).astype(np.uint8)
    # TODO-BLOCK-END


# Read the pictures
img1 = cv2.imread("dog.jpg", 1) 
img2 = cv2.imread("cat.jpg", 1) 

img1_lowpass = low_pass(img1, sigma=9.0, size=33)
img2_highpass = high_pass(img2, sigma=11.0, size=31)
# Mix pictures in equal proportions
img3 = cv2.addWeighted(img1_lowpass,0.5,img2_highpass,0.5, 0)


cv2.imwrite('left.png', img1_lowpass)
cv2.imwrite('right.png', img2_highpass)
cv2.imwrite('hybrid.png', img3)


cv2.imshow("dog",img1_lowpass)
cv2.waitKey(0)
cv2.imshow("cat", img2_highpass)
cv2.waitKey(0)
cv2.imshow("hybrid image", img3)
cv2.waitKey(0)
cv2.destroyAllWindows()






