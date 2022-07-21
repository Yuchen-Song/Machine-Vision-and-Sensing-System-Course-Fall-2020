import sys
import cv2
import numpy as np
from matplotlib import pyplot as plt

def cross_correlation_2d(img, kernel):
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
    height = img.shape[0]
    width = img.shape[1]
    m = kernel.shape[0]
    n = kernel.shape[1]
    ret = np.zeros(img.shape, np.uint8)
    if len(img.shape)==2:
        img_padded = np.zeros((height+m-1, width+n-1))
        img_padded[int((m-1)/2):int(height+(m-1)/2),int((n-1)/2):int(width+(n-1)/2)] = img[:,:]
        '''
        for h in range(height):
            for w in range(width):
                 img_padded[int(h+(m-1)/2),int(w+(n-1)/2)]=img[h,w]
                 '''
        for h in range(height):
            for w in range(width):
                ret[h,w] = np.sum(img_padded[h:h+m,w:w+n] * kernel)
                '''
                for i in range(m):
                    for j in range(n):
                        ret[h,w] = ret[h,w] + img_padded[h+i,w+j]*kernel[i,j]
                        '''
    elif len(img.shape)==3:
        img_padded = np.zeros((height+m-1, width+n-1, 3))
        img_padded[int((m-1)/2):int(height+(m-1)/2),int((n-1)/2):int(width+(n-1)/2),:] = img[:,:,:]
        for k in range(3):
            '''
            for h in range(height):
                for w in range(width):
                    img_padded[int(h+(m-1)/2),int(w+(n-1)/2), k]=img[h,w, k]
                    '''
            for h in range(height):
                for w in range(width):
                    ret[h,w,k] = np.sum(img_padded[h:h+m,w:w+n,k] * kernel)
                    '''
                    for i in range(m):
                        for j in range(n):
                            ret[h,w, k] = ret[h,w,k] + img_padded[h+i,w+j,k]*kernel[i,j]
                            '''
    return ret        
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
    m = kernel.shape[0]
    n = kernel.shape[1]
    new_kernel = np.zeros(kernel.shape)
    for i in range(m):
        for j in range(n):
            new_kernel[i,j] = kernel[m-i-1,n-j-1]
            
    return cross_correlation_2d(img, new_kernel)
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
    kernel = np.zeros((height,width))
    hbar=(height-1)/2
    wbar=(width-1)/2
    for h in range(height):
        for w in range(width):
            kernel[h,w] = 1/(2*np.pi*sigma*sigma)*np.e**(-((h-hbar)**2+(w-wbar)**2)/(2*sigma**2))
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
    ret = np.zeros(img.shape,np.uint8)
    kernel = gaussian_blur_kernel_2d(sigma,size,size)
    ret = convolve_2d(img,kernel)
    return ret
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
    img_low = low_pass(img, sigma, size)
    ret = img - img_low + 100
    return ret
    # TODO-BLOCK-END
    
def main():    
    from PIL import Image
    img_cat = np.array(Image.open('cat.jpg'))
    img_dog = np.array(Image.open('dog.jpg'))
    
    dog_low = low_pass(img_dog, sigma=5.3, size=33)
    cat_high = high_pass(img_cat, sigma=7, size=45)
    img_hybrid = cv2.addWeighted(dog_low, 0.5, cat_high, 0.75, 0.0)
    height = img_hybrid.shape[0]
    width = img_hybrid.shape[1]
    '''
    if len(img_hybrid.shape)==2:    
        for h in range(height):
            for w in range(width):
                if img_hybrid[h,w] > 220:
                    img_hybrid[h,w] = 255
                elif img_hybrid[h,w] < 40:
                    img_hybrid[h,w] = 0
    else:
        for k in range(3):
            for h in range(height):
                    for w in range(width):
                        if img_hybrid[h,w,k] > 220:
                            img_hybrid[h,w,k] = 255
                        elif img_hybrid[h,w,k] < 40:
                            img_hybrid[h,w,k] = 0
                            
    plt.imshow(img_hybrid)
    plt.axis('off')
    '''
    
    
    plt.title('cat_highpass')
    plt.axis('off')
    
    
    
    plt.subplot(131)
    plt.imshow(dog_low)
    plt.title('dog_lowpass')
    plt.axis('off')
    
    plt.subplot(132)
    plt.imshow(cat_high)
    plt.title('cat_highpass')
    plt.axis('off')
    
    plt.subplot(133)
    plt.imshow(img_hybrid)
    plt.title('img_hybrid')
    plt.axis('off')
    
    plt.show()
    
    
if __name__ ==  '__main__':
    main()


