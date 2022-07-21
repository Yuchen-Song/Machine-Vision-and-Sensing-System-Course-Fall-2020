# -*- coding: utf-8 -*-
"""
Created on Mon Nov 16 23:03:25 2020

@author: Aven
"""

from scipy.ndimage import filters
from numpy import *
from pylab import *

def compute_harris_response(im,sigma=3):
    imx = zeros(im.shape)
    #计算导数
    filters.gaussian_filter(im,(sigma,sigma),(0,1),imx)
    imy = zeros(im.shape)
    filters.gaussian_filter(im,(sigma,sigma),(1,0),imy)
    Wxx = filters.gaussian_filter(imx*imx,sigma)
   #计算harris矩阵分量   
    Wxy = filters.gaussian_filter(imx*imy,sigma)
    Wyy = filters.gaussian_filter(imy*imy,sigma)
	#计算矩阵的特征值和迹
    Wdet = Wxx*Wyy-Wxy**2
    Wtr = Wxx+Wyy
    return  Wdet/Wtr

def get_harris_points(harrisim, min_dist = 10, threshold = 0.1):
    conner_threshold = harrisim.max()*threshold
    harrisim_t = (harrisim>conner_threshold)*1
    
    coords = array(harrisim_t.nonzero()).T
    candidate_values = [harrisim[c[0],c[1]] for c in coords]
    index = argsort(candidate_values)
    allowed_locations = zeros(harrisim.shape)
    allowed_locations[min_dist:-min_dist,min_dist:-min_dist]=1
    filtered_coords = []
    for i in index:
        if allowed_locations[coords[i,0],coords[i,1]]==1:
            filtered_coords.append(coords[i])
            allowed_locations[(coords[i,0]-min_dist):(coords[i,0]+min_dist),(coords[i,1]-min_dist):(coords[i,1]+min_dist)]=0
#此处保证min_dist*min_dist只有一个harris特征点
    return filtered_coords

def plot_harris_points(image,filtered_coords):
    figure()
    gray()
    imshow(image)
    plot([p[1] for p in filtered_coords],[p[0]for p in filtered_coords],'+')
    axis('off')
    show()
    
for filename in os.listdir(r'./yosemite'):
	print(filename)
	im = array(Image.open("./resources/yosemite/" + filename).convert('L'))
	harrisim = compute_harris_response(im)
	filtered_coords = get_harris_points(harrisim, 6)
	plot_harris_points(im, filtered_coords)
