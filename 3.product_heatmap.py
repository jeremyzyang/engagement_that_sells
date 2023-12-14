#!/usr/bin/env python
# coding: utf-8

# In[1]:


from py_session import py_session
import pandas as pd
import os
import numpy as np
from natsort import natsorted, ns
import cv2
import matplotlib.pyplot as plt
import numpy as np
import itertools


# In[2]:


# get product image directory
picture_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/product_image/'

# get video directory for extracting product heatmap
video_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/frame/'
video_name = os.listdir(video_dir)

# specify directory to save output
out_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/product_heatmap/'


# In[3]:


# create a convex hull of all identified product pixels
def in_hull(p, hull):
    """
    Test if points in `p` are in `hull`

    `p` should be a `NxK` coordinates of `N` points in `K` dimensions
    `hull` is either a scipy.spatial.Delaunay object or the `MxK` array of the 
    coordinates of `M` points in `K`dimensions for which Delaunay triangulation
    will be computed
    """
    from scipy.spatial import Delaunay
    if not isinstance(hull,Delaunay):
        hull = Delaunay(hull)

    return hull.find_simplex(p)>=0


# In[4]:


# create a list of coordinates for all the pixels in an image
length = 224
width = 224

c1 = list(range(0, length))
c2 = list(range(0, width))
c3 = [c1, c2]
c = [l for l in itertools.product(*c3)]


# In[5]:


for video in video_name:    
    
    try:
        target = cv2.imread(picture_dir + video + '.jpg', 0)
        target = cv2.resize(target , (length,width))
        
        if target is not None:
            frame_dir = video_dir + video + '/'
            frame_names = os.listdir(frame_dir)
            frame_names = natsorted(frame_names)
            p_video = []
            
            # feature matching
            orb = cv2.ORB_create(nfeatures = 1000, scoreType = cv2.ORB_FAST_SCORE)
            kp1, des1 = orb.detectAndCompute(target, None)
            
            out = out_dir + video 
        
            for file_name in frame_names:
                try:
                    img = cv2.imread(frame_dir + file_name, 0)
                    img = cv2.resize(img, (length,width))
                    
                    kp2, des2 = orb.detectAndCompute(img,None)
                    # BFMatcher with default params
                    bf = cv2.BFMatcher()
                    matches = bf.knnMatch(des1, des2, k=2)
                    # apply ratio test
                    good = []
                    for m,n in matches: 
                        if m.distance < 0.75 * n.distance:
                            good.append(m)                    
                    
                    location = [kp2[g.trainIdx].pt for g in good]
                    location = np.rint(location).astype(int)
                    
                    p = np.zeros((img.shape))
                    for l in location:
                        p[l[1],l[0]] = 1
                    
                    hull = np.transpose(np.asarray(np.nonzero(p)))
                    convex = in_hull(c, hull)
                    convex = convex.reshape((length, width)).astype(np.int32) + p
                    convex[convex > 0] = 1
            
                    p_video.append(convex)                    
                
                except:
                    continue
            
            p_video_array = np.asarray(p_video, dtype=np.float32)
            np.save(out, p_video_array)
        
        else:
            continue

    except:
        print('match error')


# In[6]:


# py_session()
# 44 modules found
# IPython             	7.22.0                  ipykernel           	5.3.4                   platform            	1.0.8
# PIL                 	8.2.0                   ipython_genutils    	0.2.0                   prompt_toolkit      	3.0.17
# argparse            	1.1                     jedi                	0.17.2                  ptyprocess          	0.7.0
# backcall            	0.2.0                   json                	2.0.9                   py_session          	0.1.1
# bottleneck          	1.3.2                   jupyter_client      	6.1.12                  pygments            	2.8.1
# cffi                	1.14.5                  jupyter_core        	4.7.1                   pyparsing           	2.4.7
# colorama            	0.4.4                   kiwisolver          	1.3.1                   pytz                	2021.1
# csv                 	1.0                     logging             	0.5.1.2                 re                  	2.2.1
# ctypes              	1.1.0                   matplotlib          	3.3.4                   scipy               	1.6.2
# cv2                 	4.8.1                   natsort             	8.4.0                   six                 	1.15.0
# cycler              	0.10.0                  numpy               	1.22.4                  traitlets           	5.0.5
# dateutil            	2.8.1                   pandas              	1.2.4                   wcwidth             	0.2.5
# decimal             	1.70                    parso               	0.7.0                   zlib                	1.0
# decorator           	4.4.2                   pexpect             	4.8.0                   zmq                 	20.0.0
# distutils           	3.8.8                   pickleshare         	0.7.5                   

