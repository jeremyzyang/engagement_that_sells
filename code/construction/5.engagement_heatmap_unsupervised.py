#!/usr/bin/env python
# coding: utf-8

# In[1]:


from py_session import py_session
import cv2
import matplotlib.pyplot as plt
import numpy as np
import natsort
import os
import pickle
import imageio
from skimage import data, color
from skimage.transform import rescale, resize, downscale_local_mean


# In[2]:


video_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/frame/'
video_name = os.listdir(video_dir)

out_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/engagement_heatmap_unsupervised/'


# In[3]:


length = 224
width = 224

saliency = cv2.saliency.StaticSaliencySpectralResidual_create()

for video in video_name:
    y = []
    
    frame_name = os.listdir(video_dir + video)
    frame_name = natsort.natsorted(frame_name)
    
    if len(frame_name) < 1:
        next
    else:       
        for frame in frame_name:
            if not frame.startswith('.'):
                image = cv2.imread(video_dir + video + '/' + frame)
                image = resize(image , (length,width))
                image = np.float32(image)
                (success, saliencyMap) = saliency.computeSaliency(image)
                y.append(saliencyMap)
    
    y = np.array(y)
    out = out_dir + video
    np.save(out, y)


# In[4]:


# py_session()
# 42 modules found
# IPython             	7.22.0                  ipykernel           	5.3.4                   platform            	1.0.8
# PIL                 	8.2.0                   ipython_genutils    	0.2.0                   prompt_toolkit      	3.0.17
# argparse            	1.1                     jedi                	0.17.2                  ptyprocess          	0.7.0
# backcall            	0.2.0                   json                	2.0.9                   py_session          	0.1.1
# cffi                	1.14.5                  jupyter_client      	6.1.12                  pygments            	2.8.1
# colorama            	0.4.4                   jupyter_core        	4.7.1                   pyparsing           	2.4.7
# ctypes              	1.1.0                   kiwisolver          	1.3.1                   re                  	2.2.1
# cv2                 	4.8.1                   logging             	0.5.1.2                 scipy               	1.6.2
# cycler              	0.10.0                  matplotlib          	3.3.4                   six                 	1.15.0
# dateutil            	2.8.1                   natsort             	8.4.0                   skimage             	0.18.1
# decimal             	1.70                    numpy               	1.22.4                  traitlets           	5.0.5
# decorator           	4.4.2                   parso               	0.7.0                   wcwidth             	0.2.5
# distutils           	3.8.8                   pexpect             	4.8.0                   zlib                	1.0
# imageio             	2.9.0                   pickleshare         	0.7.5                   zmq                 	20.0.0

