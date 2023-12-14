#!/usr/bin/env python
# coding: utf-8

# In[1]:


from py_session import py_session
import random
import re
import os
import tempfile
import ssl
import cv2
import numpy as np
import imageio
from IPython import display
from urllib import request
import numpy as np
import pandas as pd
from fer import FER


# In[2]:


emo_detector = FER(mtcnn=False)

def emotion(path, max_seconds = 60, resize = (224, 224)):
    frames = []

    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS) # sample 1 frame per second 

    count = 0
    
    while vidcap.isOpened():
        sucess, frame = vidcap.read()

        if sucess and count <= fps*(max_seconds - 1):
            frame = cv2.resize(frame, resize)
            captured_emotions = emo_detector.detect_emotions(frame)
            frames.append(captured_emotions)
            count +=  (fps/1) # this advances one second
            vidcap.set(1, count)
        else:
            vidcap.release()
            break

    return frames


# In[3]:


video_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/video/'
video_name = os.listdir(video_dir)
out_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/emotion/'


# In[4]:


for i in range(len(video_name)):
    e = emotion(video_dir + video_name[i])
    
    big_d = []
    
    for j in range(len(e)):
        l = []
        for k in range(len(e[j])):
            l.append(e[j][k]['emotions'])
        d = {'angry' : sum(item['angry'] for item in l),
            'disgust' : sum(item['disgust'] for item in l),
            'fear' : sum(item['fear'] for item in l),
            'happy' : sum(item['happy'] for item in l),
            'sad' : sum(item['sad'] for item in l),
            'surprise' : sum(item['surprise'] for item in l),
            'neutral' : sum(item['neutral'] for item in l)}
        big_d.append(d)
    df = pd.DataFrame(big_d)
    df.to_csv(out_dir + video_name[i] + '.csv', index = False)


# In[5]:


# py_session()
# 69 modules found
# IPython             	7.22.0                  imageio             	2.9.0                   ptyprocess          	0.7.0
# PIL                 	8.2.0                   imageio_ffmpeg      	0.4.9                   py_session          	0.1.1
# argparse            	1.1                     ipykernel           	5.3.4                   pygments            	2.8.1
# astunparse          	1.6.3                   ipython_genutils    	0.2.0                   pyparsing           	2.4.7
# backcall            	0.2.0                   ipywidgets          	7.6.3                   pytz                	2021.1
# bottleneck          	1.3.2                   jedi                	0.17.2                  re                  	2.2.1
# certifi             	2020.12.05              json                	2.0.9                   requests            	2.31.0
# cffi                	1.14.5                  jupyter_client      	6.1.12                  scipy               	1.6.2
# chardet             	4.0.0                   jupyter_core        	4.7.1                   six                 	1.15.0
# charset_normalizer  	3.3.2                   keras_preprocessing 	1.1.2                   skimage             	0.18.1
# colorama            	0.4.4                   kiwisolver          	1.3.1                   socks               	1.7.1
# csv                 	1.0                     logging             	0.5.1.2                 tblib               	1.7.0
# ctypes              	1.1.0                   matplotlib          	3.3.4                   tensorboard         	2.11.2
# cv2                 	4.8.1                   moviepy             	1.0.3                   tensorflow          	2.5.0
# cycler              	0.10.0                  numpy               	1.22.4                  termcolor           	(1, 1, 0)
# dateutil            	2.8.1                   opt_einsum          	v3.3.0                  tqdm                	4.66.1
# decimal             	1.70                    pandas              	1.2.4                   traitlets           	5.0.5
# decorator           	4.4.2                   parso               	0.7.0                   urllib3             	1.26.4
# distutils           	3.8.8                   pexpect             	4.8.0                   wcwidth             	0.2.5
# fer                 	22.5.1                  pickleshare         	0.7.5                   wrapt               	1.12.1
# fsspec              	0.9.0                   platform            	1.0.8                   yaml                	5.4.1
# h5py                	3.1.0                   proglog             	0.1.10                  zlib                	1.0
# idna                	2.10                    prompt_toolkit      	3.0.17                  zmq                 	20.0.0

