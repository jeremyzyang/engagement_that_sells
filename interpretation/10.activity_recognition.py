#!/usr/bin/env python
# coding: utf-8

# In[1]:


from py_session import py_session
import pandas as pd
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
import tensorflow as tf
import tensorflow_hub as hub


# In[2]:


def load_video(path, max_seconds = 60, resize = (224, 224)):
    frames = []

    vidcap = cv2.VideoCapture(path)
    fps = vidcap.get(cv2.CAP_PROP_FPS) # sample 1 frame per second 

    count = 0

    while vidcap.isOpened():
        sucess, frame = vidcap.read()
        #frame = cv2.resize(frame, (224, 224))

        if sucess and count <= fps*(max_seconds - 1):
            #cv2.imwrite('frame%d.jpg' % count, frame)
            frame = cv2.resize(frame, resize)
            frames.append(frame)
            count +=  (fps/1) # this advances one second
            vidcap.set(1, count)
        else:
            vidcap.release()
            break

    return np.array(frames) / 255.0


# In[3]:


video_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/video/'
video_name = os.listdir(video_dir)


# In[4]:


# Get the kinetics-400 action labels from the GitHub repository.
KINETICS_URL = "https://raw.githubusercontent.com/deepmind/kinetics-i3d/master/data/label_map.txt"
with request.urlopen(KINETICS_URL) as obj:
  labels = [line.decode("utf-8").strip() for line in obj.readlines()]
print("Found %d labels." % len(labels))


# In[5]:


i3d = hub.load("https://tfhub.dev/deepmind/i3d-kinetics-400/1").signatures['default']


# In[6]:


def predict(sample_video):
  act = []
  prob = []
  # Add a batch axis to the sample video.
  model_input = tf.constant(sample_video, dtype=tf.float32)[tf.newaxis, ...]

  logits = i3d(model_input)['default'][0]
  probabilities = tf.nn.softmax(logits)

  #print("Top 5 actions:")
  for i in np.argsort(probabilities)[::-1][:5]:
        act.append(labels[i])
        prob.append(probabilities[i].numpy() * 100)
  return(act, prob)


# In[7]:


df = pd.DataFrame()

splitedSize = 15

for i in range(len(video_name)):
    sample_video = load_video(video_dir + video_name[i])
    sample_video_splited = [sample_video[x:x+splitedSize] for x in range(0, len(sample_video), splitedSize)]
    
    for j in range(len(sample_video_splited)):
        act = predict(sample_video_splited[j])
        d = pd.DataFrame()
        d['activity'] = act[0]
        d['rank'] = range(1,6)
        d['prob'] = act[1]
        d['video_full_id'] = video_name[i]
        d['split'] = j
        df = df.append(d, ignore_index=True)
    
df.to_csv('/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/activity/activity.csv', index = False)


# In[8]:


# py_session()
# 59 modules found
# IPython             	7.22.0                  idna                	2.10                    pygments            	2.8.1
# PIL                 	8.2.0                   imageio             	2.9.0                   pytz                	2021.1
# argparse            	1.1                     ipykernel           	5.3.4                   re                  	2.2.1
# astunparse          	1.6.3                   ipython_genutils    	0.2.0                   requests            	2.31.0
# backcall            	0.2.0                   jedi                	0.17.2                  scipy               	1.6.2
# bottleneck          	1.3.2                   json                	2.0.9                   six                 	1.15.0
# certifi             	2020.12.05              jupyter_client      	6.1.12                  socks               	1.7.1
# cffi                	1.14.5                  jupyter_core        	4.7.1                   tblib               	1.7.0
# chardet             	4.0.0                   keras_preprocessing 	1.1.2                   tensorboard         	2.11.2
# charset_normalizer  	3.3.2                   logging             	0.5.1.2                 tensorflow          	2.5.0
# colorama            	0.4.4                   numpy               	1.22.4                  tensorflow_hub      	0.15.0
# csv                 	1.0                     opt_einsum          	v3.3.0                  termcolor           	(1, 1, 0)
# ctypes              	1.1.0                   pandas              	1.2.4                   traitlets           	5.0.5
# cv2                 	4.8.1                   parso               	0.7.0                   urllib3             	1.26.4
# dateutil            	2.8.1                   pexpect             	4.8.0                   wcwidth             	0.2.5
# decimal             	1.70                    pickleshare         	0.7.5                   wrapt               	1.12.1
# decorator           	4.4.2                   platform            	1.0.8                   yaml                	5.4.1
# distutils           	3.8.8                   prompt_toolkit      	3.0.17                  zlib                	1.0
# fsspec              	0.9.0                   ptyprocess          	0.7.0                   zmq                 	20.0.0
# h5py                	3.1.0                   py_session          	0.1.1                   

