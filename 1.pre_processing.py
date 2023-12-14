#!/usr/bin/env python
# coding: utf-8

# In[1]:


from py_session import py_session
import numpy as np
import os
import pandas as pd
import cv2
import natsort
import pickle
import numpy as np
from scipy.io import wavfile


# ### frame extraction

# In[2]:


# extract frames from videos

# get video directory
video_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/training/video/'
#video_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/video/'

# specify directory to save frames
frame_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/training/frame/'
#frame_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/frame/'

os.chdir(video_dir)
video_name = natsort.natsorted(os.listdir(video_dir))


# In[3]:


# sample 1 frame per second up to 60 seconds

for f in video_name:
    
    new_dir = os.path.join(frame_dir, f)
    os.mkdir(new_dir)
    
    os.chdir(video_dir)
    vidcap = cv2.VideoCapture(f)
    
    fps = vidcap.get(cv2.CAP_PROP_FPS) # sample 1 frame per second 
    
    os.chdir(new_dir)
    
    count = 0
        
    while vidcap.isOpened():
        sucess, frame = vidcap.read()
        
        if sucess and count <= fps*(60-1):
            cv2.imwrite('frame%d.jpg' % count, frame)
            count +=  fps # this advances one second
            vidcap.set(1, count)
        else:
            vidcap.release()
            break


# ### audio extraction

# In[4]:


import sys
from moviepy.editor import *
import ffmpeg
import numpy as np
import os
import pandas as pd
import cv2
import natsort
import pandas as pd
import pickle
import imageio
import subprocess
from scipy.io import wavfile
from statistics import mean


# In[5]:


audio_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/training/audio/'


# In[6]:


for file_num, file_name in enumerate(video_name, start=1):
    if 'mouthcropped' not in file_name:
        try:
            file_path_input = video_dir + file_name
            file_path_output = audio_dir + file_name + '.wav'

            video = VideoFileClip(file_path_input)
            audio = video.audio 
            audio.write_audiofile(file_path_output)
        
        except:
            continue


# ### speech recognition

# In[7]:


audio_name = os.listdir(audio_dir)


# In[8]:


import speech_recognition as sr
r = sr.Recognizer()


# In[9]:


speech = []

for i, file_name in enumerate(audio_name): 
    f = audio_dir + '/' + file_name
    
    with sr.AudioFile(f) as source:
        audio_data = r.record(source)
        try:
            t = r.recognize_google(audio_data, language = "zh")
            speech.append(t)
        except sr.UnknownValueError:
            t = 'NA'
            speech.append(t)


# In[10]:


df = pd.DataFrame()
df['video_full_id'] = [a.replace('.wav', '') for a in audio_name]
df['speech'] = speech


# In[11]:


# py_session()
# 58 modules found
# IPython             	7.22.0                  imageio_ffmpeg      	0.4.9                   ptyprocess          	0.7.0
# PIL                 	8.2.0                   ipykernel           	5.3.4                   py_session          	0.1.1
# argparse            	1.1                     ipython_genutils    	0.2.0                   pygments            	2.8.1
# backcall            	0.2.0                   jedi                	0.17.2                  pyparsing           	2.4.7
# bottleneck          	1.3.2                   json                	2.0.9                   pytz                	2021.1
# certifi             	2020.12.05              jupyter_client      	6.1.12                  re                  	2.2.1
# cffi                	1.14.5                  jupyter_core        	4.7.1                   requests            	2.31.0
# chardet             	4.0.0                   kiwisolver          	1.3.1                   scipy               	1.6.2
# charset_normalizer  	3.3.2                   logging             	0.5.1.2                 six                 	1.15.0
# colorama            	0.4.4                   matplotlib          	3.3.4                   skimage             	0.18.1
# csv                 	1.0                     moviepy             	1.0.3                   socks               	1.7.1
# ctypes              	1.1.0                   natsort             	8.4.0                   speech_recognition  	3.8.1
# cv2                 	4.8.1                   numpy               	1.22.4                  tqdm                	4.66.1
# cycler              	0.10.0                  pandas              	1.2.4                   traitlets           	5.0.5
# dateutil            	2.8.1                   parso               	0.7.0                   urllib3             	1.26.4
# decimal             	1.70                    pexpect             	4.8.0                   wcwidth             	0.2.5
# decorator           	4.4.2                   pickleshare         	0.7.5                   zlib                	1.0
# distutils           	3.8.8                   platform            	1.0.8                   zmq                 	20.0.0
# imageio             	2.9.0            

