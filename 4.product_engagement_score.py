#!/usr/bin/env python
# coding: utf-8

# In[1]:


from py_session import py_session
import numpy as np
import os
import pandas as pd
import cv2
import natsort
import pandas as pd


# In[2]:


enga_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/engagement_heatmap/'
enga_name = os.listdir(enga_dir)

product_dir = '/n/holylfs05/LABS/jyang_lab/Lab/tiktok_sample/sales_panel/product_heatmap/'
product_name = os.listdir(product_dir)

enga_name = set(enga_name).intersection(set(product_name))
product_name = set(enga_name).intersection(set(product_name))

enga_name = sorted(list(enga_name))
product_name = sorted(list(product_name))


# In[3]:


e_score = []
p_score = []
pe_score = []
video = []

for i in range(len(product_name)):
    
    enga = np.load(enga_dir + enga_name[i])
    product = np.load(product_dir + product_name[i])
    
    if enga.shape == product.shape:
        score = np.multiply(enga, product)
        score = np.sum(score)/score.shape[0]
    
        video.append(product_name[i].replace('.npy',''))
        pe_score.append(score)
        e_score.append(np.sum(enga))
        p_score.append(np.sum(product))  
    
    else:
        duration = min(product.shape[0],enga.shape[0])
        product = product[:duration]
        enga = enga[:duration]
        
        score = np.multiply(enga, product)
        score = np.sum(score)/score.shape[0]
    
        video.append(product_name[i].replace('.npy',''))
        pe_score.append(score)
        e_score.append(np.sum(enga))
        p_score.append(np.sum(product))  


# In[4]:


df = pd.DataFrame()
df['video_full_id'] = video
df['p_score'] = p_score
df['e_score'] = e_score
df['pe_score'] = pe_score


# In[5]:


# py_session()
# 37 modules found
# IPython             	7.22.0                  ipython_genutils    	0.2.0                   prompt_toolkit      	3.0.17
# argparse            	1.1                     jedi                	0.17.2                  ptyprocess          	0.7.0
# backcall            	0.2.0                   json                	2.0.9                   py_session          	0.1.1
# bottleneck          	1.3.2                   jupyter_client      	6.1.12                  pygments            	2.8.1
# colorama            	0.4.4                   jupyter_core        	4.7.1                   pytz                	2021.1
# csv                 	1.0                     logging             	0.5.1.2                 re                  	2.2.1
# ctypes              	1.1.0                   natsort             	8.4.0                   six                 	1.15.0
# cv2                 	4.8.1                   numpy               	1.22.4                  traitlets           	5.0.5
# dateutil            	2.8.1                   pandas              	1.2.4                   wcwidth             	0.2.5
# decimal             	1.70                    parso               	0.7.0                   zlib                	1.0
# decorator           	4.4.2                   pexpect             	4.8.0                   zmq                 	20.0.0
# ipykernel           	5.3.4                   

