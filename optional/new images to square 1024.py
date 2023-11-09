# -*- coding: utf-8 -*-
"""
Created on Mon Feb 27 15:35:21 2023

@author: alex
"""
import warnings
warnings.filterwarnings('ignore')
import os
import gc
import cv2
import json
import time
import random
import numpy as np
import pandas as pd 
from tqdm import tqdm

# image_folder_path = "./jpeg/test_PROVe-AI/"
# df_train = pd.read_csv('./data/prove-ai_metadata_2023-11-02.csv')
# image_folder_path = "./jpeg/test_MSKCC_2020/"
df_train = pd.read_csv('./data/consecutive-biopsies-for-melanoma-across-year-2020_metadata_2023-11-03.csv')
image_folder_path = "./jpeg/test_HIBA/"
# df_train = pd.read_csv('./data/hiba-skin-lesions_metadata_2023-11-07.csv')
df_train['target'] = df_train.benign_malignant == 'malignant'
df_train['patient_id'] = -2
# df_train.to_csv('./data/test_HIBA.csv', index = False)
# df_train.to_csv('./data/test_PROVE_AI.csv', index = False)
df_train.to_csv('./data/test_MSKCC_2020.csv', index = False)
train_ext = pd.DataFrame()

tt = pd.concat([train_ext, df_train])
tt = pd.concat([df_train, train_ext, df_train])

def crop(img_name, new_name):
    img = cv2.imread(img_name)
    h, w, _ = img.shape
    min_d = min(h, w)
    crop_d = ((max(h, w) - min_d) /2)
    x1 = 0; y1 = 0
    x2 = w; y2 = h
    if h > w:
        y1 = int(crop_d); y2 = int(h - crop_d) 
    else:
        x1 = int(crop_d); x2 = int(w - crop_d)
        
    if min_d > 1024:
        interpolation = cv2.INTER_AREA 
    else:
        interpolation = cv2.INTER_CUBIC
        # interpolation = cv2.INTER_LINEAR 
        
    res = cv2.resize(img[y1:y2, x1:x2], (1024, 1024), interpolation = interpolation)
    cv2.imwrite(new_name, res)

fns = df_train.isic_id.to_list()
for image_name in tqdm(fns):
    crop(image_folder_path + image_name + '.jpg', image_folder_path + '1024/' + image_name + '.jpg')
