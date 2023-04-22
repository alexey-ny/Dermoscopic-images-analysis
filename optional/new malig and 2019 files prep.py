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

train_image_folder_path = "./jpeg/train/"
test_image_folder_path = "./jpeg/test/"
# jpegs2017 = './2017/jpegs/'
jpegs_ex_mal = './jpeg_extra_mal/'
jpegs_2019 = './jpeg_2019/'

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
cols = df_train.columns.to_list()


df_mal_2019 = pd.read_csv('train_malig_3_2019.csv')
print(df_mal_2019.diagnosis.value_counts())
print(df_mal_2019.benign_malignant.value_counts())
 
df_mal_new = pd.read_csv('train_malig_2_new.csv')

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
    # cv2.imwrite(img_name, res)
    
def rename_f(row):
    if '_downsampled' in row.image_name:
        r = row.image_name[:12]
    # if '_downsampled' in row.image:
    #     r = row.image[:12]
        # original = jpegs2019 + row.image + '.jpg'
        # output = jpegs2019 + r + '.jpg'
        
        # try:
        #     os.rename(original, output)
        # except WindowsError:
        #     try:
        #         os.remove(output)
        #         os.rename(original, output) 
        #     except:
        #         print('missing file')    
    else:
        r = row.image_name
        # r = row.image
    return r

df_mal_2019['image_name'] = df_mal_2019.apply(lambda x: rename_f(x), axis=1)
df_mal_2019.to_csv('train_malig_3_2019.csv', index = False)

df_train.diagnosis.value_counts()
print(df_mal_new.diagnosis.value_counts())
print(df_mal_new.benign_malignant.value_counts())
print(df_train.diagnosis.value_counts())
print(df_train.benign_malignant.value_counts())

fn_2019 = df_mal_2019.image_name.to_list()
for image_name in tqdm(fn_2019):
    crop(jpegs_2019 + image_name + '.jpg', jpegs_2019 + '1024/'  + image_name + '.jpg')

fn_2020_test = df_test.image_name.to_list()
for image_name in tqdm(fn_2020_test):
    crop(test_image_folder_path + image_name + '.jpg', test_image_folder_path + '1024/' + image_name + '.jpg')

fn_2020_train = df_train.image_name.to_list()
for image_name in tqdm(fn_2020_train):
    crop(train_image_folder_path + image_name + '.jpg', train_image_folder_path + '1024/' + image_name + '.jpg')
