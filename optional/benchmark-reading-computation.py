# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:49:02 2023
@author: alex
"""

import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import cv2
import numpy as np
import pandas as pd 
from imgaug import augmenters as iaa
import math
import logging
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score
from sklearn.utils import shuffle
# import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
import tensorflow.keras.backend as K

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras import mixed_precision

# import numba
# from numba import njit

import simplejpeg 

import time


# from turbojpeg import TurboJPEG, TJPF_GRAY, TJSAMP_GRAY, TJFLAG_PROGRESSIVE
# # using default library installation
# jpeg = TurboJPEG('C:\\libjpeg-turbo-gcc64\\bin\\libturbojpeg.dll')


n_folds = 5
DEVICE = "GPU"
MULTI = False
# USE DIFFERENT SEED FOR DIFFERENT STRATIFIED KFOLD
SEED = 1970

EFNS = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, 
        EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
        EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2]
nets_names = ['v1b0', 'v1b1', 'v1b2', 'v1b3', 'v1b4', 'v1b5', 'v1b6', 'v1b7',
              'v2b0', 'v2b1', 'v2b2']

params = {
    'IMG_SIZES' :   [768] * n_folds,
    # 'IMG_SIZES' :   [256] * n_folds,
    # 'IMG_SIZES' :   [128] * n_folds,
    'EFF_NETS' :    [9] * n_folds,
    # 'EFF_NETS' :    [8] * n_folds,
    'BATCH_SIZES' : [32] * n_folds,
    # 'BATCH_SIZES' : [64] * n_folds,
    # 'BATCH_SIZES' : [128] * n_folds,
    'EPOCHS' : [1] * n_folds,
    # 'EPOCHS' : [10] * n_folds,
    # number of times to duplicate the malignant samples, to use with augmentation to balance the classes
    'MAL_UPSAMPLE' : 5, 
    # test time augmentation steps
    'TTA' : 0,
    # possible values are 'RGB', 'HSV' and 'YCrCb'
    'COLOR_SPACE' : 'HSV' ,
    'DROP_LUM_CH' : False, # if we should drop Luminosity channel in HSV of YCrCb spaces
    '4TH_CHANNEL' : 'EI' , # either EI or MI, to estimate if one is more important than the other
    'LR_START' : 0.0005,
    'LR_MAX' : 0.0015,
    'LR_MIN' : 0.000001,
    'LR_RAMP_EP' : 5,
    'LR_SUS_EP' : 0,
    'LR_DECAY' : 0.8,
    # number of channels, to use additional channels/information besides RGB
    'N_CHANNELS' : 5
    }

loggers = {}
def myLogger(name, t_format = True):
    """
    Parameters
    ----------
    name : String, name of the logger        
    t_format : Boolean, 
        if DateTime to be used in the logger output, or jsut Message
    Returns
    -------
        Logger
    """
    global loggers
    
    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        handler = logging.FileHandler(f"{model_logs}{datetime.now().strftime('%Y%m%d')}-{nets_names[params['EFF_NETS'][0]]}-{name}.log")
        if t_format:
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        else:
            formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        loggers[name] = logger                       
        return logger


def print_log(out, loggers):
    """
    Outputs to both console and list of loggers    
    Parameters
    ----------
    out : String        
    loggers : List of loggers.
    """
    print(out)
    for logger in loggers:
        logger.info(out)
    
   
tb_logs = 'logs_tb'
model_logs = './logs_model/'

# train_image_folder_path = "F:\\melanoma_alex\\jpeg\\train\\1024\\"
# test_image_folder_path = "F:\\melanoma_alex\\jpeg\\test\\1024\\"
train_image_folder_path = "./jpeg/train/1024/"
test_image_folder_path = "./jpeg/test/1024/"

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

main_logger = myLogger('benchmark')
# res_logger = myLogger('results', False)

print_log('*' * 80, [main_logger])
print_log('Begin logging', [main_logger])
# print_log('tensorflow version:' + str(tf.__version__), [main_logger])

# print_log('Compute dtype: %s' % policy.compute_dtype, [main_logger])
# print_log('Variable dtype: %s' % policy.variable_dtype, [main_logger])

# gpu_devices = tf.config.list_logical_devices('GPU')
# if gpu_devices:
#     for gpu_device in gpu_devices:
#         print_log('device available:' + str(gpu_device), [main_logger])

# if DEVICE != "TPU":
#     if MULTI:
#         print_log("Using default strategy for multiple GPUs", [main_logger])
#         strategy = tf.distribute.MirroredStrategy(gpu_devices)
#     else:
#         print_log("Using default strategy for CPU and single GPU", [main_logger])
#         strategy = tf.distribute.get_strategy()
   
# AUTO = tf.data.experimental.AUTOTUNE
# REPLICAS = strategy.num_replicas_in_sync
# print_log(f'Number of Replicas Used: {REPLICAS}', [main_logger])

pd.set_option('display.max_columns', None)

df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
cols = df_train.columns.to_list()

df_mal_2019 = pd.read_csv('train_malig_3_2019.csv')

# lets create a test set for the Reject Classifier, 
# loading stratified lists by Patient_ID from the BaseModel 20
# to prevent leakage through similar images of the same patient
train_idx = np.load('train_patients_IDs.npy', allow_pickle = True)
test_idx = np.load('test_patients_IDs.npy', allow_pickle = True)

test_set = df_train.loc[df_train['patient_id'].isin(test_idx)]
trn_set = df_train.loc[df_train['patient_id'].isin(train_idx)]

train_set = pd.concat([trn_set, df_mal_2019[trn_set.columns]]).reset_index(drop = True)
train_IP_set = train_set['patient_id'].unique()

y_test_set = test_set['target']
test_set.drop(columns = ['target'], inplace = True, axis = 1)
print_log(y_test_set.value_counts(), [main_logger])
y_train_set = train_set['target']
print_log(y_train_set.value_counts(), [main_logger])



# Image Augmentation
sometimes = lambda aug: iaa.Sometimes(0.35, aug)
augmentation = iaa.Sequential([  
                                iaa.Fliplr(0.5),
                                iaa.Flipud(0.5),
                                iaa.LinearContrast((0.9, 1.1)),
                                iaa.Multiply((0.9, 1.1), per_channel=0.2),
                                sometimes(iaa.Cutout(nb_iterations=(1, 3), size=0.05, squared=False, fill_mode="constant", cval=0)),
                                sometimes(iaa.Crop(px=(20, 80), keep_size = True, sample_independently = False)),
                                # sometimes(iaa.Affine(rotate=(-35, 35))),
                                sometimes(iaa.Affine(scale=(0.95, 1.05)))
                            ], random_order = True)       

print_log("NOT Using numba", [main_logger])
# @njit   
def BGR2MI(img):
    """
    Computes Melanin Index
    Parameters
    ----------
    img : array
        an image in BGR format 
    Returns
    -------
    integer arrays 
        MI, including normalised and rescaled
    """
    MI = 100 * np.log10(1/(img[:,:,2].astype('float')+1))
    MI_norm = (MI * 0.0042 + 1) * 255
    return MI_norm.astype('int16')


# @njit
def BGR2EI(img):
    """
    Computes Erythema Index
    Parameters
    ----------
    img : array
        an image in BGR format 
    Returns
    -------
    integer arrays 
        EI, including normalised and rescaled
    """
    EI = 100 * (np.log10(1/(img[:,:,1].astype('float')+1)) - 1.44 * np.log10(1/(img[:,:,2].astype('float')+1)))
    EI_norm = (EI * 0.0017 + 0.4098) * 255
    return EI_norm.astype('int16')   


#---------------------------------------------------------------------------------------------
def _read(path, desired_size, color_space = 'RGB', toAugment = False, drop_luminosity = False, method = 'CV2'):
    """
    Will be used in DataGenerator
    Parameters
    ----------
    path : string
        path to the image.
    desired_size : tuple (x, y, ch)
        image size to which images shall be up/down-sized.
    color_space : string, optional
        what color space image shall be converted into. The default is 'RGB'.
    toAugment : boolean, optional
        if augmentation shall be applied to the image. The default is False.
    drop_luminosity : boolean, optional
        if Luminosity channel shall be omitted. The default is False.
    Returns
    -------
    img_exp : array
        image in the selected color space, could include extra channels for MI and EI.
    """
    if method == 'CV2':
        img_BGR = cv2.imread(path)
    elif method == 'TurboJpeg':
        in_file = open(path, 'rb')
        img_BGR = jpeg.decode(in_file.read())
        in_file.close()
    else:
        with open(path, 'rb') as f:
            tf = f.read() # Read whole file in the file_content string
            if simplejpeg.is_jpeg(tf):
                img_BGR = simplejpeg.decode_jpeg(tf, colorspace = 'bgr')
    
    # to compensate for reduced number of channels passed in the image size        
    if drop_luminosity:
        n_ch = desired_size[2] + 1
    else:
        n_ch = desired_size[2]
    if img_BGR is None:
        main_logger.warning('Error load image:', path)
    
    if toAugment: 
        img_BGR = augmentation.augment_image(img_BGR)      
    
    if desired_size[0] != img_BGR.shape[0]:
        img_BGR = cv2.resize(img_BGR, desired_size[:2], interpolation=cv2.INTER_LINEAR)
        
    if color_space == 'HSV':
        img_exp = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
        if drop_luminosity:
            img_exp  = img_exp[:, :, :2]
    elif color_space == 'YCrCb': 
        img_exp = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb )
        if drop_luminosity:
            img_exp  = img_exp[:, :, 1:]
    else:           
        img_exp = img_BGR.copy()

    # changed the order of EI and MI in version 24, 
    # now EI will be used in 4 channels model instead of MI
    if n_ch > 3:
        if params['4TH_CHANNEL'] == 'EI':
            EI_norm = BGR2EI(img_BGR)      
            EI_exp = np.expand_dims(EI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, EI_exp), axis = 2)        
        else:
            MI_norm = BGR2MI(img_BGR)
            MI_exp = np.expand_dims(MI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, MI_exp), axis = 2)        
    if n_ch > 4:
        # reverse which channel to add as 5th, since we've alreay added the 4th one
        if params['4TH_CHANNEL'] == 'EI':
            MI_norm = BGR2MI(img_BGR)
            MI_exp = np.expand_dims(MI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, MI_exp), axis = 2)        
        else:
            EI_norm = BGR2EI(img_BGR)      
            EI_exp = np.expand_dims(EI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, EI_exp), axis = 2)        

    # return (img_exp / 255) # rescale to the range of [0, 1]
    return img_exp   
  


train_list_IDs = train_set.image_name
valid_list_IDs = test_set.image_name
test_list_IDs = df_test.image_name
len_valid = len(valid_list_IDs)
# ID = valid_list_IDs.iloc[0]
img_sizes = [256, 512, 768, 1024]
# methods = ['CV2', 'TurboJpeg', 'SimpleJpeg']
methods = ['CV2', 'SimpleJpeg']
for m in methods:
    print_log(f'Method - {m}', [main_logger])
    start_m = time.perf_counter()
    for s in img_sizes:
        start_s = time.perf_counter()
        print_log(f'  Image size - {s} * {s}', [main_logger])
        for ch in range (3, params['N_CHANNELS'] + 1):
            start_ch = time.perf_counter()
            for i in range(len_valid):
                ID = valid_list_IDs.iloc[i]
                img = _read(train_image_folder_path+ID+".jpg", (s, s, ch), params['COLOR_SPACE' ], toAugment = True, drop_luminosity = False, method = m)
            end_ch = time.perf_counter()
            print_log(f'    # of channels - {img.shape[2]}, time : {end_ch - start_ch:0.3f}s', [main_logger])
            # print_log(f'    # of channels - {ch}, time : {end_ch - start_ch:0.3f}s', [main_logger])
        end_s = time.perf_counter()
        print_log(f'{" "*8} Total time : {end_s - start_s:0.3f}s', [main_logger])
    end_m = time.perf_counter()
    print_log(f'Total time for method {m}: {end_m - start_m:0.3f}s', [main_logger])
    

for logger in loggers:
    for handler in list(loggers[logger].handlers):
        print(handler)
        handler.close()
        loggers[logger].removeHandler(handler) 
            