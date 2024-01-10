# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 22:34:59 2023

@author: alex
"""

import os

import copy
import uuid
import random
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import cv2
import numpy as np
import pandas as pd 
from imgaug import augmenters as iaa
import math
from sklearn.model_selection import KFold
from sklearn.metrics import roc_auc_score, auc #, roc_curve, precision_recall_curve
from sklearn.metrics import average_precision_score, confusion_matrix
# , f1_score, precision_score, classification_report
from sklearn.utils import shuffle
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler

# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, EarlyStopping
import tensorflow.keras.backend as K
from keras.metrics import PrecisionAtRecall

from tensorflow.keras.applications.vgg16 import VGG16
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
# from tensorflow.keras.applications.densenet import DenseNet121
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, \
    EfficientNetV2B1, EfficientNetV2B2
# import tensorflow.keras.applications.efficientnet_v2.preprocess_input as eff_v2_preprocess    
from tensorflow.keras.applications.efficientnet import EfficientNetB0,\
    EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, \
    EfficientNetB5, EfficientNetB6, EfficientNetB7
# import tensorflow.keras.applications.efficientnet.preprocess_input as eff_v1_preprocess    
from tensorflow.keras import mixed_precision
from tqdm import tqdm
from numba import njit

import pickle
import simplejpeg 

from logging_utils import *
from logging_utils import myLogger, print_log, close_loggers 

from plot_utils import plot_mean_ROC, plot_mean_PR

datetime_now = datetime.now().strftime('%Y%m%d')

n_folds = 5
DEVICE = "GPU"
MULTI = False

SEED = 1970
random.seed(SEED)
tf.random.set_seed(SEED)
np.random.seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)

EFNS = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, 
        EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
        EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2,
        ResNet50V2, ResNet101V2, ResNet152V2, 
        VGG16]
nets_names = ['v1b0', 'v1b1', 'v1b2', 'v1b3', 'v1b4', 'v1b5', 'v1b6', 'v1b7',
              'v2b0', 'v2b1', 'v2b2', 'resnet50v2', 'resnet101v2', 'resnet152v2',
              'vgg16']


tb_logs = 'logs_tb'
model_logs = './logs_model/'
inferences = './inferences/'
data_path = './data/'
ISIC_models_path = './models_ISIC/'

models_ISIC = os.listdir(ISIC_models_path)

mean_metrics = 'mean_metrics.csv'
mean_metrics_cols = ['date', 'model', 'image_size', 'batch_size', 'channel', 
                     'test_set','roc_auc', 'pr_auc', 'f1', 'precision', 
                     'recall', 'uuid', 'drop_luminance']
compare_preds_columns = ['isic_id', 'target']
compare_preds_local_test_name = 'compare_saved_preds_local_test.csv'
compare_preds_NoL_test_name = 'compare_saved_preds_NoL_test.csv'
compare_probs_local_test_name = 'compare_saved_probs_local_test.csv'
compare_probs_NoL_test_name = 'compare_saved_probs_NoL_test.csv'
     
train_image_folder_path = "./jpeg/231114_all_data_ISIC/512/"
test_image_folder_path = "./jpeg/231114_all_data_ISIC/512/"     # 15% of all available ISIC data
# train_image_folder_path = "./jpeg/231115_all_data_ISIC/512/"
# test_image_folder_path = "./jpeg/231115_all_data_ISIC/512/"     # 15% of all available ISIC data
# test_Kaggle_image_folder_path = "./jpeg/test/1024/"             # online Kaggle test set for submision to check AUC score
# test_no_lesion_id_image_folder_path = "./jpeg/231115_all_data_ISIC/512/"       # additional local test set of data with no patient_id and no lesion_id, subset of full ISIC dataset
test_no_lesion_id_image_folder_path = "./jpeg/231114_all_data_ISIC/512/"       # additional local test set of data with no patient_id and no lesion_id, subset of full ISIC dataset

main_logger = myLogger('main', f"{model_logs}{datetime_now}-compare-vote-difficulty-main.log")
res_logger = myLogger('result', f"{model_logs}{datetime_now}-compare-vote-difficulty-results.log", False)

print_log('*' * 80, [main_logger])
print_log('Begin logging', [main_logger])
print_log('tensorflow version:' + str(tf.__version__), [main_logger])


pd.set_option('display.max_columns', None)

df_test = pd.read_csv(data_path + '231120_test.csv')                            # 15% of all available ISIC data
df_test_no_lesion_id = pd.read_csv(data_path + '231120_no_lesion_ID.csv')       # additional local test set of data with no patient_id and no lesion_id, subset of full ISIC dataset


if os.path.isfile(data_path + compare_preds_local_test_name):
    df_compare_preds_local_test = pd.read_csv(data_path + compare_preds_local_test_name) 
else:
    df_compare_preds_local_test = pd.DataFrame(columns = compare_preds_columns)
    df_compare_preds_local_test.isic_id = df_test.isic_id
    df_compare_preds_local_test.target = df_test.target
    
if os.path.isfile(data_path + compare_preds_NoL_test_name):
    df_compare_preds_NoL_test = pd.read_csv(data_path + compare_preds_NoL_test_name) 
else:
    df_compare_preds_NoL_test = pd.DataFrame(columns = compare_preds_columns)
    df_compare_preds_NoL_test.isic_id = df_test_no_lesion_id.isic_id
    df_compare_preds_NoL_test.target = df_test_no_lesion_id.target

if os.path.isfile(data_path + compare_probs_local_test_name):
    df_compare_probs_local_test = pd.read_csv(data_path + compare_probs_local_test_name) 
else:
    df_compare_probs_local_test = pd.DataFrame(columns = compare_preds_columns)
    df_compare_probs_local_test.isic_id = df_test.isic_id
    df_compare_probs_local_test.target = df_test.target
    
if os.path.isfile(data_path + compare_probs_NoL_test_name):
    df_compare_probs_NoL_test = pd.read_csv(data_path + compare_probs_NoL_test_name) 
else:
    df_compare_probs_NoL_test = pd.DataFrame(columns = compare_preds_columns)
    df_compare_probs_NoL_test.isic_id = df_test_no_lesion_id.isic_id
    df_compare_probs_NoL_test.target = df_test_no_lesion_id.target


    
def get_lr_callback(batch_size=8):
    lr_start   = params['LR_START']
    lr_max     = params['LR_MAX']
    lr_min     = params['LR_MIN']
    lr_ramp_ep = params['LR_RAMP_EP']
    lr_sus_ep  = params['LR_SUS_EP']
    lr_decay   = params['LR_DECAY']  
    def lrfn_simple(epoch):
        cur_step = epoch % lr_sus_ep  
        cur_cycle = int(epoch / lr_sus_ep)
        step_dec = (lr_start - lr_min) / lr_sus_ep
        step_decay_decrease = (lr_decay ** (1 / (cur_step + 1))) / (2 * 1000)
        cycle_decay_decrease = (lr_decay ** (1 / (cur_cycle + 1))) / (2* 1000)       
        lr = lr_start - step_dec * cur_step + step_decay_decrease - cycle_decay_decrease
        tf.summary.scalar('Learning Rate', data = lr, step = epoch)
        return lr

    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            # lr = (lr_max - lr_start) / (lr_ramp_ep * (epoch + 1)) + lr_start
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start        
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            # lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            lr = (lr_max - lr_min) * lr_decay**(1 / (epoch - lr_ramp_ep - lr_sus_ep + 0.000000001)) + lr_min
            print(f'Epoch {(epoch - lr_ramp_ep - lr_sus_ep):2.4}, exp = {(epoch - lr_ramp_ep - lr_sus_ep):2.4}, decay = {(lr_decay**(1 / (epoch - lr_ramp_ep - lr_sus_ep + 0.000000001))):2.4 }' )
        tf.summary.scalar('Learning Rate', data = lr, step = epoch)
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn_simple, verbose=False)
    # lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback


def get_lr_callback_kaggle(batch_size=8):
    lr_start   = 0.000005
    lr_max     = 0.00000125 * REPLICAS * batch_size
    lr_min     = 0.000001
    lr_ramp_ep = 5
    lr_sus_ep  = 0
    lr_decay   = 0.8
   
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start
            
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
            
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback

def Focal_Loss(y_true, y_pred, alpha = 0.25, gamma = 2, weight = 5):
    """
    Binary Cross Entropy modified to work better with imbalanced datasets
    Parameters
    ----------
    y_true : array, float
        Target.
    y_pred : array, float
        Predicted labels.
    alpha : float, optional
        The default is 0.25.
    gamma : float, optional
        The default is 2.
    weight : float, optional
        The default is 5.
    Returns
    -------
    float
        Computed loss.
    """
    y_true = K.flatten(tf.cast(y_true, tf.float32))
    y_pred = K.flatten(tf.cast(y_pred, tf.float32))

    BCE = K.binary_crossentropy(y_true, y_pred)
    BCE_EXP = K.exp(-BCE)
    alpha = alpha*y_true+(1-alpha)*(1-y_true)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)

    return BCE + weight * focal_loss


def build_model(dim = 128, n_ch = 3, net_ind = 0, dropout = False, focal_loss = False, PrRec_metric = False): 
# def build_model(dim = 128, n_ch = 3, net_ind = 0, dropout = False, focal_loss = False): 
    inp = Input(shape = (dim, dim, n_ch), name = 'Image')
    # x = tf.cast(inp, tf.float32)
    # if net_ind < 8:
    #     x = tf.keras.applications.efficientnet.preprocess_input(x)
    # elif net_ind < 11:
    #     x = tf.keras.applications.efficientnet_v2.preprocess_input(x)
    # elif net_ind >= 11:
    #     x = tf.keras.applications.resnet_v2.preprocess_input(x)
    # else:
    #     print('Error - no such model!')
    #     return None
        
    # base = EFNS[net_ind](input_shape = (dim, dim, n_ch), weights='imagenet', include_top = False)
    base = EFNS[net_ind](input_shape = (dim, dim, n_ch), weights = None, include_top = False)
    x = base(inp)
    # x = base(x)
    x = GlobalAveragePooling2D()(x)
    if dropout:
        x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    if dropout:
        x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid', name = 'Out')(x)
    
    model = tf.keras.Model(inputs = (inp), outputs = out)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.00005)
    pr_metric = PrecisionAtRecall(0.4, num_thresholds = 200, class_id=None, name = 'PatR', dtype=None)
    # pr_metric = PrecisionAtRecall(0.5, num_thresholds = 200, class_id=None, name = 'PatR', dtype=None)
    if PrRec_metric:
        c_metrics = [pr_metric]
    else:
        c_metrics = ['AUC']
    if focal_loss:
        model.compile(optimizer = opt, loss = Focal_Loss, metrics = ['AUC', ])
    else:
        loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
        # model.compile(optimizer = opt, loss = loss, metrics = [pr_metric])
        # model.compile(optimizer = opt, loss = loss, metrics = [pr_metric, 'AUC'])
        # model.compile(optimizer = opt, loss = loss, metrics = ['AUC'])
        model.compile(optimizer = opt, loss = loss, metrics = c_metrics)
    # model.compile(optimizer=opt, loss=loss, metrics=['AUC'])
    # model.summary()
    return model


# Below is TensorFlow code to perform coarse dropout data augmentation on tf.data.Dataset(). 
def dropout(image, DIM=256, PROBABILITY = 0.75, CT = 8, SZ = 0.2):
    # input image - is one image of size [dim,dim,3] not a batch of [b,dim,dim,3]
    # output - image with CT squares of side size SZ*DIM removed
    
    # DO DROPOUT WITH PROBABILITY DEFINED ABOVE
    P = tf.cast( tf.random.uniform([],0,1)<PROBABILITY, tf.int32)
    print(P)
    if (P==0)|(CT==0)|(SZ==0): print('rrr') 
    else: print('ffff')
    
    if (P==0)|(CT==0)|(SZ==0): return image
    
    for k in range(CT):
        # CHOOSE RANDOM LOCATION
        x = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        y = tf.cast( tf.random.uniform([],0,DIM),tf.int32)
        # COMPUTE SQUARE 
        WIDTH = tf.cast( SZ*DIM,tf.int32) * P
        ya = tf.math.maximum(0,y-WIDTH//2)
        yb = tf.math.minimum(DIM,y+WIDTH//2)
        xa = tf.math.maximum(0,x-WIDTH//2)
        xb = tf.math.minimum(DIM,x+WIDTH//2)
        # DROPOUT IMAGE
        one = image[ya:yb,0:xa,:]
        two = tf.zeros([yb-ya,xb-xa,3]) 
        three = image[ya:yb,xb:DIM,:]
        middle = tf.concat([one,two,three],axis=1)
        image = tf.concat([image[0:ya,:,:],middle,image[yb:DIM,:,:]],axis=0)
            
    # RESHAPE HACK SO TPU COMPILER KNOWS SHAPE OF OUTPUT TENSOR 
    image = tf.reshape(image,[DIM,DIM,3])
    return image


# Image Augmentation
sometimes = lambda aug: iaa.Sometimes(0.35, aug)
augmentation = iaa.Sequential([  
                                iaa.Fliplr(0.5),
                                iaa.Flipud(0.5),
                                # iaa.LinearContrast((0.9, 1.1)),
                                # iaa.Multiply((0.9, 1.1), per_channel=0.2),
                                sometimes(iaa.Cutout(nb_iterations=(1, 3), size=0.05, squared=False, fill_mode="constant", cval=0)),
                                sometimes(iaa.Crop(px=(20, 80), keep_size = True, sample_independently = False)),
                                sometimes(iaa.Affine(rotate=(-35, 35))),
                                sometimes(iaa.Affine(scale=(0.95, 1.05)))
                            ], random_order = True)       

@njit   
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


@njit
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
def read_img(path, desired_size, color_space = 'RGB', toAugment = False, drop_luminosity = False, sel_channels = '3'):
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
    with open(path, 'rb') as f:
        jf = f.read() # Read whole file in the file_content string
        if simplejpeg.is_jpeg(jf):
            img_BGR = simplejpeg.decode_jpeg(jf, colorspace = 'bgr')
    
    # to compensate for reduced number of channels passed in the image size        
    if drop_luminosity & (color_space != 'RGB'):
    # if drop_luminosity:
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
    if n_ch == 4:
        if sel_channels == 'EI':
            EI_norm = BGR2EI(img_BGR)      
            EI_exp = np.expand_dims(EI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, EI_exp), axis = 2)        
        else:
            MI_norm = BGR2MI(img_BGR)
            MI_exp = np.expand_dims(MI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, MI_exp), axis = 2)        
    if n_ch == 5:
        if params['4TH_CHANNEL'] == 'EI':
        # if params['4TH_CHANNEL'] == 'EI':
            EI_norm = BGR2EI(img_BGR)      
            EI_exp = np.expand_dims(EI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, EI_exp), axis = 2)        
        else:
            MI_norm = BGR2MI(img_BGR)
            MI_exp = np.expand_dims(MI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, MI_exp), axis = 2)        
    # if n_ch > 4:
        # reverse which channel to add as 5th, since we've alreay added the 4th one
        if params['4TH_CHANNEL'] == 'EI':
            MI_norm = BGR2MI(img_BGR)
            MI_exp = np.expand_dims(MI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, MI_exp), axis = 2)        
        else:
            EI_norm = BGR2EI(img_BGR)      
            EI_exp = np.expand_dims(EI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, EI_exp), axis = 2)        

    return (img_exp / 255) # rescale to the range of [0, 1]
    # return img_exp   
                 
#---------------------------------------------------------------------------------------------
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, list_IDs, labels = None, batch_size = 1, img_size = (512, 512, 1), 
                 img_dir = train_image_folder_path, color_space = 'RGB', testAugment = False, drop_luminosity = False,
                 sel_channels = '3',
                 *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.testAugment = testAugment
        self.color_space = color_space
        self.drop_luminosity = drop_luminosity
        self.sel_channels = sel_channels
        self.on_epoch_end()

    def __len__(self):
        n_batches = int(math.ceil(len(self.indices) / self.batch_size))
        return n_batches

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        list_IDs_temp = [self.list_IDs[k] for k in indices]
        
        if self.labels is not None:
            X, Y = self.__data_generation(list_IDs_temp, indices)
            return X, Y
        else:
            X = self.__data_generation(list_IDs_temp)
            return X
        
    def on_epoch_end(self):
        
        if self.labels is not None: 
            self.indices = np.array(self.list_IDs.index)
        else:
            self.indices = np.array(self.list_IDs.index)

    def __data_generation(self, list_IDs_temp, idxs = None):
        X = np.empty((self.batch_size, *self.img_size))
         # print("Length : ", len(list_IDs_temp), ' : ', list_IDs_temp)
        
        if self.labels is not None: # training phase
            Y = np.empty((self.batch_size), dtype=np.float16)
        
            # ID is a filename
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = read_img(self.img_dir+ID+".jpg", self.img_size, self.color_space, toAugment = True, 
                                 drop_luminosity = self.drop_luminosity, sel_channels = self.sel_channels)
                # X[i,] = _read(self.img_dir+ID+".jpg", self.img_size, self.color_space, toAugment = True, drop_luminosity = self.drop_luminosity)
                Y[i,] = self.labels.loc[idxs[i]]       
            return X, Y

        elif self.testAugment: # test phase with Augmentation
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = read_img(self.img_dir+ID+".jpg", self.img_size, self.color_space, toAugment = True, 
                                 drop_luminosity = self.drop_luminosity, sel_channels = self.sel_channels)
                # X[i,] = _read(self.img_dir+ID+".jpg", self.img_size, self.color_space, toAugment = True, drop_luminosity = self.drop_luminosity)
            return X

        else: # test phase no Augmentation
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = read_img(self.img_dir+ID+".jpg", self.img_size, self.color_space, toAugment = False, 
                                 drop_luminosity = self.drop_luminosity, sel_channels = self.sel_channels)
                # X[i,] = _read(self.img_dir+ID+".jpg", self.img_size, self.color_space, toAugment = False, drop_luminosity = self.drop_luminosity)
            return X


#---------------------------------------------------------------------------------------------
class TestSet():

    def __init__(self, list_IDs, labels = None, batch_size = 1, img_size = (512, 512, 1), channels = ['3', 'EI', 'MI', '5'],
                 img_dir = None, color_space = 'RGB', testAugment = False, drop_luminosity = False,
                 sel_channels = '3', n_folds = 5, name = '',
                 *args, **kwargs):

        self.name = name
        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.testAugment = testAugment
        self.color_space = color_space
        self.drop_luminosity = drop_luminosity
        self.sel_channels = sel_channels       

        self.roc_aucs =     {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.pr_aucs =      {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.f1s =          {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.recalls =      {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.precisions =   {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.predictions =  {'3': np.zeros((len(list_IDs), n_folds)), 
                              'EI': np.zeros((len(list_IDs), n_folds)), 
                              'MI': np.zeros((len(list_IDs), n_folds)), 
                              '5': np.zeros((len(list_IDs), n_folds)), 
                              'RGB': np.zeros((len(list_IDs), n_folds))}
        self.roc_aucs_TTA =     {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.pr_aucs_TTA =      {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.f1s_TTA =          {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.recalls_TTA =      {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.precisions_TTA =   {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.predictions_TTA =  {'3': np.zeros((len(list_IDs), n_folds)), 
                                  'EI': np.zeros((len(list_IDs), n_folds)), 
                                  'MI': np.zeros((len(list_IDs), n_folds)), 
                                  '5': np.zeros((len(list_IDs), n_folds)), 
                                  'RGB': np.zeros((len(list_IDs), n_folds))}
        self.roc_aucs_mean =    {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.pr_aucs_mean =     {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.precisions_mean =  {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.recalls_mean =     {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        self.f1s_mean =         {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
        

    def __len__(self):
        return len(self.list_IDs)

    def get_test_gen(self, cur_img_size, cur_color_space, cur_channels):
        return DataGenerator(self.list_IDs, self.labels, self.batch_size, cur_img_size, 
                             self.img_dir, cur_color_space, self.testAugment, self.drop_luminosity, cur_channels)     
                 
    def get_last_metrics(self, cur_channel, cur_fold):
        roc_auc = roc_auc_score(self.labels, self.predictions[cur_channel][:, cur_fold])
        precision, recall, f1, _ = precision_recall_fscore_support(self.labels, self.predictions[cur_channel][:, cur_fold] >= 0.5, average = None)
        prec, rec, _ = precision_recall_curve(self.labels, self.predictions[cur_channel][:, cur_fold] )
        auc_precision_recall = auc(rec, prec)
        self.roc_aucs[cur_channel].append(roc_auc)
        self.pr_aucs[cur_channel].append(auc_precision_recall)
        self.precisions[cur_channel].append(precision[1])
        self.recalls[cur_channel].append(recall[1])
        self.f1s[cur_channel].append(f1[1])
        pref = f'### Fold {cur_fold + 1}, {cur_channel} channels, -{self.name}-, without TTA, metrics'
        print_log(f'{pref:<70}: ROC AUC = {roc_auc:.4f}, PR AUC = {auc_precision_recall:.4f}, F1 = {f1[1]:.4f}, precision = {precision[1]:.4f}, recall = {recall[1]:.4f}', [main_logger, res_logger])
        cm = confusion_matrix(self.labels, self.predictions[cur_channel][:, cur_fold] >= 0.5)
        print_log(f'Confusion matrix for {cur_channel} channel, fold {cur_fold + 1}', [main_logger])
        print_log(cm, [main_logger])

    def get_avg_metrics(self, cur_channel):
        preds = self.predictions[cur_channel].mean(axis=1)
        roc_auc = roc_auc_score(self.labels, preds)
        precision, recall, f1, _ = precision_recall_fscore_support(self.labels, preds >= 0.5, average = None)
        prec, rec, _ = precision_recall_curve(self.labels, preds)
        # prec, rec, _ = precision_recall_curve(self.labels, preds >= 0.5)
        auc_precision_recall = auc(rec, prec)

        self.roc_aucs_mean[cur_channel].append(roc_auc)
        self.pr_aucs_mean[cur_channel].append(auc_precision_recall)
        self.precisions_mean[cur_channel].append(precision[1])
        self.recalls_mean[cur_channel].append(recall[1])
        self.f1s_mean[cur_channel].append(f1[1])
        pref = f'### {cur_channel} channels, -{self.name}-, averaged across 5 folds, without TTA, metrics'
        print_log(f'{pref:<85}: ROC AUC = {roc_auc:.4f}, PR AUC = {auc_precision_recall:.4f}, F1 = {f1[1]:.4f}, precision = {precision[1]:.4f}, recall = {recall[1]:.4f}', [main_logger, res_logger])


test_list_IDs = df_test.isic_id
test_no_lesion_id_list_IDs = df_test_no_lesion_id.isic_id

VERBOSE = 1

oof_preds = {}
oof_preds_TTA = {}

oof_preds_w = {}
oof_preds_w_TTA = {}



# for c_model in models_ISIC[:1]:
# for c_model in models_ISIC[1:-1]:
for c_model in models_ISIC[:-1]:
    print_log('=' * 80, [main_logger, res_logger])
    print_log(f'Inference for {c_model}', [main_logger, res_logger])
    model_path = ISIC_models_path + c_model + '/'
    all_model_files = os.listdir(model_path)
    tests_files = [f for f in all_model_files if ('CV' in f) & ('.testobj' in f)]
    
    if len(tests_files) < 2:
        print_log('Wrong number of saved test sets objects for the model, skipping!', [main_logger, res_logger])
    else:       
        # tests_files2 = [f for f in all_model_files if ('CV' in f) & ('.testobj' in f)]
        # tests_files = [f for f in tests_files if 'CV' in f]
        test_loc_name = tests_files[0]
        test_NoL_name = tests_files[1]
        # test_loc_name = c_model.split('-')[0] + '_test_loc.testobj'
        # test_NoL_name = c_model.split('-')[0] + '_test_NoL.testobj'
    
    try:
        with open(f"{model_path}{test_loc_name}", 'rb') as f: 
            test_set_loc = pickle.load(f)               
        with open(f"{model_path}{test_NoL_name}", 'rb') as f: 
            test_set_NoL = pickle.load(f)               
    except:
        print_log('Could not load saved test sets objects for the model, skipping!', [main_logger, res_logger])
        continue
    
    param_file = [f for f in all_model_files if '.params' in f]
    if len(param_file) < 1:
        print_log(f'Skipping {c_model} - no saved parametres', [main_logger, res_logger])
        continue
    #  only 1 param file supposed to exist
    with open(model_path + param_file[0], 'rb') as f: 
    # with open(param_file[0], 'rb') as f: 
        params = pickle.load(f)               
    if params['COMPUTE_RGB']:
        channels = ['RGB', '3', 'EI', 'MI', '5']
        weights  = {'RGB' : [], '3':[], 'EI':[], 'MI':[], '5':[]}
    else:
        channels = ['3', 'EI', 'MI', '5']
        weights  = {'3':[], 'EI':[], 'MI':[], '5':[]}
   
    MM_scaler = MinMaxScaler()
    
    print_log('\n'+'-'*80, [main_logger, res_logger])
    for n_ch, ch in enumerate(channels): # running the same model with inputs of just RGB, RGB+EI, RGB+MI, RGB+MI+EI for each fold
        test_set_loc.get_avg_metrics(ch)
        cm = confusion_matrix(test_set_loc.labels, test_set_loc.predictions[ch].mean(axis=1) >= 0.5)
        print_log(f'Confusion matrix for {ch}', [main_logger])
        print_log(cm, [main_logger])
        
        test_set_NoL.get_avg_metrics(ch)
        cm = confusion_matrix(test_set_NoL.labels, test_set_NoL.predictions[ch].mean(axis=1) >= 0.5)
        print_log(f'Confusion matrix for {ch}', [main_logger])
        print_log(cm, [main_logger])
           
        # compute weight based on AUC on validation set for each fold 
        # will use to give more weight to better folds for predicted target 
        scaler = (np.array(test_set_loc.roc_aucs[ch]).max() - np.array(test_set_loc.roc_aucs[ch]).min()+1) 
        weights[ch]= (test_set_loc.roc_aucs[ch] - np.array(test_set_loc.roc_aucs[ch]).min()/2) / scaler
        
        # weighted average between folds/models
        oof_preds_w[ch] = np.squeeze(test_set_loc.predictions[ch]).dot(weights[ch])
        op = MM_scaler.fit_transform(oof_preds_w[ch].reshape(-1, 1))
        auc_w = roc_auc_score(test_set_loc.labels, op)
        pr_Score , recall, f1, _ = precision_recall_fscore_support(test_set_loc.labels, op >= 0.5, average = None)
        prec, rec, thr = precision_recall_curve(test_set_loc.labels, op)
        auc_precision_recall = auc(rec, prec)
        AP_score = average_precision_score(test_set_loc.labels, op)
        ts = f'    Mean weighted over 5 folds for {ch} channels over local test set, no TTA'
        print_log(f'{ts:<85}- ROC AUC = {auc_w:.4f}, PR ROC = {auc_precision_recall:.4f}, f1 = {f1[1]:0.4f}, precision = {pr_Score[1]:0.4f}, recall = {recall[1]:0.4f}, Average precision score = {AP_score:0.4f}', [main_logger, res_logger])
        cm = confusion_matrix(test_set_loc.labels, op >= 0.5)
        print_log(f'Confusion matrix for {ch}, mean weighted', [main_logger])
        print_log(cm, [main_logger])
        
        df_compare_preds_local_test[c_model + '#' + ch] = (op.reshape(-1) >= 0.5)
        df_compare_probs_local_test[c_model + '#' + ch] = op.reshape(-1)
        
        scaler = (np.array(test_set_NoL.roc_aucs[ch]).max() - np.array(test_set_NoL.roc_aucs[ch]).min()+1) 
        weights[ch]= (test_set_NoL.roc_aucs[ch] - np.array(test_set_NoL.roc_aucs[ch]).min()/2) / scaler  
        # weighted average between folds/models
        oof_preds_w[ch] = np.squeeze(test_set_NoL.predictions[ch]).dot(weights[ch])
        op = MM_scaler.fit_transform(oof_preds_w[ch].reshape(-1, 1))
        auc_w = roc_auc_score(test_set_NoL.labels, op)
        pr_Score , recall, f1, _ = precision_recall_fscore_support(test_set_NoL.labels, op >= 0.5, average = None)
        prec, rec, thr = precision_recall_curve(test_set_NoL.labels, op)
        auc_precision_recall = auc(rec, prec)
        AP_score = average_precision_score(test_set_NoL.labels, op)
        ts = f'    Mean weighted over 5 folds for {ch} channels over NoL test set, no TTA'
        print_log(f'{ts:<85}- ROC AUC = {auc_w:.4f}, PR ROC = {auc_precision_recall:.4f}, f1 = {f1[1]:0.4f}, precision = {pr_Score[1]:0.4f}, recall = {recall[1]:0.4f}, Average precision score = {AP_score:0.4f}', [main_logger, res_logger])
        cm = confusion_matrix(test_set_NoL.labels, op >= 0.5)
        print_log(f'Confusion matrix for {ch}, mean weighted', [main_logger])
        print_log(cm, [main_logger])

        df_compare_preds_NoL_test[c_model + '#' + ch] = (op.reshape(-1) >= 0.5)
        df_compare_probs_NoL_test[c_model + '#' + ch] = op.reshape(-1)
    
        print_log('\n', [main_logger, res_logger])
 
df_compare_preds_local_test.to_csv(data_path + compare_preds_local_test_name, index = False)
df_compare_preds_NoL_test.to_csv(data_path + compare_preds_NoL_test_name, index = False)

df_compare_probs_local_test.to_csv(data_path + compare_probs_local_test_name, index = False)
df_compare_probs_NoL_test.to_csv(data_path + compare_probs_NoL_test_name, index = False)

close_loggers(loggers)

