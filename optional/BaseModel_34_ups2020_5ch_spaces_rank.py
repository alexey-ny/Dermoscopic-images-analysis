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
from sklearn.metrics import roc_auc_score, f1_score, precision_score, classification_report
from sklearn.utils import shuffle
# import keras
import tensorflow as tf
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint
import tensorflow.keras.backend as K

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras import mixed_precision
import tensorflow_addons as tfa

import numba
from numba import njit

import pickle
import simplejpeg 

n_folds = 5
DEVICE = "GPU"
MULTI = False

SEED = 1970


EFNS = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, 
        EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
        EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2]
nets_names = ['v1b0', 'v1b1', 'v1b2', 'v1b3', 'v1b4', 'v1b5', 'v1b6', 'v1b7',
              'v2b0', 'v2b1', 'v2b2']

params = {
    # 'IMG_SIZES' :   [768] * n_folds,
    'IMG_SIZES' :   [256] * n_folds,
    # 'IMG_SIZES' :   [128] * n_folds,
    # 'EFF_NETS' :    [10] * n_folds,
    'EFF_NETS' :    [0] * n_folds,
    # 'BATCH_SIZES' : [32] * n_folds,
    'BATCH_SIZES' : [16] * n_folds,
    # 'BATCH_SIZES' : [128] * n_folds,
    'EPOCHS' : [3] * n_folds,
    # 'EPOCHS' : [10] * n_folds,
    # number of times to duplicate the malignant samples, to use with augmentation to balance the classes
    'MAL_UPSAMPLE' : 10, 
    'FIT_IN_RAM' : True, # if we shall load all images for the training into the RAM 
    # test time augmentation steps
    'TTA' : 0,
    # possible values are 'RGB', 'HSV' and 'YCrCb'
    'COLOR_SPACE' : 'HSV' ,
    'DROP_LUM_CH' : False, # if we should drop Luminosity channel in HSV of YCrCb spaces
    '4TH_CHANNEL' : 'EI' , # either EI or MI, to estimate if one is more important than the other
    'LR_START' : 0.001,
    'LR_MAX'   : 0.009,
    'LR_MIN'   : 0.00005,
    # 'LR_START' : 0.0005,
    # 'LR_MAX' : 0.0015,
    # 'LR_MIN' : 0.000001,
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
        handler = logging.FileHandler(f"{model_logs}{datetime.now().strftime('%Y%m%d')}-{nets_names[params['EFF_NETS'][0]]}_img{str(params['IMG_SIZES'][0])}-{name}.log")
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
    

def get_lr_callback(batch_size=8):
    lr_start   = params['LR_START']
    lr_max     = params['LR_MAX']
    lr_min     = params['LR_MIN']
    lr_ramp_ep = params['LR_RAMP_EP']
    lr_sus_ep  = params['LR_SUS_EP']
    lr_decay   = params['LR_DECAY']  
    def lrfn(epoch):
        if epoch < lr_ramp_ep:
            lr = (lr_max - lr_start) / (lr_ramp_ep * (epoch + 1)) + lr_start
            # lr = (lr_max - lr_start) / lr_ramp_ep * epoch + lr_start        
        elif epoch < lr_ramp_ep + lr_sus_ep:
            lr = lr_max
        else:
            lr = (lr_max - lr_min) * lr_decay**(epoch - lr_ramp_ep - lr_sus_ep) + lr_min
            # lr = (lr_max - lr_min) * lr_decay**(1 / (epoch - lr_ramp_ep - lr_sus_ep)) + lr_min
        tf.summary.scalar('Learning Rate', data = lr, step = epoch)
        return lr

    lr_callback = tf.keras.callbacks.LearningRateScheduler(lrfn, verbose=False)
    return lr_callback
    
tb_logs = 'logs_tb'
model_logs = './logs_model/'

# train_image_folder_path = "F:\\melanoma_alex\\jpeg\\train\\1024\\"
# test_image_folder_path = "F:\\melanoma_alex\\jpeg\\test\\1024\\"
train_image_folder_path = "./jpeg/train/1024/"
test_image_folder_path = "./jpeg/test/1024/"
data_path = './data/'

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

main_logger = myLogger('main')
res_logger = myLogger('results', False)

print_log('*' * 80, [main_logger])
print_log('Begin logging', [main_logger])
print_log('tensorflow version:' + str(tf.__version__), [main_logger])

print_log('Compute dtype: %s' % policy.compute_dtype, [main_logger])
print_log('Variable dtype: %s' % policy.variable_dtype, [main_logger])

gpu_devices = tf.config.list_logical_devices('GPU')
if gpu_devices:
    for gpu_device in gpu_devices:
        print_log('device available:' + str(gpu_device), [main_logger])

if DEVICE != "TPU":
    if MULTI:
        print_log("Using default strategy for multiple GPUs", [main_logger])
        strategy = tf.distribute.MirroredStrategy(gpu_devices)
    else:
        print_log("Using default strategy for CPU and single GPU", [main_logger])
        strategy = tf.distribute.get_strategy()
   
AUTO = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print_log(f'Number of Replicas Used: {REPLICAS}', [main_logger])

pd.set_option('display.max_columns', None)

df_train = pd.read_csv(data_path + 'train.csv')
df_test = pd.read_csv(data_path + 'test.csv')
cols = df_train.columns.to_list()

df_mal_2019 = pd.read_csv(data_path + 'train_malig_3_2019.csv')

# lets create a test set for the Reject Classifier, 
# loading stratified lists by Patient_ID from the BaseModel 20
# to prevent leakage through similar images of the same patient
train_idx = np.load(data_path + 'train_patients_IDs.npy', allow_pickle = True)
test_idx = np.load(data_path + 'test_patients_IDs.npy', allow_pickle = True)

test_set = df_train.loc[df_train['patient_id'].isin(test_idx)].reset_index(drop = True)
# test_set = df_train.loc[df_train['patient_id'].isin(test_idx)]
trn_set = df_train.loc[df_train['patient_id'].isin(train_idx)]
mals_2020 = np.array(trn_set.loc[trn_set.target == 1].index).astype('int32')
idx_trn = np.array(trn_set.index).astype('int32')
for m in range(params['MAL_UPSAMPLE']):
    idx_trn = np.concatenate((idx_trn, mals_2020))   
np.random.shuffle(idx_trn)

train_ext = trn_set.loc[idx_trn]

train_set = pd.concat([train_ext, df_mal_2019[trn_set.columns]]).reset_index(drop = True)
# train_set = pd.concat([trn_set, df_mal_2019[trn_set.columns]]).reset_index(drop = True)
train_IP_set = train_set['patient_id'].unique()

y_test_set = test_set['target']
test_set.drop(columns = ['target'], inplace = True, axis = 1)
print_log(y_test_set.value_counts(), [main_logger])
y_train_set = train_set['target']
print_log(y_train_set.value_counts(), [main_logger])

print_log(f"Color space: {params['COLOR_SPACE']}; 4th channel: {params['4TH_CHANNEL']}; Drop luminosity channel: {params['DROP_LUM_CH']}", [main_logger])

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


def build_model(dim = 128, n_ch = 3, ef = 0, dropout = False):
    inp = Input(shape = (dim, dim, n_ch), name = 'Image')
    base = EFNS[ef](input_shape = (dim, dim, n_ch), weights=None, include_top = False)
    x = base(inp)
    x = GlobalAveragePooling2D()(x)
    if dropout:
        x = Dropout(0.1)(x)
    x = Dense(64, activation='relu')(x)
    if dropout:
        x = Dropout(0.3)(x)
    out = Dense(1, activation='sigmoid', name = 'Out')(x)
    
    model = tf.keras.Model(inputs = (inp), outputs = out)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.00005)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
    # f1 = tfa.metrics.F1Score(num_classes=2, average="macro")
    model.compile(optimizer=opt, loss=loss, metrics=['AUC'])
    # model.compile(optimizer=opt, loss=loss, metrics=['AUC', tfa.metrics.F1Score(num_classes=2, average="macro")])
    # model.compile(optimizer=opt, loss=loss, metrics=['AUC', f1])
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
                                iaa.LinearContrast((0.9, 1.1)),
                                iaa.Multiply((0.9, 1.1), per_channel=0.2),
                                sometimes(iaa.Cutout(nb_iterations=(1, 3), size=0.05, squared=False, fill_mode="constant", cval=0)),
                                sometimes(iaa.Crop(px=(20, 80), keep_size = True, sample_independently = False)),
                                # sometimes(iaa.Affine(rotate=(-35, 35))),
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
def _read(path, desired_size, color_space = 'RGB', toAugment = False, drop_luminosity = False):
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

    return (img_exp / 255) # rescale to the range of [0, 1]
    # return img_exp   
                 
#---------------------------------------------------------------------------------------------
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, list_IDs, labels = None, batch_size = 1, img_size = (512, 512, 1), 
                 img_dir = train_image_folder_path, color_space = 'RGB', testAugment = False, drop_luminosity = False,
                 *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.testAugment = testAugment
        self.color_space = color_space
        self.drop_luminosity = drop_luminosity
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
                X[i,] = _read(self.img_dir+ID+".jpg", self.img_size, self.color_space, toAugment = True, drop_luminosity = self.drop_luminosity)
                Y[i,] = self.labels.loc[idxs[i]]       
            return X, Y

        elif self.testAugment: # test phase with Augmentation
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir+ID+".jpg", self.img_size, self.color_space, toAugment = True, drop_luminosity = self.drop_luminosity)
            return X

        else: # test phase no Augmentation
            for i, ID in enumerate(list_IDs_temp):
                X[i,] = _read(self.img_dir+ID+".jpg", self.img_size, self.color_space, toAugment = False, drop_luminosity = self.drop_luminosity)
            return X

                 
#---------------------------------------------------------------------------------------------
class DataGeneratorRAM(tf.keras.utils.Sequence):

    def __init__(self, list_IDs, images_array_name = 'train', labels = None, batch_size = 1, img_size = (512, 512, 1), 
                 color_space = 'RGB', testAugment = False, drop_luminosity = False,
                 *args, **kwargs):

        self.list_IDs = list_IDs
        self.images_array_name = images_array_name
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.testAugment = testAugment
        self.color_space = color_space
        self.drop_luminosity = drop_luminosity
        self.on_epoch_end()

    def __len__(self):
        n_batches = int(math.ceil(len(self.indices) / self.batch_size))
        return n_batches

    def __getitem__(self, index):
        indices = self.indices[index*self.batch_size:(index+1)*self.batch_size]
        # print(indices)
        # list_IDs_temp = [self.list_IDs[k] for k in indices]
        
        if self.labels is not None:
            X, Y = self.__data_generation(indices)
            # X, Y = self.__data_generation(list_IDs_temp, indices)
            return X, Y
        else:
            X = self.__data_generation(indices)
            # X = self.__data_generation(list_IDs_temp)
            return X
        
    def on_epoch_end(self):
        self.indices = np.array(self.list_IDs.index)
        
        # if self.labels is not None: 
        #     self.indices = np.array(self.list_IDs.index)
        # else:
        #     self.indices = np.array(self.list_IDs.index)
            
    def get_elem(self, arr_name, ind, toAugment = False):
    # def __get_elem__(self, arr_name, ind):
        if arr_name == 'train':
            img_BGR = train_array[ind]
        elif arr_name == 'test':
            img_BGR = final_test_array[ind]
        elif arr_name == 'cross-validation':
            img_BGR = valid_array[ind]
        elif arr_name == 'test-validation':
            img_BGR = validation_test_array[ind]
        else:
            print(f'Wrong dataset/array name - {arr_name}')
            img_BGR = None
        # to compensate for reduced number of channels passed in the image size        
        if self.drop_luminosity:
            n_ch = self.img_size[2] + 1
        else:
            n_ch = self.img_size[2]
        
        if toAugment: 
            img_BGR = augmentation.augment_image(img_BGR)      
        
        if self.img_size[0] != img_BGR.shape[0]:
            img_BGR = cv2.resize(img_BGR, self.img_size[:2], interpolation=cv2.INTER_LINEAR)
            
        if self.color_space == 'HSV':
            img_exp = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2HSV)
            if self.drop_luminosity:
                img_exp  = img_exp[:, :, :2]
        elif self.color_space == 'YCrCb': 
            img_exp = cv2.cvtColor(img_BGR, cv2.COLOR_BGR2YCrCb )
            if self.drop_luminosity:
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

        return (img_exp / 255) # rescale to the range of [0, 1]

    def __data_generation(self, idxs ):
    # def __data_generation(self, list_IDs_temp, idxs = None):
        X = np.empty((self.batch_size, *self.img_size))
         # print("Length : ", len(list_IDs_temp), ' : ', list_IDs_temp)
        
        if self.labels is not None: # training phase
            Y = np.empty((self.batch_size), dtype=np.float16)
        
            # ID is an index of the image in the array
            for i, ID in enumerate(idxs):
            # for i, ID in enumerate(list_IDs_temp):
                X[i,] = self.get_elem(self.images_array_name, ID, True)
                Y[i,] = self.labels.loc[idxs[i]]       
            return X, Y

        elif self.testAugment: # test phase with Augmentation
            for i, ID in enumerate(idxs):
            # for i, ID in enumerate(list_IDs_temp):
                X[i,] = self.get_elem(self.images_array_name, ID, True)
            return X

        else: # test phase no Augmentation
            for i, ID in enumerate(idxs):
            # for i, ID in enumerate(list_IDs_temp):
                X[i,] = self.get_elem(self.images_array_name, ID, False)
            return X


# csv_logger = CSVLogger(model_logs + datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + 'effnet_training_log.csv', separator = ',', append = True)

logdir = "logs_tb/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = TensorBoard(log_dir=logdir)

train_list_IDs = train_set.image_name[:200]
valid_list_IDs = test_set.image_name[:32]
test_list_IDs = df_test.image_name[:32]

# train_list_IDs = train_set.image_name
# valid_list_IDs = test_set.image_name
# test_list_IDs = df_test.image_name

VERBOSE = 1

skf = KFold(n_splits = n_folds, shuffle = True, random_state = SEED)
oof_preds = {}
# oof_pred_rank = {}
oof_preds_rank = {}
oof_preds_w = {}
oof_preds_rank_w = {}
oof_pred = {3:[], 4:[], 5:[]}
oof_pred_rank = {3:[], 4:[], 5:[]}
# val_pred = {3:[], 4:[], 5:[]}

preds = {3: np.zeros((len(test_list_IDs), n_folds)), 4: np.zeros((len(test_list_IDs), n_folds)), 5: np.zeros((len(test_list_IDs), n_folds))}
preds_rank = {3: np.zeros((len(test_list_IDs), n_folds)), 4: np.zeros((len(test_list_IDs), n_folds)), 5: np.zeros((len(test_list_IDs), n_folds))}

models = {3:[], 4:[], 5:[]}
oof_aucs = {3:[], 4:[], 5:[]}
weights = {3:[], 4:[], 5:[]}
f1s = {3:[], 4:[], 5:[]}
precisions = {3:[], 4:[], 5:[]}

drop_luminosity_ch = params['DROP_LUM_CH']
cur_color_space = params['COLOR_SPACE']


def load_files_to_RAM(path, df, list_IDs = None):
    files_in_RAM = []
    if list_IDs is None:
        list_IDs = np.array(df.index)
    # if list_IDs is None:
    #     num_files = df.shape[0]
    # else:
    #     num_files = len(list_IDs)
    # for i in range(num_files):
    # for i, ind in enumerate(list_IDs):
    for ind in list_IDs:
        cur = df.iloc[ind]
        with open(f'{path}{cur.image_name}.jpg', 'rb') as f:
            jf = f.read() # Read whole file in the file_content string
            if simplejpeg.is_jpeg(jf):
                img_BGR = simplejpeg.decode_jpeg(jf, colorspace = 'bgr')
        if img_BGR is None:
            main_logger.warning('Error load image:', path)
        files_in_RAM.append(img_BGR)
    
    return np.array(files_in_RAM)


fit_in_RAM = params["FIT_IN_RAM"]   
if fit_in_RAM :
    final_test_array = load_files_to_RAM(test_image_folder_path, df_test, np.array(test_list_IDs.index))
    # validation_test_array = load_files_to_RAM(train_image_folder_path, test_set, np.array(test_set.index))
    validation_test_array = load_files_to_RAM(train_image_folder_path, test_set, np.array(test_set.index)[:32])
    whole_train_array = load_files_to_RAM(train_image_folder_path, train_set.iloc[:256])
    # whole_train_array = load_files_to_RAM(train_image_folder_path, train_set)

for fold, (idxT, idxV) in enumerate(skf.split(train_IP_set)):
    cur_batch_size = params['BATCH_SIZES'][fold]

    print_log('-' * 80, [main_logger, res_logger])
    print_log(f'Fold #: {fold+1}, Model: {EFNS[params["EFF_NETS"][fold]].__name__}, BS: {params["BATCH_SIZES"][fold]}, Image Size: {params["IMG_SIZES"][fold]}', [main_logger, res_logger])
    train_IPs = train_IP_set[idxT]
    val_IPs = train_IP_set[idxV]

    train_image_names = train_set.loc[train_set.patient_id.isin(train_IPs), 'image_name']
    train_image_names = train_image_names.iloc[:32]
    val_image_names = train_set.loc[train_set.patient_id.isin(val_IPs), 'image_name']
    val_image_names= val_image_names.iloc[:32]
    all_image_names = pd.concat([train_image_names, val_image_names])
    # train_image_names = train_set.loc[train_set.patient_id.isin(train_IPs), 'image_name']
    # val_image_names = train_set.loc[train_set.patient_id.isin(val_IPs), 'image_name']
    train_I = train_image_names.index.to_numpy()
    val_I = val_image_names.index.to_numpy()

    y_train_set = train_set.loc[train_set.patient_id.isin(train_IPs), 'target'].iloc[:32]
    y_val_set = train_set.loc[train_set.patient_id.isin(val_IPs), 'target'].iloc[:32]

    if fit_in_RAM:   
        # train_array = load_files_to_RAM(train_image_folder_path, train_set, np.array(all_image_names.index))
        # train_array = load_files_to_RAM(train_image_folder_path, train_set, np.array(train_image_names.index))
        # valid_array = load_files_to_RAM(train_image_folder_path, train_set, np.array(val_image_names.index))
        train_array =  whole_train_array[np.array(train_image_names.index)]
        valid_array =  whole_train_array[np.array(val_image_names.index)]
        
    for ch in range (3, params['N_CHANNELS'] + 1): # running the same model with inputs of just RGB, RGB+MI, RGB+MI+EI for each fold
        print_log('-'*25, [main_logger])
        
        if (cur_color_space != 'RGB') & (drop_luminosity_ch):
            cur_num_channels = ch-1
        else:
            cur_num_channels = ch
            
        print_log(f'Number of channels: {cur_num_channels}', [main_logger])

        # with strategy.scope():
        with tf.device('/GPU:0'):
            
            models[ch] = build_model(dim = params['IMG_SIZES'][fold], n_ch = cur_num_channels, ef = params['EFF_NETS'][fold], dropout = False)
            cur_img_size = (params['IMG_SIZES'][fold], params['IMG_SIZES'][fold], cur_num_channels)
            print_log(f'Image Size: {cur_img_size}', [main_logger])
            
            if fit_in_RAM:   
                valid_gen = DataGeneratorRAM(valid_list_IDs, 'test-validation', labels = None, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)
                test_gen = DataGeneratorRAM(test_list_IDs, 'test', labels = None, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = test_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)       
            else:
                valid_gen = DataGenerator(valid_list_IDs, labels = None, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)
                test_gen = DataGenerator(test_list_IDs, labels = None, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = test_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)       
            
            # y_train_set = train_set.loc[train_set.patient_id.isin(train_IPs), 'target']
            # y_val_set = train_set.loc[train_set.patient_id.isin(val_IPs), 'target']
            # mals_t = np.array(y_train_set.loc[y_train_set==1].index).astype('int32')
            
            # idxT_ext = train_I.copy()
            # for m in range(params['MAL_UPSAMPLE']):
            #     idxT_ext = np.concatenate((idxT_ext, mals_t))   
        
            idxT_ext = shuffle(train_I, random_state = SEED)
            # idxT_ext = shuffle(idxT_ext, random_state = SEED)
            x_trn_ext = train_image_names.loc[idxT_ext].reset_index(drop = True)
            y_trn_ext = y_train_set.loc[idxT_ext].reset_index(drop = True)
            # maybe breaking regular, not fit in RAM validation
            x_val_set = val_image_names.reset_index(drop = True)
            y_val_set = y_val_set.reset_index(drop = True)
            
            if fit_in_RAM:   
                trn_gen = DataGeneratorRAM(x_trn_ext, 'train', labels = y_trn_ext, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)
                # val_gen = DataGeneratorRAM(val_image_names, "cross-validation", labels = y_val_set, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)
                val_gen = DataGeneratorRAM(x_val_set, "cross-validation", labels = y_val_set, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)
            else:
                trn_gen = DataGenerator(x_trn_ext, labels = y_trn_ext, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)
                val_gen = DataGenerator(val_image_names, labels = y_val_set, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)
        
            # SAVE BEST MODEL EACH FOLD
            sv = ModelCheckpoint(
                f'fold_{fold}-{cur_num_channels}.h5', monitor='val_loss', verbose=1, save_best_only=True,
                save_weights_only=True, mode='min', save_freq='epoch')    
        
            history = models[ch].fit(
                trn_gen, 
                epochs = params['EPOCHS'][fold], 
                # callbacks = [sv, csv_logger, tensorboard_callback, get_lr_callback(cur_batch_size), checkInputs()], 
                callbacks = [sv, tensorboard_callback, get_lr_callback(cur_batch_size)], 
                # callbacks = [sv, csv_logger, tensorboard_callback, get_lr_callback(cur_batch_size)], 
                validation_data = val_gen,
                verbose = VERBOSE
            )
            print_log(f'Loading best model for fold {fold+1} with {cur_num_channels} channels', [main_logger])
            models[ch].load_weights(f'fold_{fold}-{cur_num_channels}.h5') 
        
            # PREDICT TEST USING TTA
            print_log('Predicting Test without TTA...', [main_logger])
            pred = models[ch].predict(test_gen, verbose = 1)[:len(test_list_IDs)]  
            df_pred = pd.DataFrame(pred[:,0], columns=['pred'] )
            preds_rank[ch][:, fold] += df_pred['pred'].rank(pct = True)
            # preds[ch][:, fold] += pred[:,0] 
            preds[ch][:, fold] += pred[:,0] 
        
            # PREDICT OOF USING TTA
            print_log('Predicting OOF without TTA...', [main_logger])
            val_pr = models[ch].predict(valid_gen, verbose = 1)[:len(valid_list_IDs)]
            # val_pred[ch] = models[ch].predict(valid_gen, verbose = 1)[:len(valid_list_IDs)]
            df_val = pd.DataFrame(val_pr[:,0], columns=['pred'] )
            val_rank = df_val['pred'].rank(pct = True)
            oof_pred[ch].append(val_pr)   
            oof_pred_rank[ch].append(val_rank)   
            # oof_pred[ch].append(val_pred[ch])   
            
            # REPORT RESULTS
            auc = roc_auc_score(y_test_set[:32], val_pr)
            auc_rank = roc_auc_score(y_test_set[:32], val_rank)
            f1 = f1_score(y_test_set[:32], val_rank > 0.5)
            # f1 = f1_score(y_test_set, val_rank > 0.95)
            # auc = roc_auc_score(y_test_set, val_pred[ch])
            oof_aucs[ch].append(auc)
            f1s[ch].append(f1)
            precision = precision_score(y_test_set[:32], val_rank > 0.5)
            # precision = precision_score(y_test_set, val_rank > 0.95)
            precisions[ch].append(precision)
            print_log(f'#### Fold {fold+1}, {cur_num_channels} channels OOF AUC without TTA = {auc:.4f}, AUC rank = {auc_rank:.4f}, F1 = {f1:.4f}, precision = {precision:.4f}', [main_logger, res_logger])
            print_log(classification_report(y_test_set[:32], val_rank > 0.5, target_names = ['benign', 'malignant']), [main_logger, res_logger])
            # print_log(classification_report(y_test_set, val_rank > 0.95, target_names = ['benign', 'malignant']), [main_logger, res_logger])
            # oof_aucs[ch].append(auc)
            # print_log(f'#### Fold {fold+1}, {cur_num_channels} channels OOF AUC without TTA = {auc:.4f}', [main_logger, res_logger])

print_log('\n'+'-'*80, [main_logger, res_logger])
for ch in range(3, params['N_CHANNELS'] + 1):
    # get average predicted target between folds for the selected number of channels
    oof_preds[ch]= np.array(oof_pred[ch]).mean(axis=0)
    auc = roc_auc_score(y_test_set, oof_preds[ch])
    print_log(f'\nOverall OOF AUC {ch} channels without TTA = {auc:.4f}', [main_logger, res_logger])
    
    oof_preds_rank[ch]= np.array(oof_pred_rank[ch]).mean(axis=0)
    auc = roc_auc_score(y_test_set, oof_preds_rank[ch])
    precision = precision_score(y_test_set, val_rank > 0.5)
    f1 = f1_score(y_test_set, val_rank > 0.5)
    # precision = precision_score(y_test_set, val_rank > 0.95)
    # f1 = f1_score(y_test_set, val_rank > 0.95)
    print_log(f'OOF AUC {cur_num_channels} channels without TTA ranked = {auc:.4f}, F1 = {f1:.4f}, precision = {precision:.4f}', [main_logger, res_logger])
    print_log(classification_report(y_test_set, val_rank > 0.5, target_names = ['benign', 'malignant']), [main_logger, res_logger])
    # print_log(classification_report(y_test_set, val_rank > 0.95, target_names = ['benign', 'malignant']), [main_logger, res_logger])

    # compute weight based on AUC on validation set for each fold 
    # will use to give more weight to better folds for predicted target 
    scaler = (np.array(oof_aucs[ch]).max() - np.array(oof_aucs[ch]).min()+1) 
    weights[ch]= (np.array(oof_aucs[ch]) - np.array(oof_aucs[ch]).min()/2)/ scaler
    
    # weighted average between folds/models
    oof_preds_w[ch] = np.squeeze(np.array(oof_pred[ch])).T.dot(weights[ch])
    auc_w = roc_auc_score(y_test_set, oof_preds_w[ch])
    print_log(f'Overall OOF AUC weighted with {ch} channels = {auc_w:.4f}', [main_logger, res_logger])
    
    oof_preds_rank_w[ch] = np.squeeze(np.array(oof_pred_rank[ch])).T.dot(weights[ch])
    auc_w = roc_auc_score(y_test_set, oof_preds_rank_w[ch])
    print_log(f'Overall OOF AUC weighted with {ch} channels ranked = {auc_w:.4f}', [main_logger, res_logger])
    
    target = preds[ch].dot(weights[ch].T)
    submission = pd.DataFrame(dict(image_name = df_test.image_name, target = target))
    submission = submission.sort_values('image_name') 
    submission.to_csv(f"{n_folds}_{nets_names[params['EFF_NETS'][0]]}_{params['IMG_SIZES'][0]}_{ch}ch_w_{params['MAL_UPSAMPLE']}ups_{params['COLOR_SPACE']}_CV{'0'+str(auc_w)[2:6]}.csv", index=False)
    
    target_ranked = preds_rank[ch].dot(weights[ch].T)
    submission = pd.DataFrame(dict(image_name = df_test.image_name, target = target_ranked))
    submission = submission.sort_values('image_name') 
    submission.to_csv(f"{n_folds}_{nets_names[params['EFF_NETS'][0]]}_{params['IMG_SIZES'][0]}_{ch}ch_w_rank_{params['MAL_UPSAMPLE']}ups_{params['COLOR_SPACE']}_CV{'0'+str(auc_w)[2:6]}.csv", index=False)
    
    pv = [1,5,25,50,75,95,99]
    pv_4 = [25,50,75,95]
    # mals_only_5 = np.zeros(5)
    mals = pd.DataFrame(y_test_set)
    print_log("Test positive preds confidence:", [main_logger, res_logger])
    for i in range(5):
        mals['preds'] = oof_pred[ch][i]
        mals_only = mals.loc[mals.target==1]
        percs = np.percentile(mals_only.preds, pv_4)
        print(percs)
        # mals_only_5[i] = percs[1]
        t = str(  [f'{p: 2.0f}%: {w:.3f}' for (p, w) in zip(pv_4, percs)])
        print_log(t, [main_logger, res_logger])

    t_perc = np.percentile(target, pv)
    t = "Target all preds confidence:" + str([f'{p: 2.0f}%: {w:.3f}' for (p, w) in zip(pv, t_perc)])
    print_log(t, [main_logger, res_logger])
    t = str( [f'{w :.2f}' for w in weights[ch]])
    print_log('Weights: ' + t, [main_logger, res_logger])
    t = str([f'{w :.4f}' for w in  oof_aucs[ch]])
    print_log('AUCs: ' + t, [main_logger, res_logger])

print_log(f'\nParameters:\n{params}', [main_logger, res_logger])

for logger in loggers:
    for handler in list(loggers[logger].handlers):
        print(handler)
        handler.close()
        loggers[logger].removeHandler(handler) 

with open(f"{n_folds}_{nets_names[params['EFF_NETS'][0]]}_{params['IMG_SIZES'][0]}_{ch}ch_w_{params['MAL_UPSAMPLE']}ups_{params['COLOR_SPACE']}_CV{'0'+str(auc_w)[2:6]}.params", 'wb') as f: 
    pickle.dump(params, f)           
    

   
def print_percs(df_name):
    df = pd.read_csv(df_name)
    pv = [1,5,25,50,75,95,99,99.5, 99.9]
    percs = np.percentile(df.target, pv)
    t = str(  [f'{p: 2.1f}%: {w:.3f}' for (p, w) in zip(pv, percs)])
    print(t)

# print_percs('Y:\Kaggle\SIIM-ISIC Melanoma\Previous comp\B0_B1_256_TTA11_20-19-18_mal_0940_09231.csv')

thersholds = [.25, .40, .45, .50, .55, .60, .75, ]
# thersholds = [.05, .15, .25, .40, .5, .60, .75, .85, .95]
for ch in [3, 4, 5]:
    print('-' * 80)
    print(f'{ch} channels:')
    for th in thersholds:
        # t_precision = precision_score(y_test_set, oof_preds[3] >= th)
        # print(t_precision)
        print(f'Threshold {th:.2f}')
        print(classification_report(y_test_set, oof_preds[ch] >= th, target_names = ['benign', 'malignant']))
        print('-' * 20)
    


