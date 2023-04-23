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
import keras
import tensorflow as tf
from tensorflow.keras.layers import *
from tensorflow.keras.callbacks import *
import tensorflow.keras.backend as K

from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras import mixed_precision

n_folds = 5
DEVICE = "GPU"
# USE DIFFERENT SEED FOR DIFFERENT STRATIFIED KFOLD
SEED = 1970

# COLOR_SPACE = 'HSV' # possible values are 'RGB', 'HSV' and 'YCrCb'

EFNS = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, 
        EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
        EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2]
nets_names = ['v1b0', 'v1b1', 'v1b2', 'v1b3', 'v1b4', 'v1b5', 'v1b6', 'v1b7',
              'v2b0', 'v2b1', 'v2b2']

params = {
    # 'IMG_SIZES' :   [512] * n_folds,
    'IMG_SIZES' :   [256] * n_folds,
    # 'IMG_SIZES' :   [128] * n_folds,
    # 'EFF_NETS' :    [1] * n_folds,
    'EFF_NETS' :    [8] * n_folds,
    'BATCH_SIZES' : [64] * n_folds,
    # 'BATCH_SIZES' : [128] * n_folds,
    'EPOCHS' : [1] * n_folds,
    # number of times to duplicate the malignant samples, to use with augmentation to balance the classes
    'MAL_UPSAMPLE' : 5, 
    # test time augmentation steps
    'TTA' : 0,
    # possible values are 'RGB', 'HSV' and 'YCrCb'
    'COLOR_SPACE' : 'HSV' ,
    'DROP_LUM_CH' : True, # if we should drop Luminosity channel in HSV of YCrCb spaces
    # 'TO_HSV' : True,     # if we are using HSV color space instead of RGB
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
    global loggers
    
    if loggers.get(name):
        return loggers.get(name)
    else:
        logger = logging.getLogger(name)
        logger.setLevel(logging.INFO)
        # handler = logging.FileHandler(model_logs + datetime.now().strftime("%Y%m%d-%H%M%S") +'-' + 'basic.log')
        handler = logging.FileHandler(f"{model_logs}{datetime.now().strftime('%Y%m%d')}-{nets_names[params['EFF_NETS'][0]]}_img{str(params['IMG_SIZES'][0])}-{name}.log")
        # handler = logging.FileHandler(model_logs + datetime.now().strftime("%Y%m%d") + '-m' + str(params['EFF_NETS'][0]) + '_img' + str(params['IMG_SIZES'][0])  +'-' + f'{name}.log')
        # handler = logging.FileHandler(model_logs + datetime.now().strftime("%Y%m%d") + '-m' + str(params['EFF_NETS'][0]) + '_img' + str(params['IMG_SIZES'][0])  +'-' + 'basic.log')
        if t_format:
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        else:
            formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        loggers[name] = logger                       
        return logger


def print_log(out, loggers):
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
train_image_folder_path = "./jpeg/train/1024/"
test_image_folder_path = "./jpeg/test/1024/"
jpegs_mal_new = './jpeg_extra_mal/'

policy = mixed_precision.Policy('mixed_float16')
mixed_precision.set_global_policy(policy)

main_logger = myLogger('main')
res_logger = myLogger('results', False)

print_log('*' * 80, [main_logger])
print_log('Begin logging', [main_logger])
print_log('tensorflow version:' + str(tf.__version__), [main_logger])

print_log('Compute dtype: %s' % policy.compute_dtype, [main_logger])
print_log('Variable dtype: %s' % policy.variable_dtype, [main_logger])

if DEVICE != "TPU":
    print_log("Using default strategy for CPU and single GPU", [main_logger])
    strategy = tf.distribute.get_strategy()
    # strategy = tf.distribute.MirroredStrategy()
    # REPLICAS_m = mirrored_strategy.num_replicas_in_sync

   
AUTO     = tf.data.experimental.AUTOTUNE
REPLICAS = strategy.num_replicas_in_sync
print_log(f'Number of Replicas Used: {REPLICAS}', [main_logger])

gpu_devices = tf.config.experimental.list_physical_devices('GPU')
if gpu_devices:
    for gpu_device in gpu_devices:
        print_log('device available:' + str(gpu_device), [main_logger])

pd.set_option('display.max_columns', None)


df_train = pd.read_csv('train.csv')
df_test = pd.read_csv('test.csv')
cols = df_train.columns.to_list()

df_mal_2019 = pd.read_csv('train_malig_3_2019.csv')
df_mal_new = pd.read_csv('train_malig_2_new.csv')

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

print_log(f"Color space: {params['COLOR_SPACE']}; 4th channel: {params['4TH_CHANNEL']}; Drop luminosity channel: {params['DROP_LUM_CH']}", [main_logger])

def Focal_Loss(y_true, y_pred, alpha=0.25, gamma=2, weight=5):
    y_true = K.flatten(tf.cast(y_true, tf.float32))
    # y_pred = K.flatten(y_pred)
    y_pred = K.flatten(tf.cast(y_pred, tf.float32))

    BCE = K.binary_crossentropy(y_true, y_pred)
    BCE_EXP = K.exp(-BCE)
    alpha = alpha*y_true+(1-alpha)*(1-y_true)
    focal_loss = K.mean(alpha * K.pow((1-BCE_EXP), gamma) * BCE)

    return BCE+weight*focal_loss

def build_model_1(dim = 128, n_ch = 3, ef = 0, dropout = False):
    inp = tf.keras.layers.Input(shape = (dim, dim, n_ch), name = 'Image')
    base = EFNS[ef](input_shape = (dim, dim, n_ch), weights=None, include_top = False)
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout:
        x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    if dropout:
        x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name = 'Out')(x)
    
    model = tf.keras.Model(inputs = (inp), outputs = out)
    
    # opt = tf.keras.optimizers.Adam(learning_rate=0.00005)
    opt = tf.keras.optimizers.Adam(learning_rate=0.001)
    # opt = tf.train.experimental.enable_mixed_precision_graph_rewrite(opt)
    # loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
    loss = Focal_Loss
    model.compile(optimizer=opt, loss=loss, metrics=['AUC'])
    return model

def build_model(dim = 128, n_ch = 3, ef = 0, dropout = False):
    inp = tf.keras.layers.Input(shape = (dim, dim, n_ch), name = 'Image')
    base = EFNS[ef](input_shape = (dim, dim, n_ch), weights=None, include_top = False)
    x = base(inp)
    x = tf.keras.layers.GlobalAveragePooling2D()(x)
    if dropout:
        x = tf.keras.layers.Dropout(0.1)(x)
    x = tf.keras.layers.Dense(64, activation='relu')(x)
    if dropout:
        x = tf.keras.layers.Dropout(0.3)(x)
    out = tf.keras.layers.Dense(1, activation='sigmoid', name = 'Out')(x)
    
    model = tf.keras.Model(inputs = (inp), outputs = out)
    
    opt = tf.keras.optimizers.Adam(learning_rate=0.00005)
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
    model.compile(optimizer=opt, loss=loss, metrics=['AUC'])
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
   
# pass an image in BGR format as an input
def BGR2MI(img):
    MI = 100 * np.log10(1/(img[:,:,2].astype('float')+1))
    MI_norm = (MI * 0.0042 + 1) * 255
    MI_s = MI + abs(MI.min())
    MI_rescaled = 255 * (MI_s / abs(MI_s.max()))
    return MI.astype('int16'), MI_rescaled.astype('int16'), MI_norm.astype('int16')

# pass an image in BGR format as an input
def BGR2EI(img):
    EI = 100 * (np.log10(1/(img[:,:,1].astype('float')+1)) - 1.44 * np.log10(1/(img[:,:,2].astype('float')+1)))
    EI_norm = (EI * 0.0017 + 0.4098) * 255
    EI = EI + abs(EI.min())
    EI_rescaled = 255 * (EI / abs(EI.max()))
    return EI.astype('int16'), EI_rescaled.astype('int16'), EI_norm.astype('int16')   

#---------------------------------------------------------------------------------------------
def _read(path, desired_size, color_space = 'RGB', toAugment = False, drop_luminosity = False):
# def _read(path, desired_size, to_HSV = False, toAugment = False):
    """Will be used in DataGenerator"""
    img_BGR = cv2.imread(path)
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
        EI, EI_rescaled, EI_norm = BGR2EI(img_BGR)      
        EI_exp = np.expand_dims(EI_norm, axis = 2)
        MI, MI_rescaled, MI_norm = BGR2MI(img_BGR)
        MI_exp = np.expand_dims(MI_norm, axis = 2)
        if params['4TH_CHANNEL'] == 'EI':
            img_exp = np.concatenate((img_exp, EI_exp), axis = 2)        
        else:
            img_exp = np.concatenate((img_exp, MI_exp), axis = 2)        
    if n_ch > 4:
        # reverse which channel to add as 5th, since we've alreay added the 4th one
        if params['4TH_CHANNEL'] == 'EI':
            img_exp = np.concatenate((img_exp, MI_exp), axis = 2)        
        else:
            img_exp = np.concatenate((img_exp, EI_exp), axis = 2)        

    return img_exp 
   
                 
#---------------------------------------------------------------------------------------------
class DataGenerator(keras.utils.Sequence):

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
        # to_HSV = self.to_HSV
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


csv_logger = CSVLogger(model_logs + datetime.now().strftime("%Y%m%d-%H%M%S") + '-' + 'effnet_training_log.csv', separator = ',', append = True)

logdir = "logs_tb/scalars/" + datetime.now().strftime("%Y%m%d-%H%M%S")
file_writer = tf.summary.create_file_writer(logdir + "/metrics")
file_writer.set_as_default()
tensorboard_callback = keras.callbacks.TensorBoard(log_dir=logdir)

train_list_IDs = train_set.image_name
valid_list_IDs = test_set.image_name
test_list_IDs = df_test.image_name

VERBOSE = 1

skf = KFold(n_splits = n_folds, shuffle = True, random_state = SEED)
oof_preds = {}
oof_preds_w = {}
oof_pred = {3:[], 4:[], 5:[]}
preds = {3: np.zeros((len(test_list_IDs), n_folds)), 4: np.zeros((len(test_list_IDs), n_folds)), 5: np.zeros((len(test_list_IDs), n_folds))}
models = {3:[], 4:[], 5:[]}
oof_aucs = {3:[], 4:[], 5:[]}
val_pred = {3:[], 4:[], 5:[]}
weights = {3:[], 4:[], 5:[]}

drop_luminosity_ch = params['DROP_LUM_CH']
cur_color_space = params['COLOR_SPACE']

for fold, (idxT, idxV) in enumerate(skf.split(train_IP_set)):
    cur_batch_size = params['BATCH_SIZES'][fold]

    print_log('-' * 80, [main_logger, res_logger])
    print_log(f'Fold #: {fold+1}, Model: {EFNS[params["EFF_NETS"][fold]].__name__}, BS: {params["BATCH_SIZES"][fold]}, Image Size: {params["IMG_SIZES"][fold]}', [main_logger, res_logger])
    train_IPs = train_IP_set[idxT]
    val_IPs = train_IP_set[idxV]

    train_image_names = train_set.loc[train_set.patient_id.isin(train_IPs), 'image_name']
    val_image_names = train_set.loc[train_set.patient_id.isin(val_IPs), 'image_name']
    train_I = train_image_names.index.to_numpy()
    val_I = val_image_names.index.to_numpy()
        
    for ch in range (3, params['N_CHANNELS'] + 1): # running the same model with inputs of just RGB, RGB+MI, RGB+MI+EI for each fold
        print_log('-'*25, [main_logger])
        # print_log(f'Number of channels: {ch}', [main_logger])
        # models[ch] = build_model(dim = params['IMG_SIZES'][fold], n_ch = ch, ef = params['EFF_NETS'][fold], dropout = False)
        
        if (cur_color_space != 'RGB') & (drop_luminosity_ch):
            cur_num_channels = ch-1
        else:
            cur_num_channels = ch
            
        print_log(f'Number of channels: {cur_num_channels}', [main_logger])
        models[ch] = build_model(dim = params['IMG_SIZES'][fold], n_ch = cur_num_channels, ef = params['EFF_NETS'][fold], dropout = False)
        cur_img_size = (params['IMG_SIZES'][fold], params['IMG_SIZES'][fold], cur_num_channels)
        print_log(f'Image Size: {cur_img_size}', [main_logger])
        
        valid_gen = DataGenerator(valid_list_IDs, labels = None, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)
        test_gen = DataGenerator(test_list_IDs, labels = None, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = test_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)       
        # valid_gen = DataGenerator(valid_list_IDs, labels = None, batch_size = params['BATCH_SIZES'][fold], img_size = (params['IMG_SIZES'][fold], params['IMG_SIZES'][fold], ch), img_dir = train_image_folder_path, to_HSV = params['TO_HSV'], testAugment = False)
        # test_gen = DataGenerator(test_list_IDs, labels = None, batch_size = params['BATCH_SIZES'][fold], img_size = (params['IMG_SIZES'][fold], params['IMG_SIZES'][fold], ch), img_dir = test_image_folder_path, to_HSV = params['TO_HSV'], testAugment = False)       
        # valid_gen = DataGenerator(valid_list_IDs, labels = None, batch_size = params['BATCH_SIZES'][fold], img_size = (params['IMG_SIZES'][fold], params['IMG_SIZES'][fold], ch), img_dir = train_image_folder_path, testAugment = False)
        # test_gen = DataGenerator(test_list_IDs, labels = None, batch_size = params['BATCH_SIZES'][fold], img_size = (params['IMG_SIZES'][fold], params['IMG_SIZES'][fold], ch), img_dir = test_image_folder_path, testAugment = False)       
        
        y_train_set = train_set.loc[train_set.patient_id.isin(train_IPs), 'target']
        y_val_set = train_set.loc[train_set.patient_id.isin(val_IPs), 'target']
        mals_t = np.array(y_train_set.loc[y_train_set==1].index).astype('int32')
        
        idxT_ext = train_I.copy()
        for m in range(params['MAL_UPSAMPLE']):
            idxT_ext = np.concatenate((idxT_ext, mals_t))   
    
        idxT_ext = shuffle(idxT_ext, random_state = SEED)
        trn_ext = train_image_names.loc[idxT_ext].reset_index(drop = True)
        y_trn_ext = y_train_set.loc[idxT_ext].reset_index(drop = True)
        
        trn_gen = DataGenerator(trn_ext, labels = y_trn_ext, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)
        val_gen = DataGenerator(val_image_names, labels = y_val_set, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)
        # trn_gen = DataGenerator(trn_ext, labels = y_trn_ext, batch_size = params['BATCH_SIZES'][fold], img_size = (params['IMG_SIZES'][fold], params['IMG_SIZES'][fold], ch), img_dir = train_image_folder_path, to_HSV = params['TO_HSV'], testAugment = False)
        # val_gen = DataGenerator(val_image_names, labels = y_val_set, batch_size = params['BATCH_SIZES'][fold], img_size = (params['IMG_SIZES'][fold], params['IMG_SIZES'][fold], ch), img_dir = train_image_folder_path, to_HSV = params['TO_HSV'], testAugment = False)
        # trn_gen = DataGenerator(trn_ext, labels = y_trn_ext, batch_size = params['BATCH_SIZES'][fold], img_size = (params['IMG_SIZES'][fold], params['IMG_SIZES'][fold], ch), img_dir = train_image_folder_path, testAugment = False)
        # val_gen = DataGenerator(val_image_names, labels = y_val_set, batch_size = params['BATCH_SIZES'][fold], img_size = (params['IMG_SIZES'][fold], params['IMG_SIZES'][fold], ch), img_dir = train_image_folder_path, testAugment = False)
        # print_log('#'*25)
    
        # SAVE BEST MODEL EACH FOLD
        sv = tf.keras.callbacks.ModelCheckpoint(
            f'fold_{fold}-{cur_num_channels}.h5', monitor='val_loss', verbose=1, save_best_only=True,
            save_weights_only=True, mode='min', save_freq='epoch')    
    
        history = models[ch].fit(
            trn_gen, 
            epochs = params['EPOCHS'][fold], 
            callbacks = [sv, csv_logger, tensorboard_callback, get_lr_callback(cur_batch_size)], 
            validation_data = val_gen,
            verbose = VERBOSE
        )
        print_log(f'Loading best model for fold {fold+1} with {cur_num_channels} channels', [main_logger])
        models[ch].load_weights(f'fold_{fold}-{cur_num_channels}.h5') 
    
        # PREDICT TEST USING TTA
        print_log('Predicting Test without TTA...', [main_logger])
        pred = models[ch].predict(test_gen, verbose = 1)[:len(test_list_IDs)]  
        preds[ch][:, fold] += pred[:,0] 
    
        # PREDICT OOF USING TTA
        print_log('Predicting OOF without TTA...', [main_logger])
        val_pred[ch] = models[ch].predict(valid_gen, verbose = 1)[:len(valid_list_IDs)]
        oof_pred[ch].append(val_pred[ch])   
        
        # REPORT RESULTS
        auc = roc_auc_score(y_test_set, val_pred[ch])
        oof_aucs[ch].append(auc)
        print_log(f'#### Fold {fold+1}, {cur_num_channels} channels OOF AUC without TTA = {auc:.4f}', [main_logger, res_logger])

print_log('\n'+'-'*80)
for ch in range(3, params['N_CHANNELS'] + 1):
    # get average predicted target between folds for the selected number of channels
    oof_preds[ch]= np.array(oof_pred[ch]).mean(axis=0)
    auc = roc_auc_score(y_test_set, oof_preds[ch])
    print_log(f'\nOverall OOF AUC {ch} channels without TTA = {auc:.4f}', [main_logger, res_logger])
    
    # compute weight based on AUC on validation set for each fold 
    # will use to give more weight to better folds for predicted target 
    scaler = (np.array(oof_aucs[ch]).max() - np.array(oof_aucs[ch]).min()+1) 
    weights[ch]= (np.array(oof_aucs[ch]) - np.array(oof_aucs[ch]).min()/2)/ scaler
    
    # weighted average between folds/models
    oof_preds_w[ch] = np.squeeze(np.array(oof_pred[ch])).T.dot(weights[ch])
    auc_w = roc_auc_score(y_test_set, oof_preds_w[ch])
    print_log(f'Overall OOF AUC weighted with {ch} channels = {auc_w:.4f}', [main_logger, res_logger])
    
    target = preds[ch].dot(weights[ch].T)
    submission = pd.DataFrame(dict(image_name = df_test.image_name, target = target))
    submission = submission.sort_values('image_name') 
    # submission.to_csv(f"5_v2b0_{params['IMG_SIZES'][0]}_{ch}ch_w_{params['MAL_UPSAMPLE']}ups_CV{'0'+str(auc_w)[2:6]}.csv", index=False)
    submission.to_csv(f"{n_folds}_{nets_names[params['EFF_NETS'][0]]}_{params['IMG_SIZES'][0]}_{ch}ch_w_{params['MAL_UPSAMPLE']}ups_{params['COLOR_SPACE']}_CV{'0'+str(auc_w)[2:6]}.csv", index=False)
    
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
    # for handler in list(main_logger.handlers):
    for handler in list(loggers[logger].handlers):
        print(handler)
        handler.close()
        # main_logger.removeHandler(handler) 
        loggers[logger].removeHandler(handler) 
            