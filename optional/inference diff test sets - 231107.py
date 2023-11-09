# -*- coding: utf-8 -*-
"""
Created on Tue Jun 20 11:33:17 2023

@author: alex
"""# -*- coding: utf-8 -*-
from tensorflow.keras.applications.resnet_v2 import ResNet50V2, ResNet101V2, ResNet152V2
from tensorflow.keras.applications.efficientnet_v2 import EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2
from tensorflow.keras.applications.efficientnet import EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7
from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
import tensorflow as tf
import tensorflow.keras.backend as K

import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import cv2
import numpy as np
import pandas as pd 
from imgaug import augmenters as iaa
import math
import logging
from sklearn.metrics import roc_auc_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, precision_recall_fscore_support
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import os
from tqdm import tqdm
import logging

import pickle
from numba import njit
import simplejpeg 

from logging_utils import *
# from logging_utils import loggers
from logging_utils import myLogger, print_log, close_loggers 

from sklearn.metrics import confusion_matrix

n_folds = 5
DEVICE = "GPU"
MULTI = False

SEED = 1970

EFNS = [EfficientNetB0, EfficientNetB1, EfficientNetB2, EfficientNetB3, 
        EfficientNetB4, EfficientNetB5, EfficientNetB6, EfficientNetB7,
        EfficientNetV2B0, EfficientNetV2B1, EfficientNetV2B2,
        ResNet50V2, ResNet101V2, ResNet152V2]
nets_names = ['v1b0', 'v1b1', 'v1b2', 'v1b3', 'v1b4', 'v1b5', 'v1b6', 'v1b7',
              'v2b0', 'v2b1', 'v2b2', 'resnet50v2', 'resnet101v2', 'resnet152v2']

models = './models/'
model_logs = './logs_model/'

train_image_folder_path = "./jpeg/train/1024/"
# test_image_folder_path = "./jpeg/test/1024/"
# test_image_folder_path = "./jpeg/test_DDI/"
# test_image_folder_path = "./jpeg/test_PROVe-AI/1024/"
# test_image_folder_path = "./jpeg/test_SMKCC_2020/1024/"

tests = [{'path' : "./jpeg/test_SMKCC_2020/1024/", 'metadata' : 'consecutive-biopsies-for-melanoma-across-year-2020_metadata_2023-11-03.csv'}, 
         {'path' : "./jpeg/test_PROVe-AI/1024/", 'metadata' : 'prove-ai_metadata_2023-11-02.csv'},
         {'path' : "./jpeg/test_HIBA/1024/", 'metadata' : 'hiba-skin-lesions_metadata_2023-11-07.csv'}]

data_path = './data/'

df_train = pd.read_csv(data_path + 'train.csv')
train_list_IDs = list(df_train.image_name)
df_2019 = pd.read_csv(data_path + 'train_malig_3_2019.csv')
train2019_list_IDs = list(df_2019.image_name)
df_test_2020 = pd.read_csv(data_path + 'test.csv')
test2020_list_IDs = list(df_test_2020.image_name)

# metadata_name = 'consecutive-biopsies-for-melanoma-across-year-2020_metadata_2023-11-03.csv'
# df_test = pd.read_csv(data_path + 'prove-ai_metadata_2023-11-02.csv')
# df_test = pd.read_csv(data_path + 'hiba-skin-lesions_metadata_2023-11-07.csv')

# df_test = pd.read_csv(data_path + metadata_name)
# df_mal = df_test.loc[df_test.benign_malignant == 'malignant']
# df_mal.diagnosis.value_counts()

# test_list_IDs = list(df_test.DDI_file)
# test_list_IDs = list(df_test.isic_id)

# intersection1 = set(test_list_IDs) & set(train_list_IDs)
# intersection2 = set(test_list_IDs) & set(train2019_list_IDs)
# intersection3 = set(test_list_IDs) & set(test2020_list_IDs)

# assert ((len(intersection1) == 0) and (len(intersection2) == 0) and (len(intersection3) == 0)), 'Duplicates in train/tets sets'

# df_test.diagnosis.value_counts()
# df_test.benign_malignant.value_counts()
# dt = df_test.loc[df_test.benign_malignant == 'malignant']
# df_test['target'] = df_test.diagnosis == 'melanoma'

# Image Augmentation
sometimes = lambda aug: iaa.Sometimes(0.35, aug)
augmentation = iaa.Sequential([  
                                iaa.Fliplr(0.5),
                                iaa.Flipud(0.5),
                                iaa.LinearContrast((0.9, 1.1)),
                                iaa.Multiply((0.9, 1.1), per_channel=0.2),
                                sometimes(iaa.Cutout(nb_iterations=(1, 3), size=0.05, squared=False, fill_mode="constant", cval=0)),
                                sometimes(iaa.Crop(px=(20, 80), keep_size = True, sample_independently = False)),
                                sometimes(iaa.Affine(rotate=(-35, 35))),
                                sometimes(iaa.Affine(scale=(0.95, 1.05)))
                            ], random_order = True)       

main_logger = myLogger('main', f"{model_logs}{datetime.now().strftime('%Y%m%d')}-single-model-multiple-tests.log")
# main_logger = myLogger('main', f"{model_logs}{datetime.now().strftime('%Y%m%d')}-{nets_names[params['EFF_NETS'][0]]}_img{str(params['IMG_SIZES'][0])}-main.log")

print_log('*' * 80, [main_logger])
print_log('Begin logging', [main_logger])

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
def _read(path, desired_size, color_space = 'RGB', toAugment = False, drop_luminosity = False, eff_version = 2):
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
        else:
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
        # if params['4TH_CHANNEL'] == 'EI':
        if channel_4 == 'EI':
            EI_norm = BGR2EI(img_BGR)      
            EI_exp = np.expand_dims(EI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, EI_exp), axis = 2)        
        else:
            MI_norm = BGR2MI(img_BGR)
            MI_exp = np.expand_dims(MI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, MI_exp), axis = 2)        
    if n_ch > 4:
        # reverse which channel to add as 5th, since we've alreay added the 4th one
        if channel_4 == 'EI':
        # if params['4TH_CHANNEL'] == 'EI':
            MI_norm = BGR2MI(img_BGR)
            MI_exp = np.expand_dims(MI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, MI_exp), axis = 2)        
        else:
            EI_norm = BGR2EI(img_BGR)      
            EI_exp = np.expand_dims(EI_norm, axis = 2)
            img_exp = np.concatenate((img_exp, EI_exp), axis = 2)        

    if eff_version == 1:
        return (img_exp / 255) # rescale to the range of [0, 1]
    elif eff_version == 2:
        return img_exp   # do not rescale for EfficientNet v2 architecture
                 
#---------------------------------------------------------------------------------------------
class DataGenerator(tf.keras.utils.Sequence):

    def __init__(self, list_IDs, labels = None, batch_size = 1, img_size = (512, 512, 1), 
                 img_dir = train_image_folder_path, color_space = 'RGB', 
                 testAugment = False, drop_luminosity = False, eff_version = 1,
                 *args, **kwargs):

        self.list_IDs = list_IDs
        self.labels = labels
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_dir = img_dir
        self.testAugment = testAugment
        self.color_space = color_space
        self.drop_luminosity = drop_luminosity
        self.eff_version = eff_version
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
                X[i,] = _read(self.img_dir+ID+".jpg", self.img_size, self.color_space, toAugment = True, drop_luminosity = self.drop_luminosity, eff_version = self.eff_version)
                Y[i,] = self.labels.loc[idxs[i]]       
            return X, Y

        elif self.testAugment: # test phase with Augmentation
            for i, ID in enumerate(list_IDs_temp):
                # X[i,] = _read(self.img_dir + ID, self.img_size, self.color_space, toAugment = True, drop_luminosity = self.drop_luminosity, eff_version = self.eff_version)
                X[i,] = _read(self.img_dir + ID+".jpg", self.img_size, self.color_space, toAugment = True, drop_luminosity = self.drop_luminosity, eff_version = self.eff_version)
            return X

        else: # test phase no Augmentation
            for i, ID in enumerate(list_IDs_temp):
                # X[i,] = _read(self.img_dir + ID, self.img_size, self.color_space, toAugment = False, drop_luminosity = self.drop_luminosity, eff_version = self.eff_version)
                X[i,] = _read(self.img_dir + ID+".jpg", self.img_size, self.color_space, toAugment = False, drop_luminosity = self.drop_luminosity, eff_version = self.eff_version)
            return X

   
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
    
    opt = tf.keras.optimizers.Adam(learning_rate = 0.00005)
    # loss = Focal_Loss
    loss = tf.keras.losses.BinaryCrossentropy(label_smoothing=0.05) 
    model.compile(optimizer=opt, loss=loss, metrics=['AUC'])
    return model


n_TTA = 0

# models_dirs = filter(os.path.isdir, os.listdir(models))
models_dirs = [ name for name in os.listdir(models) if os.path.isdir(os.path.join(models, name)) ]
# pcik just 1 best model for now
models_dirs = [models_dirs[11]]
# models_dirs = models_dirs[11 : 12]

n_models = len(models_dirs)
for j, t in enumerate(tests):
    metadata_name = t['metadata']   
    test_image_folder_path = t['path']
    print_log(f'Dataset - {test_image_folder_path}, metadata file - {metadata_name}', [main_logger])
    
    df_test = pd.read_csv(data_path + metadata_name)
    df_test['target'] = df_test.diagnosis == 'melanoma'
    y_test_set = df_test['target']
    print_log(y_test_set.value_counts(), [main_logger])
    valid_list_IDs = df_test.isic_id

    test_pred     = {3:[], 4:[], 5:[]}
    oof_pred     = {3:[], 4:[], 5:[]}
    oof_pred_TTA = {3:[], 4:[], 5:[]}
    oof_aucs     = {3:[], 4:[], 5:[]}
    oof_aucs_TTA = {3:[], 4:[], 5:[]}
    metrics      = {3:[], 4:[], 5:[]}
    metrics_TTA  = {3:[], 4:[], 5:[]}
    for i in range(n_models):
    # for i in tqdm(range(len(models_dirs[:2]))):
        m = models_dirs[i]
        print_log(f'{i+1}th model out of {n_models}: {m}', [main_logger])
        comps = m.split('_')
        if 'resnet' in comps[1]:
            effm_v = 2 # assume no need to rescale the images
        else:
            # effm_v = 1 # assume always need to rescale the images
            if 'RGB' in comps:
                effm_v = 2 # assume no need to rescale the images
            else:
                effm_v = int(comps[1][1])
            # effm_m = int(comps[1][3])
        eff_name = comps[1]
        img_size = int(comps[2])
        cur_model_dir = os.path.join(models, m)
        cur_files = os.listdir(cur_model_dir)
        param_file_name = [p for p in cur_files if 'params' in p]
        if len(param_file_name ) > 0:
            param_file_name = param_file_name[0]
            with open(f'{cur_model_dir}/{param_file_name}', 'rb') as f: 
                params = pickle.load(f)      
            eff_m = params['EFF_NETS'][0]
            cur_batch_size = params['BATCH_SIZES'][0]
            cur_color_space = params['COLOR_SPACE']
            drop_luminosity_ch = params['DROP_LUM_CH']
            channel_4 = params['4TH_CHANNEL']
        else:
            eff_m = nets_names.index(eff_name)
            cur_batch_size = 64
            if 'HSV' in comps:
                cur_color_space = 'HSV'
            else:
                cur_color_space = 'RGB'
            drop_luminosity_ch = False
            if 'EI' in comps:
                channel_4 = 'EI'
            else:
                channel_4 = 'MI'
    
        if ('EI' in comps) & ('MI' in comps):
            channels_list = ['3', 'EI', '5']
            # channels_list = ['3', 'EI', 'MI', '5']
        else:
            channels_list = ['3', '4', '5']
            
        print_log(f'  Image Size: {img_size}, BS: {cur_batch_size}, Rescaling flag: {effm_v} (1 = to rescale)', [main_logger])
        # print_log(f'  Image Size: {img_size}, BS: {cur_batch_size}', [main_logger])
        # print_log(f'  Image Size: {cur_img_size}, BS: {cur_batch_size}', [main_logger])
        # for fold in range(1):
        fold = 0
        for fold in range(5):
            print_log(f'  Fold: {fold + 1}', [main_logger])
            ch_n = 0
            ch_name  = '3'
            for ch_n, ch_name in enumerate(channels_list):
            # for ch in range(3, 6):
                ch = ch_n + 3
                if len(param_file_name ) > 0:
                    cur_img_size = (params['IMG_SIZES'][0], params['IMG_SIZES'][0], ch)
                else:
                    cur_img_size = (img_size, img_size, ch)
                                
                K.clear_session()
                model = build_model(dim = img_size, n_ch = ch, ef = eff_m, dropout = False)
                # model = build_model(dim = img_size, n_ch = ch, ef = eff_m, dropout = True)
                print('    Loading weights')
                # print_log(f'    Loading weights', [main_logger])
                model.load_weights(f'{cur_model_dir}/fold_{fold}-{ch_name}.h5') 
                # models[ch] = build_model(dim = img_size, n_ch = ch, ef = eff_m, dropout = True)
                valid_gen = DataGenerator(valid_list_IDs, labels = None, batch_size = cur_batch_size, 
                # valid_gen = DataGenerator(valid_list_IDs, labels = None, batch_size = cur_batch_size, 
                                          img_size = cur_img_size, img_dir = test_image_folder_path, 
                                          # img_size = cur_img_size, img_dir = train_image_folder_path, 
                                          color_space = cur_color_space, testAugment = False, 
                                          drop_luminosity = drop_luminosity_ch, eff_version = effm_v)
                # valid_gen = DataGenerator(valid_list_IDs, labels = None, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = False, drop_luminosity = drop_luminosity_ch)
    
                print_log('    Predicting Validation without TTA...', [main_logger])
                # print_log(f'    Predicting Validation without TTA...', [main_logger])
                val_pr = model.predict(valid_gen, verbose = 0)[:len(valid_list_IDs)]
                # val_pr = model.predict(valid_gen, verbose = 1)[:len(valid_list_IDs)]
                # val_pr = models[ch].predict(valid_gen, verbose = 1)[:len(valid_list_IDs)]
                oof_pred[ch].append(val_pr)   
                auc = roc_auc_score(y_test_set, val_pr)
                precision, recall, f1, _ = precision_recall_fscore_support(y_test_set, val_pr >= 0.5, average = None)
                oof_aucs[ch].append(auc)
                metrics[ch].append((auc, f1, precision, recall))
                print_log(f'    {ch} channels OOF AUC without TTA = {auc:.4f}, F1 = {f1[1]:.4f}, precision = {precision[1]:.4f}, recall = {recall[1]:.4f}', [main_logger])
                # print_log(f'    Fold {fold+1}, {ch} channels OOF AUC without TTA = {auc:.4f}, F1 = {f1[1]:.4f}, precision = {precision[1]:.4f}, recall = {recall[1]:.4f}', [main_logger])
    
                # print_log('    Predicting Test without TTA...', [main_logger])
                # # test_pr = model.predict(valid_gen, verbose = 1)[:len(test_list_IDs)]
                # # PREDICT TEST without USING TTA
                # test_gen = DataGenerator(test_list_IDs, labels = None, batch_size = cur_batch_size, img_size = cur_img_size, 
                #                          img_dir = test_image_folder_path, color_space = cur_color_space, testAugment = False, 
                #                          drop_luminosity = drop_luminosity_ch, sel_channels = ch)                      
    
                # test_pr = model.predict(test_gen, verbose = 1)[:len(test_list_IDs)]  
                # test_pred[ch].append(test_pr)   
                # # preds[ch][:, fold] += pred[:,0] 
        
                # PREDICT USING TTA
                if n_TTA > 0:
                    print(f'    Predicting Validation with {n_TTA} TTA...')
                    # print_log(f'    Predicting Validation with {n_TTA} TTA...', [main_logger])
                    valid_gen = DataGenerator(valid_list_IDs, labels = None, batch_size = cur_batch_size, img_size = cur_img_size, img_dir = train_image_folder_path, color_space = cur_color_space, testAugment = True, drop_luminosity = drop_luminosity_ch, eff_version = effm_v)
                    val_pr_TTA = model.predict(valid_gen, verbose = 0)[:len(valid_list_IDs)]
                    # val_pr_TTA = model.predict(valid_gen, verbose = 1)[:len(valid_list_IDs)]
                    for n in range(n_TTA - 1):
                    # for n in tqdm(range(n_TTA - 1)):
                        val_pr_TTA += model.predict(valid_gen, verbose = 0)[:len(valid_list_IDs)]
                    oof_pred_TTA[ch].append(val_pr_TTA / n_TTA)   
        
                    # REPORT RESULTS
                    auc = roc_auc_score(y_test_set, val_pr_TTA / n_TTA)
                    precision, recall, f1, _ = precision_recall_fscore_support(y_test_set, (val_pr_TTA / n_TTA) >= 0.5, average = None)
                    oof_aucs_TTA[ch].append(auc)
                    metrics_TTA[ch].append((auc, f1, precision, recall))
                    print_log(f'    {ch} channels OOF AUC with {n_TTA} TTA = {auc:.4f}, F1 = {f1[1]:.4f}, precision = {precision[1]:.4f}, recall = {recall[1]:.4f}', [main_logger])
                # print_log(f'    Fold {fold+1}, {ch} channels OOF AUC with {n_TTA} TTA = {auc:.4f}, F1 = {f1[1]:.4f}, precision = {precision[1]:.4f}, recall = {recall[1]:.4f}', [main_logger])
    
        
            
    print_log('Ensemble predictions', [main_logger])
    final_preds = []
    for ch_n, ch_name in enumerate(channels_list):
    # for ch in range(3, 6):
        ch = ch_n + 3
        val_pr_ch = np.array(oof_pred[ch]).squeeze(axis=2)
        val_pr_mean = val_pr_ch.mean(axis = 0)
        final_preds.append(val_pr_mean)
        
        auc_mean = roc_auc_score(y_test_set, val_pr_mean)
        precision, recall, f1, _ = precision_recall_fscore_support(y_test_set, val_pr_mean >= 0.5, average = None)
        print_log(f'    {ch} channels OOF AUC without TTA = {auc_mean:.4f}, F1 = {f1[1]:.4f}, precision = {precision[1]:.4f}, recall = {recall[1]:.4f}', [main_logger])

    sub = pd.DataFrame(dict(prediction = val_pr_mean, target = y_test_set))
    sub_mal = sub.loc[sub.target == 1]
    sub_ben = sub.loc[sub.target != 1]
    
    cm = confusion_matrix(sub.prediction >= 0.5, sub.target)
    print_log(cm, [main_logger])
    
    # sub10 = sub_mal.sample(30)
    sub10 = sub_mal.sample(15)
    # sub10 = sub_mal.sample(60)
    subtest = pd.concat([sub10, sub_ben])
    cm10 = confusion_matrix(subtest.prediction >= 0.5, subtest.target)
    
    auc_mean_10 = roc_auc_score(subtest.target, subtest.prediction)
    precision, recall, f1, _ = precision_recall_fscore_support(subtest.target, subtest.prediction >= 0.5, average = None)
    print_log(cm10, [main_logger])
    print_log(f'    {ch} channels OOF AUC without TTA = {auc_mean_10:.4f}, F1 = {f1[1]:.4f}, precision = {precision[1]:.4f}, recall = {recall[1]:.4f}', [main_logger])

# fp = pd.DataFrame(np.array(final_preds).T, columns = ['ch3', 'ch4', 'ch5'])
# fp['target'] = y_test_set

# fp.to_csv('Ensemble OOF predictions all channels PROVe-AI 1024.csv')
# fp.to_csv('Ensemble OOF predictions all channels PROVe-AI originals.csv')


close_loggers(loggers)
