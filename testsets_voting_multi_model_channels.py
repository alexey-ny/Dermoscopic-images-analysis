# -*- coding: utf-8 -*-
"""
Created on Mon Nov 20 22:34:59 2023

@author: alex
"""

import os


import random
import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import cv2
import numpy as np
import pandas as pd 
from sklearn.metrics import roc_auc_score, auc #, roc_curve, precision_recall_curve
from sklearn.metrics import average_precision_score, confusion_matrix
# , f1_score, precision_score, classification_report
from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve
from sklearn.preprocessing import MinMaxScaler

# import matplotlib.pyplot as plt
# import matplotlib.colors as mcolors

import tensorflow as tf
# from tensorflow.keras.layers import Input, GlobalAveragePooling2D, Dropout, Dense
# from tensorflow.keras.callbacks import TensorBoard, CSVLogger, ModelCheckpoint, EarlyStopping
# import tensorflow.keras.backend as K
# from keras.metrics import PrecisionAtRecall

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
# from tensorflow.keras import mixed_precision
# from tqdm import tqdm
# from numba import njit

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


 
test_list_IDs = df_test.isic_id
test_no_lesion_id_list_IDs = df_test_no_lesion_id.isic_id

VERBOSE = 1

oof_preds = {}
oof_preds_TTA = {}

oof_preds_w = {}
oof_preds_w_TTA = {}

for c_model in models_ISIC[:-1]:
    print_log('=' * 80, [main_logger, res_logger])
    print_log(f'Inference for {c_model}', [main_logger, res_logger])
    model_path = ISIC_models_path + c_model + '/'
    all_model_files = os.listdir(model_path)
    tests_files = [f for f in all_model_files if ('CV' in f) & ('.testobj' in f)]
    
    if len(tests_files) < 2:
        print_log('Wrong number of saved test sets objects for the model, skipping!', [main_logger, res_logger])
    else:       
        test_loc_name = tests_files[0]
        test_NoL_name = tests_files[1]
    
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

