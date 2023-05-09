# -*- coding: utf-8 -*-
"""
Created on Wed Feb  8 11:49:02 2023
@author: alex
"""

import warnings
warnings.filterwarnings('ignore')
from datetime import datetime
import numpy as np
import pandas as pd 
import math
import logging

from numba import njit

import simplejpeg 

import time
import psutil
import platform


DEVICE = "GPU"
# USE DIFFERENT SEED FOR DIFFERENT STRATIFIED KFOLD
SEED = 1970

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
        handler = logging.FileHandler(f"{model_logs}{datetime.now().strftime('%Y%m%d')}-EI_MI_compute-{name}.log")
        if t_format:
            formatter = logging.Formatter('%(asctime)s %(levelname)s %(message)s')
        else:
            formatter = logging.Formatter('%(message)s')
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        loggers[name] = logger                       
        return logger


def print_log(out, loggers=[]):
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
    
def get_size(bytes, suffix="B"):
    """
    Scale bytes to its proper format
    e.g:
        1253656 => '1.20MB'
        1253656678 => '1.17GB'
    """
    factor = 1024
    for unit in ["", "K", "M", "G", "T", "P"]:
        if bytes < factor:
            return f"{bytes:.2f}{unit}{suffix}"
        bytes /= factor
        
        
def sysinfo():
    print_log("-"*20 + "  System Information",[main_logger])
    # print_log("="*40, "System Information", "="*40)
    uname = platform.uname()
    print_log(f"System: {uname.system}",[main_logger])
    print_log(f"Node Name: {uname.node}",[main_logger])
    print_log(f"Release: {uname.release}",[main_logger])
    print_log(f"Version: {uname.version}",[main_logger])
    print_log(f"Machine: {uname.machine}",[main_logger])
    print_log(f"Processor: {uname.processor}",[main_logger])
    
    # let's print_log CPU information
    print_log("-"*20 + "  CPU Info" , [main_logger])
    # number of cores
    print_log(f"Physical cores: {psutil.cpu_count(logical=False)}")
    print_log(f"Total cores: {psutil.cpu_count(logical=True)}")
    # CPU frequencies
    cpufreq = psutil.cpu_freq()
    print_log(f"Max Frequency: {cpufreq.max:.2f}Mhz",[main_logger])
    print_log(f"Min Frequency: {cpufreq.min:.2f}Mhz",[main_logger])
    print_log(f"Current Frequency: {cpufreq.current:.2f}Mhz",[main_logger])
    
    # Memory Information
    print_log("-"*20 + "  Memory Information",[main_logger])
    # get the memory details
    svmem = psutil.virtual_memory()
    print_log(f"Total: {get_size(svmem.total)}",[main_logger])
    print_log(f"Available: {get_size(svmem.available)}",[main_logger])
    print_log(f"Used: {get_size(svmem.used)}",[main_logger])
    print_log(f"Percentage: {svmem.percent}%",[main_logger])
    print_log("-"*10 + "  SWAP" ,[main_logger])
    # get the swap memory details (if exists)
    swap = psutil.swap_memory()
    print_log(f"Total: {get_size(swap.total)}",[main_logger])
    print_log(f"Free: {get_size(swap.free)}",[main_logger])
    print_log(f"Used: {get_size(swap.used)}",[main_logger])
    print_log(f"Percentage: {swap.percent}%",[main_logger])

        
tb_logs = 'logs_tb'
model_logs = './logs_model/'

# train_image_folder_path = "F:\\melanoma_alex\\jpeg\\train\\1024\\"
# test_image_folder_path = "F:\\melanoma_alex\\jpeg\\test\\1024\\"
train_image_folder_path = "./jpeg/train/1024/"
test_image_folder_path = "./jpeg/test/1024/"

main_logger = myLogger('benchmark', False)
# main_logger = myLogger('benchmark')
# res_logger = myLogger('results', False)

print_log('=' * 80, [main_logger])
print_log('Begin logging', [main_logger])

df_train = pd.read_csv('train.csv')


print_log("Using NUMBA", [main_logger])

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

sysinfo()

print_log("\n" + "-"*20 + "  Benchmarking\n",[main_logger])
num_rounds = 3000 
print_log(f"---> Running {num_rounds} rounds",[main_logger])
start_r = time.perf_counter()
for i in range(num_rounds):
    ID = df_train.iloc[i].image_name
    path = train_image_folder_path+ID+".jpg"
    with open(path, 'rb') as f:
        tf = f.read() # Read whole file in the file_content string
        if simplejpeg.is_jpeg(tf):
            img_BGR = simplejpeg.decode_jpeg(tf, colorspace = 'bgr')
end_r = time.perf_counter()
print_log(f'Total time for Jpeg files loading: {end_r - start_r:0.3f}s', [main_logger])

print_log(f'---> Image size: {img_BGR.shape}', [main_logger])

start_MI = time.perf_counter()
for i in range(num_rounds):
    img_exp = img_BGR.copy()
    MI_norm = BGR2MI(img_BGR)
    MI_exp = np.expand_dims(MI_norm, axis = 2)
    img_exp = np.concatenate((img_exp, MI_exp), axis = 2)        
end_MI = time.perf_counter()
print_log(f'Total time for MI computing: {end_MI - start_MI:0.3f}s', [main_logger])
  
start_EI = time.perf_counter()
for i in range(num_rounds):
    img_exp = img_BGR.copy()
    EI_norm = BGR2EI(img_BGR)      
    EI_exp = np.expand_dims(EI_norm, axis = 2)
    img_exp = np.concatenate((img_exp, EI_exp), axis = 2)        
end_EI = time.perf_counter()
print_log(f'Total time for EI computing: {end_EI - start_EI:0.3f}s', [main_logger])

start_EMI = time.perf_counter()
for i in range(num_rounds):
    img_exp = img_BGR.copy()
    MI_norm = BGR2MI(img_BGR)
    MI_exp = np.expand_dims(MI_norm, axis = 2)
    img_exp = np.concatenate((img_exp, MI_exp), axis = 2)        
    EI_norm = BGR2EI(img_BGR)      
    EI_exp = np.expand_dims(EI_norm, axis = 2)
    img_exp = np.concatenate((img_exp, EI_exp), axis = 2)        
end_EMI = time.perf_counter()
print_log(f'Total time for EI+MI computing: {end_EMI - start_EMI:0.3f}s', [main_logger])



for logger in loggers:
    for handler in list(loggers[logger].handlers):
        print(handler)
        handler.close()
        loggers[logger].removeHandler(handler) 
            