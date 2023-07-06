# -*- coding: utf-8 -*-
"""
Created on Fri Jun 30 13:50:57 2023

@author: alex
"""
import logging

loggers = {}
# global loggers

def myLogger(name, filename, t_format = True):
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
        handler = logging.FileHandler(f"{filename}")
        # handler = logging.FileHandler(f"{model_logs}{datetime.now().strftime('%Y%m%d')}-{nets_names[params['EFF_NETS'][0]]}_img{str(params['IMG_SIZES'][0])}-{name}.log")
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
    
    
def close_loggers(loggers)    :
    for logger in loggers:
        for handler in list(loggers[logger].handlers):
            print(handler)
            handler.close()
            loggers[logger].removeHandler(handler) 

