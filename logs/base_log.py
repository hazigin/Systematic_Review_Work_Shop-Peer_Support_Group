#茶番と世間体
#Python Logging in Kaggle
#http://icebee.hatenablog.com/entry/2018/12/16/221533

## base_log.py
import os
from pathlib import Path
from logging import getLogger, Formatter, FileHandler, StreamHandler, INFO, DEBUG
import pathlib
import datetime

def create_logger(exp_version,path=None):
    
    if path == None:
        pwd_path = pathlib.Path(os.path.join(os.getcwd(),"logs")).resolve()    
    else:
        pwd_path = pathlib.Path(os.path.join(path,"logs")).resolve()    
    now = datetime.datetime.now()
    log_file = Path.joinpath(pwd_path,"{0}_{1}.log".format(exp_version,"{0:%Y%m%d}".format(now))).resolve()

    # logger
    logger_ = getLogger(exp_version)
    logger_.setLevel(DEBUG)

    # formatter
    fmr = Formatter("[%(levelname)s] %(asctime)s >>\t%(message)s")

    # file handler
    fh = FileHandler(log_file)
    fh.setLevel(DEBUG)
    fh.setFormatter(fmr)

    # stream handler
    ch = StreamHandler()
    ch.setLevel(INFO)
    ch.setFormatter(fmr)

    logger_.addHandler(fh)
    logger_.addHandler(ch)   

def get_logger(exp_version):
    return getLogger(exp_version)