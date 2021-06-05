import pandas as pd
import numpy as np
import os
import torch
from pathlib import Path
import transformers
from transformers import BertTokenizer
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split,KFold
#from xgboost import XGBRegressor
import optuna.integration.lightgbm as lgb
from tools.base_log import create_logger, get_logger

from tqdm import tqdm
tqdm.pandas()

VERSION = "20210605_01" # 実験番号

data_dir = Path('/home/commonLit_readability_prize/input')
train_file = data_dir / 'train.csv'
test_file = data_dir / 'test.csv'
sample_file = data_dir / 'sample_submission.csv'

class BertSequenceVectorizer:
    def __init__(self):
        #self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = 'cpu'
        self.model_name = '/media/datasets/bert-base-uncased'  # Inet-not-connect
        #self.model_name = 'bert-base-uncased'          # Inet-connect
        self.tokenizer = BertTokenizer.from_pretrained(self.model_name)
        self.bert_model = transformers.BertModel.from_pretrained(self.model_name)
        self.bert_model = self.bert_model.to(self.device)
        self.max_len = 128

    def vectorize(self, sentence : str) -> np.array:
        inp = self.tokenizer.encode(sentence)
        len_inp = len(inp)

        if len_inp >= self.max_len:
            inputs = inp[:self.max_len]
            masks = [1] * self.max_len
        else:
            inputs = inp + [0] * (self.max_len - len_inp)
            masks = [1] * len_inp + [0] * (self.max_len - len_inp)

        inputs_tensor = torch.tensor([inputs], dtype=torch.long).to(self.device)
        masks_tensor = torch.tensor([masks], dtype=torch.long).to(self.device)

        bert_out = self.bert_model(inputs_tensor, masks_tensor)
        seq_out, pooled_out = bert_out['last_hidden_state'], bert_out['pooler_output']

        if torch.cuda.is_available():    
            return seq_out[0][0].cpu().detach().numpy()
        else:
            return seq_out[0][0].detach().numpy()

class CFG:
    input_size = (512, 512, 3) 
    height  = 512
    width = 512  
    batch_size = 8
    seed = 46
    debug = False
    n_epoch = 10
    lr = 5e-5
    weight_decay = 1e-06
    n_splits = 5


create_logger(VERSION)
get_logger(VERSION).info("START")

train0 = pd.read_csv(train_file)
train0[0:2]

test0 = pd.read_csv(test_file)
test0[0:2]

sample=pd.read_csv(sample_file)
sample['target'] = 0

train1=train0[['excerpt']].copy()
test1=test0[['excerpt']].copy()
target1=train0[['target']].copy()

data1=pd.concat([train1,test1])

n=len(train1)
BSV = BertSequenceVectorizer()

data1['excerpt_bert']=data1['excerpt'].progress_apply(lambda x: BSV.vectorize(x))

excerpt2=[]
for item in data1['excerpt_bert']:
    excerpt2+=[item]

X_train0=pd.DataFrame(excerpt2[0:n])
X_test0=pd.DataFrame(excerpt2[n:])
y_train0 = target1

skf = KFold(n_splits = CFG.n_splits)

epoch = 0

for train_index,val_index in skf.split(X_train0,y_train0):
    
    get_logger(VERSION).info("START EPOCH :{}".format(epoch))

    x_train, x_val = X_train0.iloc[train_index], X_train0.iloc[val_index]
    y_train, y_val = y_train0.iloc[train_index], y_train0.iloc[val_index]

    train_data = lgb.Dataset(x_train, label=y_train)
    val_data = lgb.Dataset(x_val, label=y_val)
    #2021/6/5 BEST PARAM
    #{'objective': 'regression', 'metric': 'l1', 'boosting': 'gbdt', 'verbosity': -1, 'feature_pre_filter': False, 'lambda_l1': 3.170007641952339e-05, 'lambda_l2': 1.196275292684629e-05,
    #  'num_leaves': 31, 'feature_fraction': 0.4, 'bagging_fraction': 1.0, 'bagging_freq': 0, 'min_child_samples': 20, 'num_iterations': 1000, 'early_stopping_round': None}
    
    params = {
        'objective': 'regression',
        'metric': 'mae',
        "boosting": "gbdt",
        "verbosity": -1,
    }

    model = lgb.train(
        params,
        train_data,
        valid_names=['train', 'valid'],
        valid_sets=[train_data, val_data],
        verbose_eval=100,
    )

    best_params = model.params
    print(best_params)
   
    preds = model.predict(X_test0, num_iteration=model.best_iteration)
    sample['target'] += preds / CFG.n_splits

    get_logger(VERSION).info("END EPOCH :{}".format(epoch))
    epoch += 1

sample.to_csv(data_dir /'submission.csv',index=None)
create_logger(VERSION)
get_logger(VERSION).info("END")