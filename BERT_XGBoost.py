import pandas as pd
import numpy as np
import os
import torch
from pathlib import Path
import transformers
from transformers import BertTokenizer
from sklearn.utils import shuffle
from sklearn.metrics import classification_report, log_loss, accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.model_selection import ShuffleSplit

from xgboost import XGBRegressor

from tqdm import tqdm
tqdm.pandas()

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

train0 = pd.read_csv(train_file)
train0[0:2]

test0 = pd.read_csv(test_file)
test0[0:2]

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

X_train0=excerpt2[0:n]
X_test0=excerpt2[n:]
y_train0 = target1

X = np.array(X_train0)
y = np.array(y_train0)

clf = XGBRegressor(max_depth=3,n_estimators=1000,learning_rate=0.01)

ss = ShuffleSplit(n_splits=5,train_size=0.8,test_size=0.2,random_state=0) 

for train_index, test_index in ss.split(X): 
    X_train, X_test = X[train_index], X[test_index]
    Y_train, Y_test = y[train_index], y[test_index]
    clf.fit(X_train, Y_train) 
    print(clf.score(X_test, Y_test))

y_pred = clf.predict(np.array(X_test0))

sample=pd.read_csv(sample_file)

subm=sample
subm['target']=y_pred
subm.to_csv(data_dir /'submission.csv',index=None)
