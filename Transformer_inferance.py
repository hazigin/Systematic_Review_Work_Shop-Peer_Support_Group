# TensorFlow related
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from pathlib import Path
import argparse
import json
import os
import gc
import sys
import datetime
import numpy as np
import pandas as pd

from sklearn.model_selection import StratifiedKFold

# HuggingFace related
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertModel
from kaggle_secrets import UserSecretsClient

if "debugpy" in sys.modules:
    current_path = '/home/commonLit_readability_prize/'
else:
    current_path = '.'

sys.path.append(os.path.join(current_path,"logs"))
from base_log import create_logger, get_logger

jsonpath = os.path.join(current_path,"configs/default.json")
parser = argparse.ArgumentParser()
parser.add_argument('--config', default=jsonpath)
options = parser.parse_args()
config = json.load(open(options.config))

datapath = Path(current_path,'data')
inputdatapath = Path(datapath,'input')
outputdatapath = Path(datapath,'output')
traindatapaht=Path(inputdatapath,"train.csv")
testdatapath=Path(inputdatapath,"test.csv")
sampledatapath=Path(inputdatapath,"sample_submission.csv")
submitpath=Path(outputdatapath,"submission.csv")
weightdatapath = Path(current_path,'weight')

id_col = config["ID_name"]
target_col = config["target_col"]
text_col = 'excerpt'

VERSION = config["exp_version"]
# modelの設定
batch_size = config["CFG"]["batch_size"]
epochs = config["CFG"]["n_epoch"]
seed = config["CFG"]["seed"]
epoch = config["CFG"]["epoch"]
max_token_length = config["CFG"]["max_token_length"]

# Use the tokenizer of your choice
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# Save the tokenizer so that you can download the files and move it to a Kaggle dataset.
#tokenizer.save_pretrained(save_dir)

create_logger(VERSION,current_path)
get_logger(VERSION).info(os.getcwd())
get_logger(VERSION).info("Transformer_inferance.py START:{0:%Y%m%d%H%M%S}.csv".format(datetime.datetime.now()))

df_test = pd.read_csv(testdatapath)
test_encodings = tokenizer(list(df_test.excerpt.values), truncation=True, padding='max_length', max_length=max_token_length)

# You can use a Transformer model of your choice.
transformer_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

AUTOTUNE = tf.data.AUTOTUNE

# Note that some tokenizers also returns 'token_id'. Modify this function accordingly. 
@tf.function
def parse_data(from_tokenizer):
    input_ids = from_tokenizer['input_ids']
    attention_mask = from_tokenizer['attention_mask']

    return {'input_ids': input_ids,
            'attention_mask': attention_mask}

# Utility function to build dataloaders
def get_testdataloaders(test_encodings):
    testloader = tf.data.Dataset.from_tensor_slices(test_encodings)

    testloader = (
        testloader
        .map(parse_data, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    
    return testloader

def CommonLitModel():
    # Input layers
    input_ids = Input(shape=(max_token_length,), dtype=tf.int32, name="input_ids")
    attention_mask = Input(shape=(max_token_length,), dtype=tf.int32, name="attention_mask")
    
    # Transformer backbone to extract features
    sequence_output = transformer_model(input_ids=input_ids, attention_mask=attention_mask)[0]
    clf_output = sequence_output[:, 0, :]
    
    # Dropout to regularize 
    clf_output = Dropout(0.1)(clf_output)
    
    # Output layer with linear activation as we are doing regression. 
    out = Dense(1, activation='linear')(clf_output)
    
    # Build model 
    model = Model(inputs=[input_ids, attention_mask], outputs=out)
    
    return model

# Sanity check model
tf.keras.backend.clear_session()
model = CommonLitModel()

model.load_weights(Path(weightdatapath,"weight-{epoch:04d}.ckpt".format(epoch=epoch)))

testloader = get_testdataloaders(test_encodings)

p_tst = model.predict(testloader)

sub = pd.read_csv(sampledatapath, index_col=id_col)
sub[target_col] = p_tst
sub.to_csv(submitpath)

get_logger(VERSION).info("Transformer_inferance.py END:{0:%Y%m%d%H%M%S}.csv".format(datetime.datetime.now()))