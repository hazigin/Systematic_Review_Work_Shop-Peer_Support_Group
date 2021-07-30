# TensorFlow related
import tensorflow as tf
print(tf.__version__)
from tensorflow.keras.layers import *
from tensorflow.keras.models import * 
from pathlib import Path
import argparse
import os
import gc
import numpy as np
import pandas as pd
import json
import sys
import datetime
from sklearn.model_selection import StratifiedKFold
from sklearn.model_selection import train_test_split

# HuggingFace related
from transformers import DistilBertTokenizerFast
from transformers import TFDistilBertModel

from kaggle_secrets import UserSecretsClient

if "debugpy" in sys.modules:
    current_path = '/home/Systematic_Review_Work_Shop-Peer_Support_Group'
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

VERSION = config["exp_version"]
# modelの設定
batch_size = config["CFG"]["batch_size"]
epochs = config["CFG"]["n_epoch"]
seed = config["CFG"]["seed"]
epoch = config["CFG"]["epoch"]
max_token_length = config["CFG"]["max_token_length"]
earlys_patience = config["CFG"]["earlys_patience"]
reduce_lr_plateau = config["CFG"]["reduce_lr_plateau"]

# Use the tokenizer of your choice
tokenizer = DistilBertTokenizerFast.from_pretrained('distilbert-base-uncased')
# Save the tokenizer so that you can download the files and move it to a Kaggle dataset.
#tokenizer.save_pretrained(save_dir)

create_logger(VERSION,current_path)
get_logger(VERSION).info(os.getcwd())
get_logger(VERSION).info("Transformer.py START:{0:%Y%m%d%H%M%S}.csv".format(datetime.datetime.now()))

# dataの準備
df_train = pd.read_csv(traindatapaht)
X_train, X_val, y_train, y_val = train_test_split(df_train.abstract.values, df_train.judgement.values,
                                                 test_size=0.2, random_state=seed)

train_encodings = tokenizer(list(X_train.astype('str')), truncation=True, padding=True, max_length=max_token_length)
val_encodings = tokenizer(list(X_val.astype('str')), truncation=True, padding=True, max_length=max_token_length)

AUTOTUNE = tf.data.AUTOTUNE

# Note that some tokenizers also returns 'token_id'. Modify this function accordingly. 
@tf.function
def parse_data(from_tokenizer, target):
    input_ids = from_tokenizer['input_ids']
    attention_mask = from_tokenizer['attention_mask']
    
    target = tf.cast(target, tf.float32)
    
    return {'input_ids': input_ids,
            'attention_mask': attention_mask}, target

# Utility function to build dataloaders
def get_dataloaders(train_encodings, train_label, val_encodings, val_label):
    trainloader = tf.data.Dataset.from_tensor_slices((dict(train_encodings), list(train_label)))
    validloader = tf.data.Dataset.from_tensor_slices((dict(val_encodings), list(val_label)))

    trainloader = (
        trainloader
        .shuffle(1024)
        .map(parse_data, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )

    validloader = (
        validloader
        .map(parse_data, num_parallel_calls=AUTOTUNE)
        .batch(batch_size)
        .prefetch(AUTOTUNE)
    )
    
    return trainloader, validloader

trainloader, validloader = get_dataloaders(train_encodings, y_train, val_encodings, y_val)

# You can use a Transformer model of your choice.
transformer_model = TFDistilBertModel.from_pretrained('distilbert-base-uncased')

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
    out = Dense(1, activation='sigmoid')(clf_output)
    
    # Build model 
    model = Model(inputs=[input_ids, attention_mask], outputs=out)
    
    return model

# Sanity check model
tf.keras.backend.clear_session()
model = CommonLitModel()

# Early stopping 
earlystopper = tf.keras.callbacks.EarlyStopping(
    monitor='val_loss', patience=earlys_patience, verbose=0, mode='min',
    restore_best_weights=True
)

# Reduce LR on Plateau
reducelrplateau = tf.keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', factor=0.2, patience=reduce_lr_plateau
)
   
# 4. Initialize model
tf.keras.backend.clear_session()
model = CommonLitModel()

# Compile
optimizer = tf.keras.optimizers.Adam(lr=1e-5)
model.compile(optimizer, loss='mean_squared_error', metrics=[tf.keras.metrics.RootMeanSquaredError()])    

# 5. Train the model
_ = model.fit(trainloader, 
            epochs=epochs, 
            validation_data=validloader,
            callbacks=[reducelrplateau,
                        earlystopper])

# 6. Evaluate on validation dataset.
loss, rmse = model.evaluate(validloader)

# 7. Save model
model.save_weights(Path(weightdatapath,"weight-{epoch:04d}.ckpt".format(epoch=epoch)))

get_logger(VERSION).info("Transformer.py END:{0:%Y%m%d%H%M%S}.csv".format(datetime.datetime.now()))