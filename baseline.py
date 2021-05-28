import pandas as pd
import numpy as np

targetdir = "/home/kaggle_training/ml_code_resource/house-prices-advanced-regression-techniques"

def feature_create(train,test):
    df_train = train.copy()
    df_test = test.copy()
    df_train["evel_set"] = 0
    df_test["evel_set"] = 1
    df_temp = pd.concat([df_train,df_test])
    del df_train,df_test
    
    
    
    df_train = df_temp[df_temp["evel_set"]==0]
    df_test = df_temp[df_temp["evel_set"]==1]
    df_train.drop("evel_set",axis=1,inplace=True)
    df_test.drop("evel_set",axis=1,inplace=True)
    del df_temp
    return df_train,df_test

def model_create(train):
    sc_x = StandardScaler()
    sc_y = StandardScaler()
    x_train = train
    y_train = train["SalePrice"]
    x_train.drop("SalePrice",axis=1,inplace=True)
    x_train.drop("Id",axis=1,inplace=True)
    
    return gbrt,sc_y

if __name__ == "__main__":
    df_train_org = pd.read_csv(targetdir + "/input/train.csv")
    df_test_org = pd.read_csv(targetdir + "/input/test.csv")

    df_train,df_test = feature_create(df_train_org,df_test_org)
    del df_train_org,df_test_org

    model,scaler = model_create(df_train)
    sc_x = StandardScaler()
    df_test_Id = df_test["Id"]
    df_test = df_test.drop("Id",axis=1)
    df_test.drop("SalePrice",axis=1,inplace=True)

    df_test_std = sc_x.fit_transform(df_test)
    pred = model.predict(df_test_std)
    pred = scaler.inverse_transform(pred)
    df_sub_pred = pd.DataFrame(pred).rename(columns={0:"SalePrice"})
    df_submit = pd.DataFrame({
        "Id": df_test_Id,
        "SalePrice": df_sub_pred["SalePrice"]
    })
    df_submit.to_csv(targetdir + '/submission.csv', 
