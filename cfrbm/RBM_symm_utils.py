#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:18:50 2018

@author: baifrank
"""
import numpy as np
import pandas as pd
from sklearn import linear_model

from RBM_torch import utils

def RBM_symm(u_rbm_best,i_rbm_best,train_tensor_u,train_tensor_i,training_set,test_set):
    
    ##将在test集上mae最小的u_RBM和i_RBM提取出来，构建s_RBM
    output_ui = utils.predict(train_tensor_u, u_rbm_best, do_round=True,range_15=True)
    output_iu = utils.predict(train_tensor_i, i_rbm_best, do_round=True,range_15=True)
    ##现在y为最终的真实评分，x分别为rating_ui和rating_iu，现在构造这3个变量的DataFrame
    ##将torch tensor变成array，并最终转化成dataframe
    ui_rating_train=output_ui[training_set > 0].numpy()
    iu_rating_train=output_iu.t()[training_set > 0].numpy()
    ui_rating_test=output_ui[test_set > 0].numpy()
    iu_rating_test=output_iu.t()[test_set > 0].numpy()
    ##提取y，即真实的打分
    y_rating_train=training_set[training_set > 0].numpy()
    y_rating_test=test_set[test_set > 0].numpy()
    data={'x_ui':ui_rating_train,'x_iu':iu_rating_train,'y':y_rating_train}
    data_train = pd.DataFrame(data)
    data={'x_ui':ui_rating_test,'x_iu':iu_rating_test,'y':y_rating_test}
    data_test = pd.DataFrame(data)
        
    ##使用岭回归进行模型拟合
    X,y = data_train[['x_ui','x_iu']],data_train['y']
    ##使用Ridge回归进行 a*x_ui+b*x_iu+intercept=y的线性模型拟合
    lr_ridge = linear_model.Ridge(alpha=0.1)
    X,y = data_train[['x_ui','x_iu']],data_train['y']
    lr_ridge.fit(X,y)
    yhat_train = lr_ridge.predict(X=data_train[['x_ui','x_iu']])
    yhat_test = lr_ridge.predict(X=data_test[['x_ui','x_iu']])
    
    yhat_train[yhat_train < 0] = 0
    yhat_train[yhat_train > 5] = 5
    yhat_train = np.round(yhat_train)
    yhat_test[yhat_test < 0] = 0
    yhat_test[yhat_test > 5] = 5
    yhat_test = np.round(yhat_test)
    
    mae_symm_train = sum(abs(yhat_train - data_train['y']))/len(yhat_train)
    mae_symm_test = sum(abs(yhat_test - data_test['y']))/len(yhat_test)
    mse_symm_train = sum(np.power(abs(yhat_train - data_train['y']),2))/len(yhat_train)
    mse_symm_test = sum(np.power(abs(yhat_test - data_test['y']),2))/len(yhat_test)
    
    print(mae_symm_train,mse_symm_train,mae_symm_test,mse_symm_test)
    
    
    
            
            
            
            
            
