#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Mar 19 10:08:04 2018

@author: baifrank
"""

import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
from sklearn import linear_model
import matplotlib.pyplot as plt
import copy


from RBM_torch import dataset
from RBM_torch import utils
from RBM_torch import rbm
from RBM_torch import RBM_symm_utils

###########################################################################

if __name__ == "__main__":
    
    path_1m = '/Users/baifrank/Desktop/recomm_output/ml_1m_output'
    train_path, test_path = path_1m+'/ua.train.txt', path_1m+'/ua.test.txt'
    ##################---1m---###############################
    ##user id | item id | rating 
    ua_train = pd.read_table(path_1m+'/ua.train.txt',sep=' ')
    ua_test = pd.read_table(path_1m+'/ua.test.txt',sep=' ')

    R_train,R_test = ua_train[['user_id','item_id','rating']],ua_test[['user_id','item_id','rating']]

    uid_train = np.unique(ua_train['user_id'])
    iid_train = np.unique(ua_train['item_id'])
    iid_test = np.unique(ua_test['item_id'])
    iid_all = np.unique(np.append(iid_train,iid_test))

    nb_users,nb_movies = len(uid_train),len(iid_all)
    
    training_set = torch.FloatTensor(dataset.convert(R_train.as_matrix(),nb_users,nb_movies))
    test_set = torch.FloatTensor(dataset.convert(R_test.as_matrix(),nb_users,nb_movies))
    ##设置评分档数
    k = 5
        
    ##实值矩阵Binary化，列数*5
    train_tensor_u = utils.expand(training_set, k, neg=0)
    train_tensor_i = utils.expand(training_set.t(), k, neg=0)
    train_tensor_neg_u = utils.expand(training_set, k, neg=1)
    train_tensor_neg_i = utils.expand(training_set.t(), k, neg=1)
    
    ##构造条件RBM需要的：用户是否评分信息
    training_set_c = copy.deepcopy(training_set)
    training_set_c[training_set_c>0] = 1
    
    ##user/item based##
    def rbm_run(nh,nb_epoch,k_gibbs,batch_size,decay,momentum,item_based):
                
        l_mae_train, l_mse_train, l_mae_test, l_mse_test =[], [], [], []
        ##初始化一个rbm的class
        if item_based == True:
            train_tensor, train_tensor_neg = train_tensor_i, train_tensor_neg_i
            nb_rows, nv = nb_movies, nb_users
            r_input = training_set_c.t()
        else:
            train_tensor, train_tensor_neg = train_tensor_u, train_tensor_neg_u
            nb_rows, nv = nb_users, nb_movies
            r_input = training_set

        rbm_model = rbm.RBM(nv,nh)
        rbm_model.params_init()
        
        ##开始迭代之前先初始化前一次迭代的梯度    
        prev_gw = torch.randn(rbm_model.dim)
        prev_gbv = torch.randn(1, rbm_model.k*rbm_model.num_visible)
        prev_gbh = torch.randn(1, rbm_model.num_hidden)
        prev_gD = torch.randn(rbm_model.num_visible, rbm_model.num_hidden)
     
        ##Training the RBM
        for epoch in range(1, nb_epoch + 1):
            print("Calculating:"+str(epoch)+'/'+str(nb_epoch))
            for id_start in range(0, nb_rows - batch_size, batch_size):
                v0 = train_tensor[id_start:id_start+batch_size]
                vk = train_tensor[id_start:id_start+batch_size]
                v0_neg = train_tensor_neg[id_start:id_start+batch_size]
                r0_input = r_input[id_start:id_start+batch_size]
                ##Gibbs Sampling    
                ph0,h0 = rbm_model.sample_hidden_c(v0,r0_input)
                for k in range(k_gibbs):
                    _,hk = rbm_model.sample_hidden_c(vk,r0_input)
                    _,vk = rbm_model.sample_visible(hk)
                    vk[v0_neg == -1] = 0  
                phk,_ = rbm_model.sample_hidden_c(vk,r0_input)
                                
                rbm_model.train_c(v0,vk,ph0,phk,r0_input,prev_gw,prev_gbv,prev_gbh,prev_gD,w_lr=0.01,v_lr=0.01,h_lr=0.01,D_lr=0.001,decay=decay,momentum=momentum)
                prev_gw, prev_gvb, prev_gvh, prev_gD = rbm_model.gradient_c(v0, vk, ph0, phk, r0_input)
            ##重构之后计算误差
            data_recons = utils.predict(train_tensor, rbm_model, do_round = True, range_15 = True)
            if item_based == True:
                data_recons = data_recons.t()
            mae_train, mse_train = utils.calculate_error(training_set,data_recons)
            mae_test, mse_test = utils.calculate_error(test_set,data_recons)
            print(mae_train, mse_train, mae_test, mse_test)
            
            l_mae_train.append(mae_train)
            l_mse_train.append(mse_train)
            l_mae_test.append(mae_test)
            l_mse_test.append(mse_test)
            
        return rbm_model, l_mae_train, l_mse_train, l_mae_test,l_mse_test
    

    ##输出
    "l_mae_train, l_mse_train, l_mae_test,l_mse_test"
    ##user_based
    rbm_model3_u, l_mae_train3_u, l_mse_train3_u, l_mae_test3_u, l_mse_test3_u = rbm_run(nh=100,nb_epoch=200,k_gibbs=10,batch_size=100,decay=0,momentum=0,item_based=False)
    utils.show_min_loss(l_mae_train3_u, l_mae_test3_u)
    ##item_based
    rbm_model3_i, l_mae_train3_i, l_mse_train3_i, l_mae_test3_i, l_mse_test3_i = rbm_run(nh=100,nb_epoch=200,k_gibbs=10,batch_size=100,decay=0,momentum=0,item_based=True)
    utils.show_min_loss(l_mae_train3_i, l_mae_test3_i)
    
    ##作图观察误差走势
    x = range(0,400)
    y = l_mae_train3_u
    plt.figure()
    plt.plot(x,y,"-",label="nh=100,k_gibbs=10",color="red")
    plt.xlabel("iteration")
    plt.ylabel("mse")
    plt.title("User based RBM(Train set)")
    plt.legend()
    plt.show()

    ##基于uRBM和iRBM构建sRBM，并输出误差
    RBM_symm_utils.RBM_symm(rbm_model3_u,rbm_model3_i,train_tensor_u,train_tensor_i,training_set,test_set)
    
    ##将表现最好的RBM模型重构的打分结果储存在本地
    rbm_model_best_100k = rbm_model3_u
    recons_100k = utils.predict(train_tensor_u, rbm_model_best_100k, do_round = True, range_15 = True)
    l_recons_100k = recons_100k.numpy().flatten().tolist()
    ##定义对应的行与列的id
    
    ##将rbm的打分转化成dataframe输出到本地储存
    iid_list = list(range(1,1683))*943
    uid_list = []
    for i in range(1,944):
        for j in range(1682):
            uid_list.append(i)
    
    data_pre_100k={'iid':iid_list,'uid':uid_list,'pre':l_recons_100k}
    df_pre_100k = pd.DataFrame(data_pre_100k)
    df_pre_100k.to_csv("/Users/baifrank/Desktop/recomm_output/pre_best_100k.csv",index=False)

    