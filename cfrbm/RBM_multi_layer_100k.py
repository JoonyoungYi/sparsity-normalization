#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Mar 13 11:55:39 2018

@author: baifrank
"""


import numpy as np
import pandas as pd
import torch
import torch.nn.parallel
from sklearn import linear_model
import matplotlib.pyplot as plt

from RBM_torch import dataset
from RBM_torch import utils
from RBM_torch import rbm
from RBM_torch import RBM_symm_utils

###########################################################################

if __name__ == "__main__":
    
    path = '/Users/baifrank/Desktop/ml-100k'
    train_path, test_path = path+'/ua.base', path+'/ua.test'
    
    ##导入用户数量，电影数量，torch形式的原始评分矩阵(train+test)//load number of users,movies and rating matrix
    nb_users,nb_movies,training_set,test_set = dataset.data_input(train_path,test_path)
    k = 5
        
    ##实值矩阵Binary化，列数*5//soft-max the real-value rating(to 0 and 1 rating)
    train_tensor_u = utils.expand(training_set, k, neg=0)
    train_tensor_i = utils.expand(training_set.t(), k, neg=0)
    train_tensor_neg_u = utils.expand(training_set, k, neg=1)
    train_tensor_neg_i = utils.expand(training_set.t(), k, neg=1)

    ##user/item based##
    def rbm_run(nh,nb_epoch,k_gibbs,batch_size,decay,momentum,item_based):
                
        l_mae_train, l_mse_train, l_mae_test, l_mse_test =[], [], [], []
        ##初始化一个rbm的class//initialize a class of RBM
        if item_based == True:
            train_tensor, train_tensor_neg = train_tensor_i, train_tensor_neg_i
            nb_rows, nv = nb_movies, nb_users
        else:
            train_tensor, train_tensor_neg = train_tensor_u, train_tensor_neg_u
            nb_rows, nv = nb_users, nb_movies

        rbm_model = rbm.RBM(nv,nh)
        rbm_model.params_init()
        
        ##开始迭代之前先初始化前一次迭代的梯度//initialize the previous gradient before the first iteration   
        prev_gw = torch.randn(rbm_model.dim)
        prev_gbv = torch.randn(1, rbm_model.k*rbm_model.num_visible)
        prev_gbh = torch.randn(1, rbm_model.num_hidden)
     
        ##Training the RBM
        for epoch in range(1, nb_epoch + 1):
            print("Calculating:"+str(epoch)+'/'+str(nb_epoch))
            for id_start in range(0, nb_rows - batch_size, batch_size):
                v0 = train_tensor[id_start:id_start+batch_size]
                vk = train_tensor[id_start:id_start+batch_size]
                v0_neg = train_tensor_neg[id_start:id_start+batch_size]
                ##Gibbs Sampling    
                ph0,h0 = rbm_model.sample_hidden(v0)
                for k in range(k_gibbs):
                    _,hk = rbm_model.sample_hidden(vk)
                    _,vk = rbm_model.sample_visible(hk)
                    vk[v0_neg == -1] = 0  
                phk,_ = rbm_model.sample_hidden(vk)
                                
                rbm_model.train(v0, vk, ph0, phk,prev_gw, prev_gbv, prev_gbh, w_lr=0.01,v_lr=0.01,h_lr=0.01,decay=decay,momentum=momentum)
                prev_gw, prev_gvb, prev_gvh = rbm_model.gradient(v0, vk, ph0, phk)
            ##重构之后计算误差//compute the error after reconstruction
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
    

    ##再以h_btm为可见层，建立RBM_layer1_u/biuld the 2 layer based on the bottom layer RBM
    ##定义只有bianry值的RBM(与k无关)：user/item based##
    def rbm_run_binary(data_input,nh,nb_epoch,k_gibbs,batch_size,decay,momentum):
        
        ##用来储存重构误差//store the reconstruction error
        l_mae = []
        nb_rows, nv = data_input.shape[0], data_input.shape[1]
        rbm_model = rbm.RBM(nv, nh, k=1)
        rbm_model.params_init()

        ##开始迭代之前先初始化前一次迭代的梯度//get the previous gradient before start the next iteration    
        prev_gw = torch.randn(rbm_model.dim)
        prev_gbv = torch.randn(1, rbm_model.num_visible)
        prev_gbh = torch.randn(1, rbm_model.num_hidden)
     
        ##Training the RBM
        for epoch in range(1, nb_epoch + 1):
            print("Calculating:"+str(epoch)+'/'+str(nb_epoch))
            for id_start in range(0, nb_rows - batch_size, batch_size):
                v0 = data_input[id_start:id_start+batch_size]
                vk = data_input[id_start:id_start+batch_size]
                ##Gibbs Sampling    
                ph0,h0 = rbm_model.sample_hidden(v0)
                for k in range(k_gibbs):
                    _,hk = rbm_model.sample_hidden(vk)
                    _,vk = rbm_model.sample_visible_binary(hk)
                phk,_ = rbm_model.sample_hidden(vk)
                                
                rbm_model.train(v0, vk, ph0, phk,prev_gw, prev_gbv, prev_gbh, w_lr=0.01,v_lr=0.01,h_lr=0.01,decay=decay,momentum=momentum)
                prev_gw, prev_gvb, prev_gvh = rbm_model.gradient(v0, vk, ph0, phk)
            ##重构之后计算误差//reconstruction error
            data_recons, _ = rbm_model.sample_vhv_binary(data_input)
            mae = torch.mean(torch.abs(data_input-data_recons))
            print(mae)
            l_mae.append(mae)
            
        return rbm_model, l_mae
    
    
    ##误差输出：l_mae_train, l_mse_train, l_mae_test,l_mse_test//output of error
    ##从底层至顶层隐藏层个数：100-200-100//100-200-100 units in hidden layers
    ##user_based
    ###############################构建最底层的RBM//build the bottom RBM###################################
    rbm_btm_u,l_mae_train_btm_u, l_mse_train_btm_u, l_mae_test_btm_u, l_mse_test_btm_u = rbm_run(nh=100,nb_epoch=200,k_gibbs=10,batch_size=100,decay=0,momentum=0,item_based=False)
    mae_train_multi_1, mse_train_multi_1 = l_mae_train_btm_u[-1], l_mse_train_btm_u[-1]
    mae_test_multi_1, mse_test_multi_1 = l_mae_test_btm_u[-1], l_mse_test_btm_u[-1]
    ##获得输出的h层:h_btm//get the hidden layer of the bottom layer RBM
    _, h_btm = rbm_btm_u.sample_hidden(train_tensor_u)

    ###############################构建第二层的RBM//build the second RBM###################################
    rbm_layer1_u ,l_mae_layer1 = rbm_run_binary(h_btm,nh=200,nb_epoch=200,k_gibbs=10,batch_size=100,decay=0,momentum=0)
    ##获得layer1层的隐藏层:h_layer1
    _, h_layer1 = rbm_layer1_u.sample_hidden(h_btm)
    ##建立到第二层RBM的时候，计算到最底层的重构误差
    _, h_btm_recons = rbm_layer1_u.sample_visible_binary(h_layer1) 
    p_v_btm_recons, _ = rbm_btm_u.sample_visible(h_btm_recons)
    v_btm_recons_revert = utils.revert_expected_value(p_v_btm_recons)
    v_btm_recons_revert[v_btm_recons_revert > 5] = 5

    mae_train_multi_2, mse_train_multi_2 = utils.calculate_error(training_set, v_btm_recons_revert)
    mae_test_multi_2, mse_test_multi_2 = utils.calculate_error(test_set, v_btm_recons_revert)
    
    ###############################构建第三层的RBM（顶层）############################
    rbm_layer2_u, l_mae_layer2 = rbm_run_binary(h_layer1,nh=100,nb_epoch=200,k_gibbs=10,batch_size=100,decay=0,momentum=0)
    ##获得layer2层的隐藏层:h_layer2
    _, h_layer2 = rbm_layer2_u.sample_hidden(h_layer1)
    ##建立到第三层RBM的时候，计算到最底层的重构误差
    _, h_layer1_recons = rbm_layer2_u.sample_visible_binary(h_layer2) 
    _, h_btm_recons = rbm_layer1_u.sample_visible_binary(h_layer1_recons) 
    p_v_btm_recons, _ = rbm_btm_u.sample_visible(h_btm_recons)
    v_btm_recons_revert = utils.revert_expected_value(p_v_btm_recons)
    v_btm_recons_revert[v_btm_recons_revert > 5] = 5

    mae_train_multi_3, mse_train_multi_3 = utils.calculate_error(training_set, v_btm_recons_revert)
    mae_test_multi_3, mse_test_multi_3 = utils.calculate_error(test_set, v_btm_recons_revert)

    ##item_based
    rbm_model3_i, l_mae_train3_i, l_mse_train3_i, l_mae_test3_i, l_mse_test3_i = rbm_run(nh=100,nb_epoch=400,k_gibbs=10,batch_size=100,decay=0,momentum=0,item_based=True)
    utils.show_min_loss(l_mae_train3_i, l_mae_test3_i)
    ##multi-layer:the same with user_based RBM













