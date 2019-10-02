#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 20:41:24 2018

@author: baifrank
"""

# Importing the libraries
import numpy as np
import pandas as pd
import torch

##id转变成连续整数
def continuous_id(df_input,df_uid_transform,df_iid_transform):
    
    ##将continuous id merge上original的dataframe
    df_step1 = pd.merge(df_input,df_uid_transform,left_on='user_id',right_on='uid',how='left')
    df_step2 = pd.merge(df_step1,df_iid_transform,left_on='item_id',right_on='iid',how='left')
    df_step3 = df_step2[['uid_transform','iid_transform','rating']]
    df_step3.columns = ['user_id','item_id','rating']
    return(df_step3)
    

##将DataFrame转化成user*item的矩阵形式
def convert(data,nb_users,nb_movies):
    
    new_data = []
    ##对每一个用户进行操作
    for id_users in range(1, nb_users + 1):
        id_movies = data[:,1][data[:,0] == id_users]##该用户评价过的movie的id
        id_ratings = data[:,2][data[:,0] == id_users]##相应的对movie的rating
        ratings = np.zeros(nb_movies)
        ratings[id_movies - 1] = id_ratings
        new_data.append(list(ratings))
    return new_data

##将数据以DataFrame的形式导入，在uid和iid连续化以后以矩阵的形式储存
def data_input(train_path,test_path):
    
    names_col = ['user_id','item_id','rating','timestamp']
    ua_train = pd.read_table(train_path,names=names_col)
    ua_test = pd.read_table(test_path,names=names_col)
    
    uid_train = np.unique(ua_train['user_id'])    
    iid_train = np.unique(ua_train['item_id'])
    iid_test = np.unique(ua_test['item_id'])

    nb_users = len(uid_train)##用户数量
    iid_all = np.unique(np.append(iid_train,iid_test))##train+test集所有电影数量
    nb_movies = len(iid_all)
    
    ##构造原始id与转化id对应的DataFrame
    uid_train.sort()
    df_uid_transform = {'uid':uid_train,'uid_transform':range(1,len(uid_train)+1)}
    df_uid_transform = pd.DataFrame(df_uid_transform)

    iid_all.sort()
    df_iid_transform = {'iid':iid_all,'iid_transform':range(1,len(iid_all)+1)}
    df_iid_transform = pd.DataFrame(df_iid_transform)
    
    '''
    ##将对照表写到本地
    df_uid_transform.to_csv("/Users/baifrank/Desktop/recomm_output/uid_transform.csv",index=False)
    df_iid_transform.to_csv("/Users/baifrank/Desktop/recomm_output/iid_transform.csv",index=False)
    '''
    ##id连续化
    R_train = continuous_id(ua_train,df_uid_transform,df_iid_transform)
    R_test = continuous_id(ua_test,df_uid_transform,df_iid_transform)

    ##将DataFrame转化成user*item的矩阵形式
    training_set = convert(R_train.as_matrix(),nb_users,nb_movies)
    test_set = convert(R_test.as_matrix(),nb_users,nb_movies)

    ##最终转化成torch评分矩阵
    training_set = torch.FloatTensor(training_set)
    test_set = torch.FloatTensor(test_set)
    
    return nb_users,nb_movies,training_set,test_set



