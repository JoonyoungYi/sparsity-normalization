#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 20:51:32 2018

@author: baifrank
"""
import torch

def _expand_line(line, k=5):##将一个数组延长成0、1的形式，长度＊5
    expanded = [0] * (len(line) * k)
    for i, el in enumerate(line):
        if float(el) != 0.:
            el = float(el)
            expanded[(i*k) + int(round(el)) - 1] = 1
    return expanded


def _expand_line_neg(line, k=5):##将一个数组延长成0、1的形式，长度＊5,如果用户没有打分则赋值5个01
    expanded = [0] * (len(line) * k)
    for i, el in enumerate(line):
        if float(el) != 0.:
            el = float(el)
            expanded[(i*k) + int(round(el)) - 1] = 1
        if float(el) == 0.:
            expanded[(i*k):(k*(i+1))] = [-1]*k
    return expanded


def expand(data, k=5, neg=0):##将数据：np.array变成0/1binary矩阵的形式，列数变成以前的五倍
    new = []
    if neg:
        for m in data:
            new.append(_expand_line_neg(m.tolist()))
    else:
        for m in data:
            new.append(_expand_line(m.tolist()))
    return torch.FloatTensor(new)


def revert_expected_value(m, k=5, do_round=True):##由Binary的矩阵返回实数矩阵
    ##m是输出的的v＝1的条件概率矩阵，size不写死
    mask = torch.FloatTensor(range(1,6)).view(5,-1)
    nb_rows = m.shape[0]

    if do_round:
        expected_rt = torch.round(torch.mm(m.view(-1,5),mask))
    else:
        expected_rt = torch.mm(m.view(-1,5),mask)
    return expected_rt.view(nb_rows,-1)


def predict(input_data, rbm_input, do_round = True, range_15 = True):
    ##input_data为输入的矩阵，rbm_input为训练好的RBM模型
    _,pv1 = rbm_input.sample_vhv(input_data)
    v_recons_revert = revert_expected_value(pv1, rbm_input.k, do_round)
    if range_15 == True:
        v_recons_revert[v_recons_revert>5] = 5
    return v_recons_revert


##定义函数：输入真实评分矩阵和重构评分矩阵，输出mae、mse(输入的需要是实值的评分矩阵)
def calculate_error(data_real,data_recons):
    mat_dif = torch.abs(data_real-data_recons)
    mae = torch.mean(mat_dif[data_real > 0])        
    mse = torch.mean(torch.pow(mat_dif[data_real >0], 2))
    return mae, mse

##显示储存误差的list中最小值和其对应的位置
def show_min_loss(l_train,l_test):
    train_min, test_min = min(l_train), min(l_test)
    iter_train, iter_test = l_train.index(train_min), l_test.index(test_min)
    print("Iteration_"+str(iter_train)+"'s train_loss:"+str(train_min))   
    print("Iteration_"+str(iter_test)+"'s test_loss :"+str(test_min))        




