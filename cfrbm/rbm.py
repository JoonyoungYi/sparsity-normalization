#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Mar  9 21:56:47 2018

@author: baifrank
"""
import torch


##建立RBM的类，声明模型中所有的参数，参数更新公式
class RBM(object):
      
    
    def __init__(self, num_visible, num_hidden,k=5):
        
        self.dim = (k*num_visible, num_hidden)
        self.num_visible = num_visible
        self.num_hidden = num_hidden
        self.k = k
        
    def params_init(self):
        
        ##torch.randn:从N(0,1)中随机抽取
        self.weights = torch.randn(self.dim)
        self.vbias = torch.randn(1, self.k*self.num_visible)
        self.hbias = torch.randn(1, self.num_hidden)
        self.D = torch.randn(self.num_visible,self.num_hidden)##conditional RBM的权重


    def sample_hidden(self, vis):
        
        wx = torch.mm(vis, self.weights)
        activation = wx + self.hbias.expand_as(wx)
        p_h_given_v = torch.sigmoid(activation)
        ##torch.rand:从[0,1]均匀分布中抽取一组随机数
        h1_sample = torch.sign(p_h_given_v-torch.rand(p_h_given_v.shape))##如果是nan那么sign之后为0
        h1_sample[h1_sample == -1] = 0

        return p_h_given_v, h1_sample
    
    ##条件RBM抽取隐藏层的函数
    def sample_hidden_c(self, vis, r_input):
        
        wx = torch.mm(vis, self.weights)
        activation = wx + self.hbias.expand_as(wx) + torch.mm(r_input, self.D)
        p_h_given_v = torch.sigmoid(activation)
        h1_sample = torch.sign(p_h_given_v-torch.rand(p_h_given_v.shape))
        h1_sample[h1_sample == -1] = 0
        
        return p_h_given_v, h1_sample



    def sample_visible(self, hid):
        
        wy = torch.mm(hid, self.weights.t())
        activation = wy + self.vbias.expand_as(wy)
        activation_exp = torch.exp(activation).view(-1,self.k)
        ##如果exp之后出现0值，则在下面的概率中直接赋一个比较小的值
        activation_exp[activation_exp == 0] = 0.01
        ##将其中inf值变成矩阵中的最大值
        max_activation_exp = activation_exp[activation_exp != float('inf')].max()
        activation_exp[activation_exp == float('inf')] = max_activation_exp
        activation_exp_sum = torch.mm(activation_exp,torch.ones(self.k,self.k).view(-1,self.k))
        max_activation_exp_sum = activation_exp_sum[activation_exp_sum != float('inf')].max()
        activation_exp_sum[activation_exp_sum == float('inf')] = max_activation_exp_sum
        activation_exp_sum[activation_exp_sum == 0] = 0.05

        ##如果相除之后出现inf值，则在下面的概率中直接赋值为1
        p_v_given_h = torch.div(activation_exp,activation_exp_sum).view(-1,self.k*self.num_visible)
        p_v_given_h[p_v_given_h == float('inf')] = 1
        v1_sample = torch.sign(p_v_given_h-torch.rand(p_v_given_h.shape))
        v1_sample[v1_sample == -1] = 0
        
        return p_v_given_h,v1_sample
    
    ##没有k，纯粹binary进行抽样
    def sample_visible_binary(self, hid):
        
        wy = torch.mm(hid, self.weights.t())
        activation = wy + self.vbias.expand_as(wy)
        p_v_given_h = torch.sigmoid(activation)
        v1_sample = torch.sign(p_v_given_h-torch.rand(p_v_given_h.shape))
        v1_sample[v1_sample == -1] = 0
        
        return p_v_given_h,v1_sample

    
    ##来回完成一次从v0->h0->v1的采样
    def sample_vhv(self,v0):
        
        _, h0 = self.sample_hidden(v0)
        pv1, v1 = self.sample_visible(h0)
        
        return v1,pv1
    
    ##没有k，纯粹的bianry抽样
    def sample_vhv_binary(self,v0):
        
        _, h0 = self.sample_hidden(v0)
        pv1, v1 = self.sample_visible_binary(h0)
        
        return v1,pv1
        
    
    def gradient(self, v0, vk, ph0, phk):
        
        gw = torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        gbv = torch.sum((v0 - vk), 0)
        gbh = torch.sum((ph0 - phk), 0)
        
        return gw,gbv,gbh
    
    
    def train(self, v0, vk, ph0, phk, prev_gw, prev_gbv, prev_gbh, w_lr=0.001,v_lr=0.001,h_lr=0.001,decay=0,momentum=0.9):
        
        ##添加学习率衰减和动量系数
        gw, gbv, gbh = self.gradient(v0, vk, ph0, phk)
        if decay:
            gw -= decay * self.weights
            gbv -= decay * self.vbias
            gbh -= decay * self.hbias
        self.weights += momentum * prev_gw + w_lr * gw
        self.vbias += momentum * prev_gbv + v_lr * gbv
        self.hbias += momentum * prev_gbh + h_lr * gbh
        
        
    ##条件RBM的梯度公式和参数更新公式    
    def gradient_c(self, v0, vk, ph0, phk, r_input):
        
        gw = torch.mm(v0.t(), ph0) - torch.mm(vk.t(), phk)
        gbv = torch.sum((v0 - vk), 0)
        gbh = torch.sum((ph0 - phk), 0)
        gD = torch.mm(r_input.t(), ph0 - phk )
        
        return gw,gbv,gbh,gD   
    
    def train_c(self, v0, vk, ph0, phk, r_input, prev_gw, prev_gbv, prev_gbh, prev_gD, w_lr=0.001,v_lr=0.001,h_lr=0.001,D_lr=0.001,decay=0,momentum=0.9):
        
        ##添加学习率衰减和动量系数
        gw, gbv, gbh, gD = self.gradient_c(v0, vk, ph0, phk, r_input)
        if decay:
            gw -= decay * self.weights
            gbv -= decay * self.vbias
            gbh -= decay * self.hbias
            gD -= decay * self.D
        self.weights += momentum * prev_gw + w_lr * gw
        self.vbias += momentum * prev_gbv + v_lr * gbv
        self.hbias += momentum * prev_gbh + h_lr * gbh
        self.D += momentum * prev_gD + D_lr * gD
        
    

        
        
        
        
        
        