#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/SeqX2Y_PyTorch/project/models/seq2seq_4DCT_voxelmorph.py
Project: /workspace/SeqX2Y_PyTorch/project/models
Created Date: Wednesday January 17th 2024
Author: Hao Ouyang
-----
Comment:

Have a good code time :)
-----
Last Modified: Thursday April 18th 2024 4:50:52 am
Modified By: the developer formerly known as Hao Ouyang at <ouyanghaomail@gmail.com>
-----
Copyright (c) 2024 The University of Tsukuba
-----
HISTORY:
Date      	By	Comments
----------	---	---------------------------------------------------------
'''

import torch
import torch.nn as nn

from ConvLSTMCell3d import ConvLSTMCell
from layers import SpatialTransformer
from unet_utils import *

class EncoderDecoderConvLSTM(nn.Module):
    def __init__(self, nf, in_chan, size1, size2, size3):
        super(EncoderDecoderConvLSTM, self).__init__()

        """ ARCHITECTURE 

        # Encoder (ConvLSTM)
        # Encoder Vector (final hidden state of encoder)
        # Decoder (ConvLSTM) - takes Encoder Vector as input
        # Decoder (3D CNN) - produces regression predictions for our model

        """
        # BxCx1xDxWxH

        self.encoder1_conv = nn.Conv3d(in_channels=in_chan,
                                     out_channels=nf,
                                     kernel_size=(3, 3, 3),
                                     padding=(1, 1, 1))

        self.down1 = nn.MaxPool3d(kernel_size=2, stride=2)

        self.ConvLSTM3d1 = ConvLSTMCell(input_dim=nf,
                                        hidden_dim=nf,
                                        kernel_size=(3,3,3),
                                        bias=True)
        self.ConvLSTM3d2 = ConvLSTMCell(input_dim=nf,
                                        hidden_dim=nf,
                                        kernel_size=(3, 3, 3),
                                        bias=True)
        self.ConvLSTM3d3 = ConvLSTMCell(input_dim=nf,
                                        hidden_dim=nf,
                                        kernel_size=(3, 3, 3),
                                        bias=True)
        self.ConvLSTM3d4 = ConvLSTMCell(input_dim=nf,
                                        hidden_dim=nf,
                                        kernel_size=(3, 3, 3),
                                        bias=True)

        self.up1 = nn.Upsample(scale_factor=2, mode='trilinear')

        self.out = ConvOut(nf)

        self.transformer = SpatialTransformer((size1, size2, size3))




    def autoencoder(self, x, seq_len, rpm_x, rpm_y, future_step, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6, h_t7, c_t7): #!origin
    # def autoencoder(self, x, seq_len, rpm_x, rpm_y, future_step, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6):
        latent = []
        out = []
        # encoder
        e1 = []
        e2 = []
        e3 = []

        # for t in range(seq_len): # test_LUNA.py used this
        # for t in range(0, seq_len, -1):
        # for t in range(seq_len-1, 0, -1): # train.py used this
        for t in range(seq_len-1, -1, -1): # train.py used this
            #print(rpm_x.shape, rpm_y.shape)
            h_t1 = self.encoder1_conv(x[:,t,...])
            down1 = self.down1(h_t1)

            h_t4, c_t4 = self.ConvLSTM3d1(input_tensor=down1,
                                   cur_state=[h_t4,c_t4])
            h_t5, c_t5 = self.ConvLSTM3d2(input_tensor = h_t4, # c_t5 h_t5.shape=>[1,96,35,60,70]  input:(nf=96, in_chan=1, size1=70, size2=120, size3=140)
                                   cur_state = [h_t5,c_t5])

#------------------nadeemlab ORG--注意：原本的No2D中不需要解开此代码------------------------
            # ! here, multiply the rpm and feature
            # h_t5 = torch.mul(h_t5,torch.squeeze(rpm_x[0,t])) # from back to front
            # ！h_t5 = torch.mul(h_t5,torch.squeeze(rpm_x[0,t-1]))
            # simple multiplication between rpm and feature

#------------------myself add 1d rpm 我自己的对比试验新增代码------------------------
            # ! here, multiply the rpm and feature
            # 原本这行没被注释？ h_t5 = torch.mul(h_t5,torch.squeeze(rpm_x[0,t])) # from back to front
            # ！h_t5 = torch.mul(h_t5,torch.squeeze(rpm_x[0,t-1]))
            # simple multiplication between rpm and feature
 #------------------myself add 1d rpm ------------------------
                               
            encoder_vector = h_t5


        for t in range(future_step):

            h_t6, c_t6 = self.ConvLSTM3d3(input_tensor=encoder_vector,
                                   cur_state=[h_t6, c_t6])
            
            h_t7, c_t7 = self.ConvLSTM3d4(input_tensor=h_t6, # c_t7 h_t7.shape=>[1,96,35,60,70]  input:(nf=96, in_chan=1, size1=70, size2=120, size3=140)
                                   cur_state=[h_t7, c_t7])
            
#------------------nadeemlab ORG--注意：原本的No2D中不需要解开此代码------------------------
            # h_t7 = torch.mul(h_t7, torch.squeeze(rpm_y[0,t]))
#------------------nadeemlab ORG--注意：原本的No2D中不需要解开此代码------------------------
            # Simple multiplication between rpm and later phase features
            encoder_vector = h_t7
            latent += [h_t7]  # 了解到 h_t7 是一个形状为 torch.Size([1, 96, 35, 60, 70]) 的张量后，这行代码 latent += [h_t7] 的操作意味着将这个五维张量作为一个元素添加到名为 latent 的列表中。在这个上下文中，latent 可能被用来收集一系列的张量，每个张量可能代表不同时间步的潜在表示或特征图。通过这种方式，可以在列表中追踪并存储多个时间步的状态。

            # encoder_vector = h_t6 # delete 1 convlstm open this
            # latent += [h_t6]

        latent = torch.stack(latent,1)
        latent = latent.permute(0,2,1,3,4,5)
        timestep = latent.shape[2]

        output_img = []
        output_dvf = []
        # spatial transformer = transformer
        for i in range(timestep):
            output_ts = self.up1(latent[:,:,i,...]) # output_ts torch.Size([1, 96, 70, 120, 140])
            dvf = self.out(output_ts) # dvf torch.Size([1, 3, 70, 120, 140])
            # 这里的x[:,0,...]就代表了输入的初始相位图像X0 (initial phase image), 然后用spatial transform对其进行变换 
            warped_img = self.transformer(x[:,0,...],dvf) # warped_img torch.Size([1, 1, 70, 120, 140]), x torch.Size([1, 4, 1, 70, 120, 140])
            output_img += [warped_img] # 
            output_dvf += [dvf]

        output_img = torch.stack(output_img,1) # output_img torch.Size([1, 3, 1, 70, 120, 140])
        output_dvf = torch.stack(output_dvf,1) # output_dvf torch.Size([1, 3, 3, 70, 120, 140])
        output_img = output_img.permute(0,2,1,3,4,5) # output_img torch.Size([1, 1, 3, 70, 120, 140])
        output_dvf = output_dvf.permute(0,2,1,3,4,5) # output_dvf torch.Size([1, 3, 3, 70, 120, 140])

        return output_img, output_dvf


    def forward(self, x, rpm_x, rpm_y, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """

        # ? i think the seq_len need to map with the future seq
        # ? maybe, the seq_len mean that for one patient, in different seq_len break time (continuous)
        
        # find size of different input dimensions
        b, seq_len, _, d, h, w = x.size()

        # initialize hidden states
        # shape: 1, 96, 70, 112, 112
        h_t4, c_t4 = self.ConvLSTM3d1.init_hidden(batch_size=b, image_size=(int(d // 2),int(h // 2),int(w // 2)))
        h_t5, c_t5 = self.ConvLSTM3d2.init_hidden(batch_size=b, image_size=(int(d // 2), int(h // 2), int(w // 2)))
        h_t6, c_t6 = self.ConvLSTM3d3.init_hidden(batch_size=b, image_size=(int(d // 2), int(h // 2), int(w // 2)))
        h_t7, c_t7 = self.ConvLSTM3d4.init_hidden(batch_size=b, image_size=(int(d // 2), int(h // 2), int(w // 2))) #!origin

        # autoencoder forward
        # outputs = self.autoencoder(x, seq_len, future_seq, h_t1, c_t1, h_t2, c_t2, h_t3, c_t3, m_t3, h_t4, c_t4, m_t4,
        #                           h_t5, c_t5, m_t5, h_t6, c_t6, m_t6, h_t7, c_t7, h_t8, c_t8)
        outputs = self.autoencoder(x, seq_len, rpm_x, rpm_y, future_seq, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6, h_t7, c_t7) # !origin
        # outputs = self.autoencoder(x, seq_len, rpm_x, rpm_y, future_seq, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6) # delete 1 convlstm open this

        return outputs
