#!/usr/bin/env python3
# -*- coding:utf-8 -*-
'''
File: /workspace/SeqX2Y_PyTorch/project/models/Time_series_seq2seq_4DCT_voxelmorph.py
Project: /workspace/SeqX2Y_PyTorch/project/models
Created Date: Tuesday January 9th 2024
Author: Kaixu Chen
-----
Comment:

Have a good code time :)
-----
Last Modified: Wednesday January 17th 2024 9:50:54 am
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

# 3D CNN
class Encoder3DCNN(nn.Module):

    #! I think the 3D CNN structure have some problem.
    def __init__(self, in_channels, out_channels):
        super(Encoder3DCNN, self).__init__()
        
        self.conv1 = nn.Conv3d(in_channels, 64, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv3d(64, 128, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU(inplace=True)
        self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2)) # 原本老陈写的是这个，2d正方形图片整个输入用这个
        # self.pool1 = nn.MaxPool3d(kernel_size=(1, 2, 1), stride=(1, 2, 1)) # 当输入2d图片改变为左右两半，就需要用这个，不然shape不对

        self.conv3 = nn.Conv3d(128, 256, kernel_size=3, padding=1)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool3d(kernel_size=(2, 1, 1), stride=(2, 1, 1))

        self.conv4 = nn.Conv3d(256, out_channels, kernel_size=1)  # Reduce channel depth to desired output
        self.relu4 = nn.ReLU(inplace=True)
        
        # self.conv3 = nn.Conv3d(128, out_channels, kernel_size=3, padding=1)
        # self.relu3 = nn.ReLU(inplace=True)

        # self.pool = nn.MaxPool3d(kernel_size=2, stride=2)
    
    def forward(self, x):
        x = self.relu1(self.conv1(x))
        x = self.relu2(self.conv2(x))
        x = self.pool1(x)
        x = self.relu3(self.conv3(x))
        x = self.pool2(x)
        x = self.relu4(self.conv4(x)) # 在这准备增加pytorch-grad-cam
        # x = self.relu1(self.conv1(x))
        # x = self.relu2(self.conv2(x))
        # x = self.relu3(self.conv3(x))
        # x = self.pool(x)

        return x # torch.Size([1, 96, 1, 64, 64])

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

        # 添加3D CNN encoder
        self.encoder3d_cnn = Encoder3DCNN(in_channels=3, out_channels=nf)

    def autoencoder(self, x, seq_len, batch_2D, future_step, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6, h_t7, c_t7): #!origin
        """ autoencoder-decoder structure for 4DCT and time series data.

        Args:
            x (torch.Tensor): 4DCT data, (batch, seq_len, channel, depth, height, width)
            seq_len (int): the length of the 4DCT data
            batch_2D (torch.Tensor): time series data, (batch, channel, t, height, width)
            future_step (_type_): _description_
        """        

        latent = []
        
        # for t in range(seq_len): # test_LUNA.py used this
        # for t in range(seq_len-1, 0, -1): # In4 Out3 use This 这个循环将从 seq_len-1 开始，逐步递减，直到但不包括0。例如，如果 seq_len 是5，则循环将遍历以下序列：4, 3, 2, 1
        for t in range(seq_len-1, -1, -1): # In3 Out3 use This

            # 应用3D CNN encoder
            time_series_fat = self.encoder3d_cnn(batch_2D[:, :, :-1, ...])  # batch_2D torch.Size([1, 3, 3, 128, 128])
            # time_series_fat = self.encoder3d_cnn(batch_2D[:, :, :, ...]) 

            h_t1 = self.encoder1_conv(x[:,t,...])
            down1 = self.down1(h_t1)

            h_t4, c_t4 = self.ConvLSTM3d1(input_tensor=down1,
                                   cur_state=[h_t4,c_t4])
            h_t5, c_t5 = self.ConvLSTM3d2(input_tensor = h_t4, # c_t5 h_t5.shape=>[1,96,35,60,70]  input:(nf=96, in_chan=1, size1=70, size2=120, size3=140)
                                   cur_state = [h_t5,c_t5])

            # check shape 
            assert len(h_t5.shape) == len(time_series_fat.shape), "the dimension of h_t5 and batch_2D_encoded is not same."

            # fuse the 4DCT feature and time series feature, for encoder
            # encoder_vector = h_t5 @ time_series_fat # 相乘，需要解开
            encoder_vector = h_t5 + time_series_fat

            # 2d的时候以下这段原本是没注释的，这样上面的h_t5 @ time_series_fat就没用了，不知到是否存在问题？现将其注释再进行计算
            # encoder_vector = h_t5

        for t in range(future_step): # 这里不用改成future_step-1

            time_series_fat = self.encoder3d_cnn(batch_2D[:, :, 1:, ...])
            # time_series_fat = self.encoder3d_cnn(batch_2D[:, :, :, ...])

            h_t6, c_t6 = self.ConvLSTM3d3(input_tensor=encoder_vector,
                                   cur_state=[h_t6, c_t6])
            h_t7, c_t7 = self.ConvLSTM3d4(input_tensor=h_t6, # c_t7 h_t7.shape=>[1,96,35,60,70]  input:(nf=96, in_chan=1, size1=70, size2=120, size3=140)
                                   cur_state=[h_t7, c_t7])

            # check shape
            assert len(h_t7.shape) == len(time_series_fat.shape), "the dimension of h_t7 and batch_2D_encoded is not same."

            # fuse the 4DCT feature and time series feature, for decoder
            # decoder_vector = h_t7 @ time_series_fat # 相乘，需要解开
            decoder_vector = h_t7 + time_series_fat
            
            # 2d的时候以下这段原本是没注释的，这样上面的h_t7 @ time_series_fat就没用了，不知到是否存在问题？现将其注释再进行计算
            # decoder_vector = h_t7

            latent += [decoder_vector]
            # 了解到 h_t7 是一个形状为 torch.Size([1, 96, 35, 60, 70]) 的张量后，这行代码 latent += [h_t7] 的操作意味着将这个五维张量作为一个元素添加到名为 latent 的列表中。在这个上下文中，latent 可能被用来收集一系列的张量，每个张量可能代表不同时间步的潜在表示或特征图。通过这种方式，可以在列表中追踪并存储多个时间步的状态。

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


    def forward(self, x, batch_2D, future_seq=0, hidden_state=None):

        """
        Parameters
        ----------
        input_tensor:
            5-D Tensor of shape (b, t, c, h, w)        #   batch, time, channel, height, width
        """
        
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
        outputs = self.autoencoder(x, seq_len, batch_2D, future_seq, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6, h_t7, c_t7) # !origin
        # outputs = self.autoencoder(x, seq_len, rpm_x, rpm_y, future_seq, h_t4, c_t4, h_t5, c_t5, h_t6, c_t6) # delete 1 convlstm open this

        return outputs
