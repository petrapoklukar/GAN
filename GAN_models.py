#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Nov 21 09:17:10 2019

@author: petrapoklukar

Base implementation thanks to https://github.com/eriklindernoren/PyTorch-GAN/blob/master/implementations/infogan/infogan.py
"""

import torch.nn as nn
import torch
import numpy as np

# --------------------------------------------------------------- #
# --- To keep track of the dimensions of convolutional layers --- #
# --------------------------------------------------------------- #
class TempPrintShape(nn.Module):
    def __init__(self, message):
        super(TempPrintShape, self).__init__()
        self.message = message
        
    def forward(self, feat):
#        print(self.message, feat.shape)
        return feat 

    
# ----------------------------------------------- #
# --- Convolutional Generator & Distriminator --- #
# ----------------------------------------------- #
class ConvolutionalGenerator(nn.Module):
    def __init__(self, config):
        super(ConvolutionalGenerator, self).__init__()
        self.latent_dim = config['latent_dim']
        self.channel_dims = config['channel_dims'] 
        self.dropout = config['dropout'] 

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(self.latent_dim, self.channel_dims[0], 4, 1, 0, bias=False),
            TempPrintShape('ConvTrans1'),
            nn.BatchNorm2d(self.channel_dims[0]),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),
            
            nn.ConvTranspose2d(self.channel_dims[0], self.channel_dims[1], 4, 2, 1, bias=False),
            TempPrintShape('ConvTrans2'),
            nn.BatchNorm2d(self.channel_dims[1]),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),

            nn.ConvTranspose2d(self.channel_dims[1], self.channel_dims[2], 4, 2, 1, bias=False),
            TempPrintShape('ConvTrans3'),
            nn.BatchNorm2d(self.channel_dims[2]),
            nn.ReLU(True),
            nn.Dropout(p=self.dropout),

            nn.ConvTranspose2d(self.channel_dims[2], self.channel_dims[3], 4, 2, 1, bias=False),
            TempPrintShape('ConvTrans4'),
            nn.Tanh())
        
    def forward(self, x):
        return self.conv(x)
    
    
class ConvolutionalDiscriminator(nn.Module):
    def __init__(self, config):
        super(ConvolutionalDiscriminator, self).__init__()
        self.channel_dims = config['channel_dims'] 
        self.dropout = config['dropout'] 
        self.conv = nn.Sequential(
            nn.Conv2d(self.channel_dims[0], self.channel_dims[1], 4, 2, 1, bias=False),
            TempPrintShape('Conv1'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout),
            
            nn.Conv2d(self.channel_dims[1], self.channel_dims[2], 4, 2, 1, bias=False),
            TempPrintShape('Conv2'),
            nn.BatchNorm2d(self.channel_dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout),
            
            nn.Conv2d(self.channel_dims[2], self.channel_dims[3], 4, 2, 1, bias=False),
            TempPrintShape('Conv3'),
            nn.BatchNorm2d(self.channel_dims[3]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout),

            nn.Conv2d(self.channel_dims[3], self.channel_dims[4], 4, 1, 0, bias=False),
            TempPrintShape('Conv4'),
            nn.Sigmoid())

    def forward(self, x):
        return self.conv(x).view(-1)
    
    
class ConvolutionalDiscriminator_D2(nn.Module):
    def __init__(self, config):
        super(ConvolutionalDiscriminator_D2, self).__init__()
        self.channel_dims = config['channel_dims'] 
        self.dropout = config['dropout'] 
        self.conv = nn.Sequential(
            nn.Conv2d(self.channel_dims[0], self.channel_dims[1], 4, 3, 1, bias=False),
            TempPrintShape('Conv1'),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout),
            
            nn.Conv2d(self.channel_dims[1], self.channel_dims[2], 4, 3, 1, bias=False),
            TempPrintShape('Conv2'),
            nn.BatchNorm2d(self.channel_dims[2]),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(p=self.dropout),

            nn.Conv2d(self.channel_dims[2], self.channel_dims[3], 4, 3, 0, bias=False),
            TempPrintShape('Conv3'),
            nn.Sigmoid())

    def forward(self, x):
        return self.conv(x).view(-1)



# ---------------------------------------- #
# --- Linear Generator & Distriminator --- #
# ---------------------------------------- #
class LinearGenerator(nn.Module):
    def __init__(self, config):
        super(LinearGenerator, self).__init__()
        self.latent_dim = config['latent_dim']
        self.linear_dims = config['linear_dims'] # [256, 512, 1024]
        self.image_size = config['image_size']
        self.image_channels = config['image_channels']        
        self.output_dim = self.image_size * self.image_size * self.image_channels
        self.dropout = config['dropout'] 

        self.lin = nn.Sequential(
                nn.Linear(self.latent_dim, self.linear_dims[0]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                
                nn.Linear(self.linear_dims[0], self.linear_dims[1]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                
                nn.Linear(self.linear_dims[1], self.linear_dims[2]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                
                nn.Linear(self.linear_dims[2], self.output_dim),
                nn.Tanh()
                )
        
    def forward(self, x):
        x = x.view(-1, self.latent_dim)
        out_lin = self.lin(x)
        out = out_lin.reshape(-1, self.image_channels, self.image_size, self.image_size)
        return out
    
    
class LinearDiscriminator(nn.Module):
    def __init__(self, config):
        super(LinearDiscriminator, self).__init__()
        
        self.linear_dims = config['linear_dims'] # [256, 512, 1024]
        self.image_size = config['image_size']
        self.image_channels = config['image_channels']        
        self.output_dim = self.image_size * self.image_size * self.image_channels
        self.dropout = config['dropout'] 

        self.lin = nn.Sequential(
                nn.Linear(self.output_dim, self.linear_dims[0]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                
                nn.Linear(self.linear_dims[0], self.linear_dims[1]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                
                nn.Linear(self.linear_dims[1], self.linear_dims[2]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                
                nn.Linear(self.linear_dims[2], 1),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = x.view(-1, self.output_dim)
        return self.lin(x)
    
    
class LinearDiscriminator_D2(nn.Module):
    def __init__(self, config):
        super(LinearDiscriminator_D2, self).__init__()
        self.linear_dims = config['linear_dims'] # [512, 256]
        self.image_size = config['image_size']
        self.image_channels = config['image_channels']        
        self.output_dim = self.image_size * self.image_size * self.image_channels
        self.dropout = config['dropout'] 

        self.lin = nn.Sequential(
                nn.Linear(self.output_dim, self.linear_dims[0]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                
                nn.Linear(self.linear_dims[0], self.linear_dims[1]),
                nn.ReLU(),
                nn.Dropout(p=self.dropout),
                
                nn.Linear(self.linear_dims[1], 1),
                nn.Sigmoid()
                )

    def forward(self, x):
        x = x.view(-1, self.output_dim)
        return self.lin(x)
    
    
# --------------------------------- #
# --- Testing the architectures --- #
# --------------------------------- #
if __name__ == '__main__':
    d_config = {
            'class_name': 'ConvolutionalDiscriminator',
            'channel_dims': [3, 16, 32, 64, 1],
            'dropout': 0.2
            }

    g_config = {
            'class_name': 'ConvolutionalGenerator',
            'latent_dim': 100,
            'channel_dims': [256, 128, 64, 3],
            'dropout': 0.2
            }
    
    G = ConvolutionalGenerator(g_config)
    D = ConvolutionalDiscriminator(d_config)
    
    z = torch.Tensor(size=(25, g_config['latent_dim'], 1, 1)).normal_()
    gen_x = G(z)
    is_valid = D(gen_x)
    
    print('Input noise: ', z.shape)
    print('Generated x: ', gen_x.shape)
    print('Is valid?: ', is_valid.shape)
    
    d1_config = {
            'class_name': 'ConvolutionalDiscriminator_D2',
            'channel_dims': [3, 16, 32, 1],
            'dropout': 0.2
            }
    d = ConvolutionalDiscriminator_D2(d1_config)
    print(d(gen_x))