#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jan 30 09:35:41 2020

@author: petrapoklukar
"""

import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import GAN_models
import GAN_training as model

import matplotlib
matplotlib.use('Agg') # Must be before importing matplotlib.pyplot or pylab!
import matplotlib.pyplot as plt
import numpy as np
from importlib.machinery import SourceFileLoader
import os
import argparse

parser = argparse.ArgumentParser(description='training a GAN')
parser.add_argument('--config_name', default=None, type=str, help='the path to save/load the model')
parser.add_argument('--train', default=0, type=int, help='set it to train the model')
parser.add_argument('--chpnt_path', default='', type=str, help='set it to train the model')
parser.add_argument('--eval', default=0, type=int, help='evaluates the trained model')
parser.add_argument('--device', default=None, type=str, help='the device for training, cpu or cuda')


class ImageDataset(Dataset):
    def __init__(self, dataset_name, path_to_data, device=None):
        if not device:
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        transform_list = transforms.Compose([
                transforms.Resize(32),
                transforms.ToTensor(),
                transforms.Normalize((0.5,), (0.5,))
                ])
        if dataset_name == 'MNIST':
            self.data = datasets.MNIST(path_to_data, train=True,
                                 download=True, transform=transform_list)
        
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        return self.data[idx][0]


if __name__ == '__main__':
    args = parser.parse_args()
    
#    # Laptop TESTING
#    args.config_name = 'GAN_MNIST_l201'
#    args.train = 1
#    args.chpnt_path = ''#'models/GAN_MNIST/gan_checkpoint9.pth'
#    args.device = None
#    args.eval = 1
    
    # Load config
    config_file = os.path.join('.', 'configs', args.config_name + '.py')
    export_directory = os.path.join('.', 'models', args.config_name)
    if (not os.path.isdir(export_directory)):
        os.makedirs(export_directory)
    
    print(' *- Config name: {0}'.format(args.config_name))
    
    config_file = SourceFileLoader(args.config_name, config_file).load_module().config 
    config_file['train_config']['exp_name'] = args.config_name
    config_file['train_config']['exp_dir'] = export_directory # the place where logs, models, and other stuff will be stored
    
    # Set the device
    if args.device == None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = args.device
        
    config_file['train_config']['device'] = device
    print('Device is: {}'.format(device))

    # Load the data 
    path_to_data = config_file['data_config']['path_to_data']
    dataset = ImageDataset('MNIST', path_to_data)
    # Laptop TESTING
#    dataset =  torch.utils.data.Subset(dataset, np.arange(100))
    dloader = DataLoader(dataset, batch_size=config_file['train_config']['batch_size'],
                         shuffle=True, num_workers=2)
    dloader_iter = iter(dloader)
    imgs = dloader_iter.next()
    print('Input data is of shape: {}'.format(imgs.shape))

    # Init the model
    model = model.GAN(config_file)
    
    # Train the model
    if args.train:
        model.train(dloader, chpnt_path=args.chpnt_path)

    # Evaluate the model
    if args.eval:
        eval_config = config_file['eval_config']
        if not args.train:
            model.load_model(eval_config)
        if str(device) == 'cpu': 
            # Reimport interactive backend
            import matplotlib
            matplotlib.use('Qt5Agg') # Must be before importing matplotlib.pyplot or pylab!
            import matplotlib.pyplot as plt
        
        n_samples = eval_config['n_test_samples']
        n_repeats = eval_config['n_repeats']

        # Sample usual noise and the categorical noise, fix structured continous noise
        for repeat in range(n_repeats):        
            
            # Sample the rest and keep the fixed one
            z_noise = model.sample_latent_noise(n_samples)
            
            # Generate an image
            gen_x = model.generator(z_noise)
            gen_x_plotrescale = (gen_x + 1.) / 2.0 # Cause of tanh activation
            filename = 'evalImages_r{}'.format(str(repeat))
            model.plot_image_grid(gen_x_plotrescale, filename, model.test_dir,
                                  n=n_samples)
            plt.close()

