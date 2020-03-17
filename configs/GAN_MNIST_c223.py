#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 14:56:20 2020

@author: petrapoklukar
"""


config = {
        'discriminator_config': {
            'class_name': 'ConvolutionalDiscriminator_D2',
            'channel_dims': [1, 64, 128, 1],
            'dropout': 0
            },

        'generator_config': {
            'class_name': 'ConvolutionalGenerator',
            'latent_dim': 100,
            'channel_dims': [256, 128, 64, 1],
            'dropout': 0.3
            },
        
        'data_config': {
                'input_size': None,
                'usual_noise_dim': 100,
                'path_to_data': '../datasets/MNIST'
                },
                
        'train_config': {
                'batch_size': 512,
                'epochs': 100,
                'snapshot': 20, 
                'console_print': 1,
                'optim_type': 'Adam',
                'gen_lr_schedule': [(0, 2e-4)],
                'gen_b1': 0.7,
                'gen_b2': 0.999,
                'dis_lr_schedule': [(0, 2e-4)],
                'dis_b1': 0.7,
                'dis_b2': 0.999,
                'input_noise': False,
                'input_variance_increase': 0,
                'grad_clip': True,
                'dis_grad_clip': 2,
                'gen_grad_clip': 1,
                
                'filename': 'gan',
                'random_seed': 1602
                },
                
        'eval_config': {
                'filepath': 'models/{0}/gan_model.pt',
                'load_checkpoint': False,
                'n_test_samples': 25,
                'n_repeats': 5
                }
        }