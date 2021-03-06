#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar 12 15:44:15 2020

@author: petrapoklukar
"""

config = {
        'discriminator_config': {
            'class_name': 'ConvolutionalDiscriminator_D2',
            'channel_dims': [1, 64, 128, 1],
            'dropout': 0.3
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
                'gen_b1': 0.8,
                'gen_b2': 0.999,
                'dis_lr_schedule': [(0, 2e-4)],
                'dis_b1': 0.8,
                'dis_b2': 0.999,
                'input_noise': True,
                'input_variance_increase': 1,
                'grad_clip': False,
                'dis_grad_clip': None,
                'gen_grad_clip': None,
                
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