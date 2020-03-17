#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Mar 14 09:49:29 2020

@author: petrapoklukar
"""


config = {
        'discriminator_config': {
            'class_name': 'LinearDiscriminator_D2',
            'linear_dims': [512, 256],
            'dropout': 0.5,
            'image_channels': 1,
            'image_size': 32
            }, 

        'generator_config': {
            'class_name': 'LinearGenerator',
            'latent_dim': 100,
            'linear_dims': [256, 512, 1024],
            'dropout': 0.5,
            'image_channels': 1,
            'image_size': 32
            },
        
        'data_config': {
                'input_size': None,
                'usual_noise_dim': 100,
                'path_to_data': '../datasets/MNIST'
                },
                
        'train_config': {
                'batch_size': 256,
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
                'input_variance_increase': None,
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
        