#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 13:49:34 2020

@author: petrapoklukar
"""

#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Mar 11 09:01:40 2020

@author: petrapoklukar
"""

config = {
        'discriminator_config': {
            'class_name': 'ConvolutionalDiscriminator',
            'channel_dims': [1, 64, 128, 256, 1]
            },

        'generator_config': {
            'class_name': 'ConvolutionalGenerator',
            'latent_dim': 100,
            'channel_dims': [256, 128, 64, 1]
            },
        
        'data_config': {
                'input_size': None,
                'usual_noise_dim': 100,
                'path_to_data': '../datasets/MNIST'
                },
                
        'train_config': {
                'batch_size': 128,
                'epochs': 10,
                'snapshot': 2, 
                'console_print': 1,
                'optim_type': 'Adam',
                'gen_lr_schedule': [(0, 1e-3)],
                'gen_b1': 0.5,
                'gen_b2': 0.999,
                'dis_lr_schedule': [(0, 2e-4)],
                'dis_b1': 0.5,
                'dis_b2': 0.999,
                
                'filename': 'gan',
                'random_seed': 1201
                },
                
        'eval_config': {
                'filepath': 'models/GAN_MNIST/gan_model.pt',
                'load_checkpoint': False,
                'n_test_samples': 25,
                'n_repeats': 5
                }
        }