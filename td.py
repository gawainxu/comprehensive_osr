#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 15:53:39 2024

@author: zhi
"""

import matplotlib.pyplot as plt
import argparse
import numpy as np
from dataUtil import get_gradcam_datasets, get_train_datasets

def parse_option():
    
    parser = argparse.ArgumentParser('argument for grad cam')
    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"], help='dataset')
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--action", type=str, default="training_supcon",
                    choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])  
    parser.add_argument("--randaug", type=int, default=0)
    
    opt = parser.parse_args()
 
    return opt


opt = parse_option()
dataset1 = get_train_datasets(opt)
dataset2 = get_gradcam_datasets(opt)


for i in range(0, 30000, 100):
    
    img1, _ = dataset1[i][0]
    img2, _ = dataset2[i]

    img1 = img1.numpy()

    img1 = np.transpose(img1, (2, 1, 0))
    
    plt.close('all') 
    plt.subplot(2, 1, 1)
    plt.imshow(img1)
    plt.subplot(2, 1, 2)
    plt.imshow(img2)
    plt.savefig("./plots/" + str(i) + ".png")


