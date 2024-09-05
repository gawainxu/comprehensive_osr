#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 29 16:37:10 2024

@author: zhi
"""

import os
import pickle
import numpy as np

"""
folder = "/home/zhi/projects/datasets/CIFAR-10-C"
os.chdir(folder)

file_list = os.listdir(folder)

labels = np.load("labels.npy")

for name in file_list:
    if "label" not in name:
        data = np.load(name)
        for i in range(5):
            data_i = data[i*10000:(i+1)*10000]
            label_i = labels[i*10000:(i+1)*10000]
            
            new_name = name.split(".")[0] + "_" + str(i+1)
            with open(new_name, 'wb') as f:
                pickle.dump((data_i, label_i), f)
                
"""               
from torch.utils.data import Dataset

class cifar10_corruption(Dataset):
    
    def __init__(self, root, corruption, serverity):
        data_name = corruption + "_" + str(serverity)
        with open(root + data_name, "rb") as f:
            self.data, self.label = pickle.load(f)
            
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, index):
        return self.data[index], self.label[index]
    

root = "/home/zhi/projects/datasets/CIFAR-10-C/"
corruption = "fog"
serverity = 1
dataset = cifar10_corruption(root, corruption, serverity)
