#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Apr 24 14:38:44 2024

@author: zhi
"""

import pickle
import matplotlib.pyplot as plt


with open("loss_5_sim_plain", "rb") as f:
    loss = pickle.load(f)
    

loss0 = [l[0] for l in loss]
loss1 = [l[1] for l in loss]
loss2 = [l[2] for l in loss]

with open("loss_5_400", "rb") as f:
    loss_400 = pickle.load(f)
    

loss0_400 = [l[0] for l in loss_400]
loss1_400 = [l[1] for l in loss_400]
loss2_400 = [l[2] for l in loss_400]

plt.plot(loss0+loss0_400)