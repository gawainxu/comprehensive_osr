#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 29 10:20:39 2024

@author: zhi
"""


import matplotlib.pyplot as plt
import numpy as np

"""
plot the change of augmentation methods
"""

"""
aurocs = [86.564, 88.108, 88.126, 88.52,  89.246, 89.652]   # 88.612,
stds = [1.05, 1.24, 1.77, 1.7,  1.1, 1.04]    # 0.6,
x = [i for i in range(len(aurocs))]

aurocs = np.array(aurocs)
stds = np.array(stds)
x = np.array(x)

labels = ["no augmentation", "mixup 0.1", "mixup 0.2", "cutmix", "vanilla GradMix", "GradMix"]    # "cutmix 0.2"

plt.plot(x, aurocs, "b", marker="*")
#plt.fill_between(x, aurocs-stds, aurocs+stds, alpha=0.2)
plt.xticks(x, labels, rotation=15, fontsize=10)
plt.grid(axis="y", linestyle="--", linewidth=0.5, zorder=0)
plt.grid(axis="x", linestyle="--", linewidth=0.5, zorder=0)
plt.ylabel("AUROC (Cifar10)", fontsize=15)
plt.xlabel("Augmentation Methods", fontsize=15)

plt.savefig("aug_cifar.pdf")
"""




aurocs = [75.082, 76.202, 77.154, 79.162-1.5,  77.7975, 78.062]      #78.008,
stds = [3.394682607,2.274405417,4.138940686,3.138251105, 3.424785881,1.933512348]    # 1.501156221,
x = [i for i in range(len(aurocs))]

aurocs = np.array(aurocs)
stds = np.array(stds)
x = np.array(x)
 
labels = ["no augmentation", "mixup 0.1", "mixup 0.2", "cutmix",  "vanilla GradMix", "GradMix"]

plt.plot(x, aurocs, "b", marker="*")
#plt.fill_between(x, aurocs-stds, aurocs+stds, alpha=0.2)
plt.xticks(x, labels, rotation=15, fontsize=10)
plt.grid(axis="y", linestyle="--", linewidth=0.5, zorder=0)
plt.grid(axis="x", linestyle="--", linewidth=0.5, zorder=0)
plt.ylabel("AUROC (TinyImgNet)", fontsize=15)
plt.xlabel("Augmentation Methods", fontsize=15)

plt.savefig("aug_imgnet.pdf")
