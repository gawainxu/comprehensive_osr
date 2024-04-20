#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Oct 11 13:57:45 2020

@author: zhi
"""

import os
import argparse
import numpy as np
import random
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from matplotlib.colors import ListedColormap
import matplotlib.colors as mcolors

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd 
import pickle 

import torch
import torch.nn as nn

from manipulate_features import find_dominant
from dataUtil import sortFeatures
from util import  feature_stats


def parse_option():

    parser = argparse.ArgumentParser('argument for visulization')
    parser.add_argument("--inlier_features_path", type=str, default="/features/cifar10_resnet18_original_data__vanilia__SimCLR_2.0_trail_0_128_warm_800_test_known")
    parser.add_argument("--outlier_features_path", type=str, default=None)  # 
    parser.add_argument("--inlier_features_path1", type=str, default=None)
    parser.add_argument("--outlier_features_path1", type=str, default=None) 
    parser.add_argument("--inlier_features_path2", type=str, default=None)
    parser.add_argument("--outlier_features_path2", type=str, default=None)
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--save_path", type=str, default="/plots/cifar10_resnet18_original_data__vanilia__SimCLR_2.0_trail_0_128_warm_800_test_known_tsne.pdf")
    parser.add_argument("--reduced_len", type=int, default=30)

    opt = parser.parse_args()
    opt.main_dir =os.getcwd()
    opt.inlier_features_path = opt.main_dir + opt.inlier_features_path
    opt.save_path = opt.main_dir + opt.save_path
    if opt.outlier_features_path is not None:
        opt.outlier_features_path = opt.main_dir + opt.outlier_features_path

    if opt.inlier_features_path1 is not None:
        opt.inlier_features_path1 = opt.main_dir + opt.inlier_features_path1

    if opt.outlier_features_path1 is not None:
        opt.outlier_features_path1 = opt.main_dir + opt.outlier_features_path1

    if opt.inlier_features_path2 is not None:
        opt.inlier_features_path2 = opt.main_dir + opt.inlier_features_path2

    if opt.outlier_features_path2 is not None:
        opt.outlier_features_path2 = opt.main_dir + opt.outlier_features_path2

    return opt


def pca(inMat, nComponents):
    
    # It is better to make PCA transformation before tSNE
    pcaFunction = PCA(nComponents)
    outMat = pcaFunction.fit_transform(inMat)

    return outMat    
    
    

def tSNE(inMat, nComponents):
    """
    The function used to visualize the high-dimensional hyper points 
    with t-SNE (t-distributed stochastic neighbor embedding)
    https://towardsdatascience.com/why-you-are-using-t-sne-wrong-502412aab0c0
    https://towardsdatascience.com/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b
    """
    
    inEmbedded = TSNE(n_components=nComponents, perplexity=30).fit_transform(inMat)
    return inEmbedded
    
"""
def feature_evaluation(sorted_features, features_stats):

    # intra similarity
    for features_c in sorted_features
"""

def class_centers(sorted_exemplar_features):

    centers = []
    for exemplar_features_c in sorted_exemplar_features:
        center = np.mean(np.array(exemplar_features_c), axis=0)
        centers.append(center)

    return centers


def center_similarity(testing_features, centers):

    closest_similarities = []
    for idx, testing_feature in enumerate(testing_features):
        similarities = []
        for center in centers:
            
            similarity = np.matmul(center, testing_feature) / np.linalg.norm(center) / np.linalg.norm(testing_feature)
            similarities.append(similarity)

        closest_similarities.append(np.max(np.array(similarities)))
        print(similarities)
    
    return closest_similarities

    
if __name__ == "__main__":
    
    
    opt = parse_option()
    
    with open(opt.inlier_features_path, "rb") as f:
        #_, features_inliers, _, labels_inliers = pickle.load(f)
        features_inliers, _, _, labels_inliers = pickle.load(f)                        #
        features_inliers = np.squeeze(np.array(features_inliers))
        print(features_inliers.shape)

    if opt.inlier_features_path1 is not None:
        with open(opt.inlier_features_path1, "rb") as f:
            features_inliers1, _, _, labels_inliers1 = pickle.load(f) 
            features_inliers1 = np.squeeze(np.array(features_inliers1))
        features_inliers = np.concatenate((features_inliers, features_inliers1), axis=1)

        if opt.inlier_features_path2 is not None:
            with open(opt.inlier_features_path2, "rb") as f:
                features_inliers2, _, _, labels_inliers2 = pickle.load(f) 
                features_inliers2 = np.squeeze(np.array(features_inliers2))
            features_inliers = np.concatenate((features_inliers, features_inliers2), axis=1)

    #features_inliers, removed_ind = find_dominant(features_inliers, opt.reduced_len)                    # !!!!!!!!!!!!!!!!!!!!
    sorted_features = sortFeatures(features_inliers, labels_inliers, opt.num_classes)
    features_stats = feature_stats(sorted_features)
    

    if opt.outlier_features_path is not None:
        with open(opt.outlier_features_path, "rb") as f:
            #_, features_outliers, _, labels_outliers = pickle.load(f)
            features_outliers, _, _, labels_outliers = pickle.load(f)
            features_outliers = np.squeeze(np.array(features_outliers))
            labels_outliers = np.squeeze(np.array(labels_outliers))
            
        if opt.outlier_features_path1 is not None:
           with open(opt.outlier_features_path1, "rb") as f:
                features_outliers1, _, _, labels_outliers1 = pickle.load(f)
                features_outliers1 = np.squeeze(np.array(features_outliers1))
                labels_outliers1 = np.squeeze(np.array(labels_outliers1))
           features_outliers = np.concatenate((features_outliers, features_outliers1), axis=1)

           if opt.outlier_features_path2 is not None:
            with open(opt.outlier_features_path2, "rb") as f:
                 features_outliers2, _, _, labels_outliers2 = pickle.load(f)
                 features_outliers2 = np.squeeze(np.array(features_outliers2))
                 labels_outliers2 = np.squeeze(np.array(labels_outliers2))
            features_outliers = np.concatenate((features_outliers, features_outliers2), axis=1)

        features_outliers = np.repeat(features_outliers, 20, axis=0)
        labels_outliers = np.repeat(labels_outliers, 20, axis=0)
        features_test = np.concatenate((features_inliers, features_outliers), axis=0)
        labels_test = np.concatenate((labels_inliers, labels_outliers), axis=0)
        
    else:
        features_test = features_inliers
        labels_test = labels_inliers

    #centers = class_centers(sorted_features)
    #closest_similarities_inliers = center_similarity(features_inliers, centers)
    #closest_similarities_outliers = center_similarity(features_outliers, centers)
    #print("Average Similarity Inliers: ", np.mean(np.array(closest_similarities_inliers)))
    #print("Average Similarity Outliers: ", np.mean(np.array(closest_similarities_outliers)))
        
    indices = range(len(features_test))
    indices = random.sample(indices, 2000)    # 5000
    
    features_test = features_test[indices]
    labels_test = labels_test[indices]
        
    features_SNE = np.empty([0, 2])    
    features = pca(np.squeeze(np.array(features_test)), 50)
    features = tSNE(features, 2)
    features_SNE = np.concatenate((features_SNE, features), 0)

    
    f = {"feature_1": features_SNE[:, 0], 
         "feature_2": features_SNE[:, 1],
         "label": labels_test}
    
    fp = pd.DataFrame(f)
    
    a4_dims = (8, 6)
    fig, ax = plt.subplots(figsize=a4_dims)
    
    colors1 = plt.cm.gist_heat_r(np.linspace(0.1, 1, opt.num_classes))
    colors2 = plt.cm.binary(np.linspace(0.99, 1, 1))
    colors = np.vstack((colors1, colors2))
    mymap = mcolors.LinearSegmentedColormap.from_list('my_colormap', colors)

    if opt.outlier_features_path is None:
        color_palette = sns.color_palette("hls", opt.num_classes)
    else:
        color_palette = sns.color_palette("hls", opt.num_classes) + ["k"]
     
    scatter_plot=sns.scatterplot(ax=ax, x="feature_1", y="feature_2", hue="label",
                                 palette=color_palette, data=fp,
                                 legend="brief", alpha=0.5)
    fig.savefig(opt.save_path)

    
    """
    https://medium.com/swlh/how-to-create-a-seaborn-palette-that-highlights-maximum-value-f614aecd706b
    
    'green','orange','brown','blue','red', 'yellow', 'pink', 'purple', 'c', 'grey'
    ,'brown','blue','red', 'yellow', 'pink', 'purple', 'c', 'grey',
                            'rosybrown', 'm', 'y', 'tan', 'lime', 'azure', 'sky', 'darkgreen',
                            'grape', 'jade'
    
    sns.color_palette("hls", num_classes)
    """