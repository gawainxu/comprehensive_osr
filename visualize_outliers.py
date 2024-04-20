import torch

import os
import cv2
import argparse
import pickle
import numpy as np
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt

from util import  feature_stats
from distance_utils  import sortFeatures
from dataUtil import get_train_datasets, get_outlier_datasets
from dataUtil import osr_splits_inliers, osr_splits_outliers
from feature_reading import breaks

"""
The file to visualize the samples that are far from the center and close to the center
"""


def parse_option():

    parser = argparse.ArgumentParser('argument for visualization outliers')
    parser.add_argument("--train_feature_path", type=str, default="/features/mnist_resnet18_original_data__vanilia__SimCLR_1.0_trail_0_128_50_test_known")
    parser.add_argument("--outlier_feature_path", type=str, default="/features/mnist_resnet18_original_data__vanilia__SimCLR_1.0_trail_0_128_50_test_unknown")
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--datasets", type=str, default="mnist")
    parser.add_argument("--trail", type=int, default=0)
    parser.add_argument("--k_out", type=int, default=30)
    parser.add_argument("--action", type=str, default="outlier_visualization")
    parser.add_argument("--img_save_path", type=str, default="/temp_outlier/")

    opt = parser.parse_args()
    opt.main_dir = os.getcwd()
    opt.train_feature_path = opt.main_dir + opt.train_feature_path
    opt.outlier_feature_path = opt.main_dir + opt.outlier_feature_path
    opt.img_save_path = opt.main_dir + opt.img_save_path

    return opt


def pick_images(dataset, preds, distances, group, opt):

    images = []
    for i, img in enumerate(dataset):
        img = img.numpy()
        if img.shape[0] == 3:
            img = np.transpose(img, (1,2,0))
        
        pred = preds[i]
        dis = distances[i]

        if dis > 40:
            continue
      
        save_path = opt.img_save_path + str(pred) + "_" + str(round(dis)) + "_" + group + "_" + str(i) + ".png"
        print(save_path)
            
        if opt.datasets == "mnist":
            img = np.transpose(img, (1, 2, 0))
            print(img.shape)
            cv2.imwrite(save_path, img*255, [cv2.IMWRITE_PNG_COMPRESSION, 0])
        else:
            img = (1/(2 * 3)) * img + 0.5
            plt.imsave(save_path, img)
        images.append(images)
        
    return images


def downsample_dataset(dataset, ori_class_size, new_class_size, num_classes):
    
    imgs = []
    for nc in range(num_classes):
        for i in range(new_class_size):
            img, _ = dataset[nc*ori_class_size+i]
            imgs.append(img)

    return imgs
    


if __name__ == "__main__":
    
    opt = parse_option()

    opt.inlier_classes = osr_splits_inliers[opt.datasets][opt.trail]
    opt.ourlier_classes = osr_splits_outliers[opt.datasets][opt.trail]

    with open(opt.train_feature_path, "rb") as f:
        features_train, _, _, labels_train = pickle.load(f)   

    with open(opt.outlier_feature_path, "rb") as f:
        features_outlier, _, _, _ = pickle.load(f)

    sorted_features_train = sortFeatures(features_train, labels_train, opt)
    stats = feature_stats(sorted_features_train)
    dataset = get_outlier_datasets(opt)
    imgs = downsample_dataset(dataset, 500, 300, opt.num_classes)
    print("downsampled dataset ", len(imgs))

    distances = []
    preds = []
    for i, feature in enumerate(features_outlier):
        diss = []
        for nc in range(opt.num_classes):
            real_class_num = opt.inlier_classes[nc]
            mu, var = stats[nc]
            dis = mahalanobis(feature, mu, np.linalg.inv(var))
            diss.append(dis)

        pred = np.argmin(np.array(diss))
        pred = opt.inlier_classes[pred]
        print(pred)
        preds.append(pred)
        distances.append(np.max(np.array(diss)))

    images = pick_images(dataset=imgs, preds=preds, distances=distances, group="outlier", opt=opt) 