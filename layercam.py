#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb  2 23:05:06 2024

@author: zhi
"""

import os
import argparse
import pickle
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colormaps
import PIL
import copy

import torch
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torchvision.transforms.functional import to_pil_image

from networks.resnet_big import SupConResNet, LinearClassifier
from main_testing import load_model
from losses import SupConLoss
from dataUtil import get_gradcam_datasets, get_train_datasets, osr_splits_inliers
from torchvision import transforms
from dataUtil import mean_mapping, std_mapping
from util import TwoCropTransform

#https://towardsdatascience.com/grad-cam-in-pytorch-use-of-forward-and-backward-hooks-7eba5e38d569
#https://github.com/jacobgil/pytorch-grad-cam

# global variables
gradients = None
activations = None


def parse_option():
    
    parser = argparse.ArgumentParser('argument for grad cam')
    parser.add_argument('--datasets', type=str, default='tinyimgnet',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn"], help='dataset')
    parser.add_argument("--num_classes", type=int, default=6)
    parser.add_argument("--trail", type=int, default=3)
    parser.add_argument("--linear", type=bool, default=False)

    # for "single" and "sim"
    parser.add_argument("--class_idx", type=int, default=0)
    parser.add_argument("--data_id", type=int, default=30)
    # for "supcon"
    parser.add_argument("--class_idxs", type=list, default=[4])
    parser.add_argument("--data_ids", type=list, default=[[123,231,56,23, 30, 84], [65,67,23,87, 90, 100]])
    parser.add_argument("--img_size", type=int, default=64)
    parser.add_argument("--feature_id", type=int, default=5)
    parser.add_argument("--bsz", type=int, default=256)
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])  
    parser.add_argument("--randaug", type=int, default=0)
    
    parser.add_argument('--model', type=str, default="resnet18")
    parser.add_argument("--model_path", type=str, default="/save/SupCon/tinyimgnet_models/tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_3_128_256/ckpt_epoch_600.pth")
    #  tinyimgnet_resnet18_original_data__mixup_positive_alpha_1.0_beta_1.0_layersaliencymix_no_SimCLR_1.0_1.0_0.05_trail_3_128_256    tinyimgnet_resnet18_original_data__vanilia__SimCLR_1.0_1.0_0.05_trail_3_128_256
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss function')
    
    parser.add_argument("--mode", type=str, default="sim", choices=["single", "supcon", "sim"], help="Mode of the loss function that used to compute the loss")
    
    opt = parser.parse_args()
    opt.main_dir = os.getcwd()

    opt.output_path = opt.main_dir + "/" + opt.model_path.split("/")[-2] +"_cam/"+str(opt.class_idx)
    if opt.linear is False:
        opt.output_path += "_" + str(opt.feature_id)
    if not os.path.exists(opt.output_path):
        os.makedirs(opt.output_path)

    opt.model_path = opt.main_dir + opt.model_path
    opt.class_id = osr_splits_inliers[opt.datasets][opt.trail][opt.class_idx]
        
    opt.feature_path = "./featuremaps/" + str(opt.class_id)+"_"+str(opt.data_id)
    if not os.path.exists(opt.feature_path):
        os.makedirs(opt.feature_path)
        
    return opt


def backward_hook(module, grad_input, grad_output):
    
    global gradients
    print("Backward Hook Running")
    gradients = grad_output
    print (f"Gradient Size: {gradients[0].size()}")
    
    
def forward_hook(module, args, output):
    
    global activations
    print("Forward Hook Running")
    activations = output
    print (f"Gradient Size: {activations.size()}")
    
    
def process_heatmap(heatmap, img, save_path, opt):
    
    plt.close('all') 
    # relu on top of the heatmap
    heatmap = F.relu(heatmap)

    # normalize the heatmap 
    heatmap /= torch.max(heatmap)

    # draw the heatmap
    plt.matshow(heatmap.detach())
    
    # Create a figure and plot the first image
    fig, ax = plt.subplots()
    ax.axis('off') # removes the axis markers
    
    # First plot the original image
    ax.imshow(to_pil_image(img, mode='RGB'))

    # Resize the heatmap to the same size as the input image and defines
    # a resample algorithm for increasing image resolution
    # we need heatmap.detach() because it can't be converted to numpy array while
    # requiring gradients
    overlay = to_pil_image(heatmap.detach(), mode='F').resize((opt.img_size, opt.img_size), resample=PIL.Image.BICUBIC)

    # Apply any colormap you want
    cmap = colormaps['jet']
    overlay = (255 * cmap(np.asarray(overlay) * 3)[:, :, :3]).astype(np.uint8)     

    # Plot the heatmap on the same axes, 
    # but with alpha < 1 (this defines the transparency of the heatmap)
    ax.imshow(overlay, alpha=0.2, interpolation='nearest')

    # Show the plot
    #plt.show()
    fig.savefig(save_path)
    

def process_featuremap(feature_maps, img, opt):
    
    feature_maps = feature_maps.detach()
    num_features = feature_maps.shape[0]
    for n in range(num_features):
        plt.close("all")
        fig, ax = plt.subplots()
        ax.axis('off') # removes the axis markers
        ax.imshow(to_pil_image(img, mode='RGB'))
        f_n = feature_maps[n,:,:]
        f_n = to_pil_image(f_n, mode="F").resize((opt.img_size, opt.img_size), resample=PIL.Image.BILINEAR)
        cmap = colormaps['jet']
        f_n = (255 * cmap(np.asarray(f_n) ** 2)[:, :, :3]).astype(np.uint8)
        ax.imshow(f_n, alpha=0.4, interpolation='nearest')
        plt.savefig(opt.feature_path + "/" + str(n) + ".png")
    

if __name__ == "__main__":
    
    opt = parse_option()
    
    # load model
    model = SupConResNet(name=opt.model)
    if opt.linear is True:
        linear = LinearClassifier(name=opt.model, num_classes=opt.num_classes)
        model, linear = load_model(model, linear, opt.model_path)
        linear = linear.cpu()
    else:
        linear = None
        model = load_model(model, linear, opt.model_path)
    
    model.eval()
    model = model.cpu()
    
    mean = mean_mapping[opt.datasets]
    std = std_mapping[opt.datasets]
    normalize = transforms.Normalize(mean=mean, std=std)
     
    # register hook
    forward_hook = model.encoder.layer3[-1].register_forward_hook(hook=forward_hook) 
    backward_hook = model.encoder.layer3[-1].register_full_backward_hook(hook=backward_hook)
    
    # excute
    if opt.mode == "single":
        
        data_transform = transforms.Compose([#transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),                                     
                                             #transforms.RandomResizedCrop(size=64, scale=(0.2, 1.)),
                                             #transforms.RandomHorizontalFlip(),
                                             #transforms.RandomGrayscale(p=0.2),
                                             transforms.ToTensor(), normalize])
        # load dataset
        dataset = get_gradcam_datasets(opt, opt.class_idx)
        img, _ = dataset[opt.data_id]                           
        img_transformed = data_transform(img)
        if linear is not None:
            out = linear(model.encoder(img_transformed.unsqueeze(0)))     ### encoder!!!
            out_id = torch.squeeze(out)[opt.class_idx]
        else:
            out = model(img_transformed.unsqueeze(0))
            out_id = torch.squeeze(out)[opt.feature_id]

        out_id.backward()                                          # TODO if is it for the target output layer
        pooled_gradients = torch.mean(gradients[0], dim=[0, 2, 3])
        
        process_featuremap(activations[0], np.asarray(img), opt)
        # weight the channels by corresponding gradients
        for i in range(activations.size()[1]):
            activations[:, i, :, :] *= pooled_gradients[i]
        
        # average the channels of the activations
        heatmap = torch.mean(activations, dim=1).squeeze()
        opt.save_path = opt.output_path + "/" + str(opt.data_id) + ".png"
        process_heatmap(heatmap, np.asarray(img), opt.save_path, opt)
        
    elif opt.mode == "sim":

        opt.output_path = opt.main_dir + "/" + opt.model_path.split("/")[-2] +"_cam/"
        if not os.path.exists(opt.output_path):
            os.makedirs(opt.output_path)

        dataset = get_train_datasets(opt)
        dataset_cam = get_gradcam_datasets(opt)
        criterion = SupConLoss(temperature=opt.temp)    # SimCLR loss
        data_loader = torch.utils.data.DataLoader(dataset, batch_size=opt.bsz, shuffle=False,
                                                  num_workers=4, pin_memory=True, drop_last=True)
    
        for idx, (images, labels) in enumerate(data_loader):
            
            if idx!=12:
                continue
            
            images1 = images[0]
            images2 = images[1]
            images = torch.cat([images1, images2], dim=0)
            
            features = model(images)
            features1, features2 = torch.split(features, [256, 256], dim=0)
            features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)
            loss1 = criterion(features, labels)
            loss2 = criterion(features)          # torch.matmul(f1, f2.T)  #   labels
            loss = loss1 + loss2
            loss.backward()
        
            gradients = gradients[0]
         
            for i in range(opt.bsz):
                
                activation_i = activations[i]
                gradient_i = gradients[i]
                with torch.no_grad():
                    activation_maps = activation_i * F.relu(gradient_i)
                    cam = torch.sum(activation_maps, dim=0) #.unsqueeze(0)
                
                opt.save_path = opt.output_path + "/" + str(i) + ".png"
                imgs, _ = dataset_cam[idx*opt.bsz + i]                                      # TODO this index is only for one batch
                process_heatmap(cam, np.asarray(imgs), opt.save_path, opt)
            
            #cam = F.interpolate(cam, size=(32, 32), mode="bilinear", align_corners=False)
            #cam_min, cam_max = cam.min(), cam.max()
            #norm_cam = (cam - cam_min).div(cam_max - cam_min + 1e-8).data
            #norm_cam = torch.squeeze(norm_cam)
        
            #heatmap0 = torch.mean(activations[0], dim=0) 
            #heatmap1 = torch.mean(activations[1], dim=0) 
            
            
            #opt.save_path1 = opt.output_path + "/sim2.png"
            #process_heatmap(heatmap1, np.asarray(imgs), opt.save_path1, opt)
    
    elif opt.mode == "supcon":
        
        data_transform = transforms.Compose([transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),                                     
                                              transforms.RandomResizedCrop(size=64, scale=(0.2, 1.)),
                                              #transforms.RandomHorizontalFlip(),
                                              transforms.RandomGrayscale(p=0.2),
                                              transforms.ToTensor(), normalize])
        datasets = []
        imgs_transformed1 = []
        imgs_transformed2 = []
        imgs = []
        labels = []
        criterion = SupConLoss(temperature=opt.temp)
        for idx, ci in enumerate(opt.class_idxs):
            dataset = get_gradcam_datasets(opt, ci)
            datasets.append(dataset)
            class_data_ids = opt.data_ids[idx]
            for i in class_data_ids:
                img, label = dataset[i]
                img_transformed1 = data_transform(img)
                img_transformed2 = data_transform(img)
                img_transformed1 = torch.unsqueeze(img_transformed1, dim=0)
                img_transformed2 = torch.unsqueeze(img_transformed2, dim=0)
                imgs_transformed1.append(img_transformed1)
                imgs_transformed2.append(img_transformed2)
                imgs.append(np.array(img))
                labels.append(label)
        
        imgs_transformed1 = torch.cat(imgs_transformed1, dim=0)
        imgs_transformed2 = torch.cat(imgs_transformed2, dim=0)
        imgs_transformed = torch.cat((imgs_transformed1, imgs_transformed2), dim=0)
        labels = torch.tensor(labels)
        features = model(imgs_transformed)
        f1, f2 = torch.split(features, [imgs_transformed1.shape[0], imgs_transformed1.shape[0]], dim=0)
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        loss = criterion(features, labels)
        loss.backward()
        
        
        pooled_gradients = torch.mean(gradients[0], dim=[2, 3])
        for i in range(imgs_transformed1.shape[0]):             # *2
            process_featuremap(activations[i], imgs[i], opt)
           
            # weight the channels by corresponding gradients
            for d in range(activations.size()[1]):
                activations[i, d, :, :] *= pooled_gradients[i, d]
                
            opt.save_path = opt.main_dir + "/cam/grad_cam_" + opt.datasets + "_" + str(i) + ".png"
            heatmap = torch.mean(activations[i], dim=0) 
            process_heatmap(heatmap, imgs_transformed1[i], opt.save_path, opt)

    
    backward_hook.remove()
    forward_hook.remove()
    
    