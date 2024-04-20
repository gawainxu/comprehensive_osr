from __future__ import print_function

import os
import sys
import argparse
import time
import math
import pickle
import random
import copy

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model, label_convert
from dataUtil import mixup_positive_features, mixup_negative_features, vanilla_mixup
from dataUtil import num_inlier_classes_mapping, mixup_hybrid_features
from networks.resnet_big import SupConResNet, LinearClassifier
from networks.simCNN import simCNN_contrastive
from networks.resnet_preact import SupConpPreactResNet
from networks.mlp import SupConMLP
from losses import SupConLoss
from loss_mixup import SupConLoss_mix
from dataUtil import get_train_datasets, mixup_negative

import matplotlib
matplotlib.use('Agg')

try:
    import apex
    from apex import amp, optimizers
except ImportError:
    pass


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--save_freq', type=int, default=50,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='1000',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument("--pretrained", type=int, default=1)

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18', choices=["resnet18", "resnet34", "preactresnet18", "preactresnet34", "simCNN", "MLP"])
    parser.add_argument('--datasets', type=str, default='cifar10',
                        choices=["cifar-10-100-10", "cifar-10-100-50", 'cifar10', "tinyimgnet", 'mnist', "svhn", "cub", "aircraft"], help='dataset')
    parser.add_argument('--mean', type=str, help='mean of dataset in path in form of str tuple')
    parser.add_argument('--std', type=str, help='std of dataset in path in form of str tuple')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size', type=int, default=32, help='parameter for RandomResizedCrop')
    parser.add_argument("--augmentation_list", type=list, default=[])
    parser.add_argument("--argmentation_n", type=int, default=1)
    parser.add_argument("--argmentation_m", type=int, default=6)

    # method
    parser.add_argument('--method', type=str, default='SupCon',
                        choices=['SupCon', 'SimCLR', "SimCLR_CE", "MoCo"], help='choose method')
    parser.add_argument("--method_lam", type=float, default=1.0)
    parser.add_argument("--trail", type=int, default=0, choices=[0,1,2,3,4,5], help="index of repeating training")
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    # temperature
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss')
    parser.add_argument("--clip", type=float, default=None, help="for gradient clipping")

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument("--augmentation_method", type=str, default="vanilia", 
                        choices=["vanilia", "mixup_negative", "mixup_positive", "mixup_hybrid", "mixup_vanilla", "mixup_vanilla_features"])
    parser.add_argument("--data_method", type=str, default="original", 
                        choices=["original", "upsampling"])
    parser.add_argument("--architecture", type=str, default="single", choices=["single", "multi"])
    parser.add_argument("--ensemble_num", type=int, default=1)
    parser.add_argument("--feat_dim", type=int, default=128)


    # upsampling parameters
    parser.add_argument("--upsample", type=bool, default=False)
    parser.add_argument("--portion_out", type=float, default=0.5)
    parser.add_argument("--upsample_times", type=int, default=1)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--last_model_path", type=str, default=None)

    # mixup parameters
    parser.add_argument("--alpha_negative", type=float, default=0.2, help="between 0.2 to 0.4")
    parser.add_argument("--alpha_positive", type=float, default=0.2, help="between 0.2 to 0.4")
    parser.add_argument("--alpha_hybrid", type=float, default=10)
    parser.add_argument("--beta_hybrid", type=float, default=0.3)
    parser.add_argument("--alpha_vanilla", type=float, default=10)
    parser.add_argument("--beta_vanilla", type=float, default=0.3)
    parser.add_argument("--intra_inter_mix_positive", type=bool, default=True, help="intra=True, inter=False")
    parser.add_argument("--intra_inter_mix_negative", type=bool, default=True, help="intra=True, inter=False")
    parser.add_argument("--intra_inter_mix_hybrid", type=bool, default=True, help="intra=True, inter=False")
    parser.add_argument("--mixup_positive", type=bool, default=False)
    parser.add_argument("--mixup_negative", type=bool, default=False)
    parser.add_argument("--mixup_hybrid", type=bool, default=False)
    parser.add_argument("--mixup_vanilla", type=bool, default=False)
    parser.add_argument("--mixup_vanilla_features", type=bool, default=False)
    parser.add_argument("--positive_p", type=float, default=0.5)
    parser.add_argument("--alfa", type=float, default=1)
    parser.add_argument("--positive_method", type=str, default="no", choices=["max_similarity", "min_similarity", "random", "prob_similarity", "self_contract", "no"])
    parser.add_argument("--negative_method", type=str, default="no", choices=["max_similarity", "random", "even", "no"])
    parser.add_argument("--hybrid_method", type=str, default="no", choices=["min_similarity", "random", "no"])
    parser.add_argument("--vanilla_method", type=str, default="no", choices=["reverse", "random", "no"])

    opt = parser.parse_args()

    # check if dataset is path that passed required arguments
    if opt.datasets == 'path':
        assert opt.data_folder is not None \
            and opt.mean is not None \
            and opt.std is not None

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = '../datasets/'
    opt.model_path = './save/SupCon/{}_models'.format(opt.datasets)

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))

    opt.model_name = opt.datasets + "_" + opt.model
    if opt.data_method == "upsampling":
        opt.model_name += "_upsampling_data_" + str(opt.portion_out) + "_" + str(opt.upsample_times)
    else:
        opt.model_name += "_original_data_"

    if opt.augmentation_method == "mixup_negative":
        opt.model_name += "_mixup_negative_" + str(opt.negative_method) + "_intra_" + str(opt.intra_inter_mix_negative) + "_alpha_" + str(opt.alpha_negative)  
    elif opt.augmentation_method == "mixup_hybrid":
        opt.model_name += "_mixup_hybrid_" + str(opt.hybrid_method) + "_intra_" + str(opt.intra_inter_mix_hybrid) +  "_alpha_" + str(opt.alpha_hybrid) + "_beta_" + str(opt.beta_hybrid) + "_alfa_" + str(opt.alfa)             
    elif opt.augmentation_method == "mixup_positive":
        opt.model_name += "_mixup_positive_" + str(opt.positive_method) + "_intra_" + str(opt.intra_inter_mix_positive) + "_alpha_" + str(opt.alpha_positive) + "_p_" + str(opt.positive_p)
    elif opt.augmentation_method == "mixup_vanilla":
        opt.model_name += "_mixup_vanilla_" +  opt.vanilla_method + "_alpha_" + str(opt.alpha_vanilla) + "_beta_" + str(opt.beta_vanilla) + "_alfa_" + str(opt.alfa)
    elif opt.augmentation_method == "mixup_vanilla_features":
        opt.model_name += "_mixup_vanilla_features_" + opt.vanilla_method + "_alpha_" + str(opt.alpha_vanilla) + "_beta_" + str(opt.beta_vanilla) + "_alfa_" + str(opt.alfa)
    elif opt.augmentation_method == "vanilia":
        opt.model_name += "_vanilia_"

    if opt.method == "SupCon":
        opt.model_name += "_SupCon_"
    elif opt.method == "SimCLR":
        opt.model_name += "_SimCLR_" + str(opt.method_lam) + "_"
    elif opt.method == "SimCLR_CE":
        opt.model_name += "_SimCLR_CE_" + str(opt.method_lam) + "_"

    opt.model_name += 'trail_{}'.format(opt.trail) + "_" + str(opt.feat_dim) + "_" + str(opt.batch_size)

    # warm-up for large-batch training,
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.model_name = '{}_warm'.format(opt.model_name)
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate

    opt.save_folder = os.path.join(opt.model_path, opt.model_name)
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    
    opt.num_classes = num_inlier_classes_mapping[opt.datasets]

    return opt


def set_loader(opt):
    # construct data loader
    
    if opt.upsample is True and opt.last_feature_path is not None:
        
        last_features_list = []
        last_feature_labels_list = []
        
        with open(opt.last_feature_path, "rb") as f:
            last_features, _, _, last_feature_labels = pickle.load(f)
            last_features_list.append(last_features)
            last_feature_labels_list.append(last_feature_labels)

        last_model = load_model(opt)
        train_dataset =  get_train_datasets(opt, last_features_list=last_features_list, last_feature_labels_list=last_feature_labels_list, last_model=last_model)
    else:
        train_dataset =  get_train_datasets(opt)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler, drop_last=True)

    return train_loader



def set_model(opt):

    if opt.datasets == "mnist":
        in_channels = 1
    else:
        in_channels = 3

    if opt.model == "resnet18" or opt.model == "resnet34":
        model = SupConResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "preactresnet18" or opt.model == "preactresnet34":
        model = SupConpPreactResNet(name=opt.model, feat_dim=opt.feat_dim, in_channels=in_channels)
    elif opt.model == "MLP":
        model = SupConMLP(feat_dim=opt.feat_dim)
    else:
        model = simCNN_contrastive(opt, feature_dim=opt.feat_dim, in_channels=in_channels)

    #if opt.last_model_path is not None:
    #    load_model(opt, model)
    
    if opt.mixup_vanilla or opt.mixup_vanilla_features:
        criterion1 = SupConLoss(temperature=opt.temp)
        if opt.method_lam == "SupCon":
            criterion2 = SupConLoss_mix(temperature=opt.temp)
        else:
            criterion2 = SupConLoss(temperature=opt.temp)
    else:
        criterion1 = SupConLoss(temperature=opt.temp)
        criterion2 = SupConLoss(temperature=opt.temp)

    if opt.method == "SimCLR_CE":
        criterion1 = SupConLoss(temperature=opt.temp)
        criterion2 = torch.nn.CrossEntropyLoss()
        linear = LinearClassifier(name=opt.model, num_classes=opt.num_classes)
    elif opt.method == "MoCo":
        # TODO loss function for MoCo
    else:
        linear = None

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        if linear is not None:
            linear = linear.cuda()
        criterion1 = criterion1.cuda()
        if criterion2 is not None:
            criterion2 = criterion2.cuda()
        cudnn.benchmark = True

    return model, linear, criterion1, criterion2



def load_model(opt, model=None):
    if model is None:
        model = SupConResNet(name=opt.model)

    ckpt = torch.load(opt.last_model_path, map_location='cpu')
    state_dict = ckpt['model']

    new_state_dict = {}
    for k, v in state_dict.items():
        k = k.replace("module.", "")
        new_state_dict[k] = v

    state_dict = new_state_dict
    model.load_state_dict(state_dict)
    model.cuda()
    model.eval()

    return model



def train(train_loader, model, linear, criterion1, criterion2, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    losses_ssl = AverageMeter()
    losses_sup = AverageMeter()
    losses1 = AverageMeter()
    losses2 = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):

        data_time.update(time.time() - end)
        images1 = images[0]
        images2 = images[1]
        
        images = torch.cat([images1, images2], dim=0)
        labels_convert = label_convert(labels, num_classes=opt.num_classes)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            images1 = images1.cuda(non_blocking=True)
            images2 = images2.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
            labels_convert  = labels_convert.cuda(non_blocking=True)

        if opt.mixup_vanilla:
            imgs_mix1, imgs_mix2, labels_mix_negative_convert, lam = vanilla_mixup(images1, images2, y=labels_convert, alpha=opt.alpha_vanilla, beta=opt.beta_vanilla, device="cuda", mode=opt.vanilla_method)   #mixup_negative(images, images, labels, labels, alpha=opt.alpha_negative)         
            images_mix = torch.cat([imgs_mix1, imgs_mix2], dim=0)
            if torch.cuda.is_available():
                images_mix = images_mix.cuda(non_blocking=True)
                labels_mix_negative_convert = labels_mix_negative_convert.cuda(non_blocking=True)
                labels_convert = labels_convert.cuda(non_blocking=True)
       
        bsz = labels.shape[0]

        # warm-up learning rate
        #warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        features1, features2 = torch.split(features, [bsz, bsz], dim=0)
        features = torch.cat([features1.unsqueeze(1), features2.unsqueeze(1)], dim=1)

        if linear is not None:
            logits = linear(model.encoder(images))
            labels_linear = torch.cat([labels, labels], dim=0)

        if opt.mixup_vanilla:
            features_mix_vanilla = model(images_mix)
            features1_mix_vanilla, features2_mix_vanilla = torch.split(features_mix_vanilla, [bsz, bsz], dim=0)
            features_mix_vanilla = torch.cat([features1_mix_vanilla.unsqueeze(1), features2_mix_vanilla.unsqueeze(1)], dim=1)
            if opt.method == "SimCLR" or opt.method == "SimCLR_CE":
                features_mix_vanilla_mix_ori = torch.cat([features1_mix_vanilla.unsqueeze(1), features2.unsqueeze(1)], dim=1)
                features_mix_vanilla_ori_mix = torch.cat([features2_mix_vanilla.unsqueeze(1), features1.unsqueeze(1)], dim=1)      # maybe use one extra batch?  MOCO

        if opt.mixup_vanilla_features:
            features1_mix_vanilla, features2_mix_vanilla, labels_mix_negative_convert, lam = vanilla_mixup(features1.unsqueeze(1), features2.unsqueeze(1), y=labels_convert, alpha=opt.alpha_vanilla, beta=opt.beta_vanilla, device="cuda")
            features_mix_vanilla = torch.cat([features1_mix_vanilla, features2_mix_vanilla], dim=1)

        if opt.mixup_negative:
            f1_negative, f2_negative = mixup_negative_features(features1, features2, labels, labels, opt.alpha_negative, method=opt.negative_method)
            features_negative = torch.cat([f1_negative.unsqueeze(1), f2_negative.unsqueeze(1)], dim=1)
        else:
            features_negative = None

        if opt.mixup_positive:
            f1_positive, f2_positive = mixup_positive_features(features1, features2, labels, labels, opt.alpha_positive, p=opt.positive_p, num_classes=opt.num_classes, method=opt.positive_method)
            features_positive = torch.cat([f1_positive.unsqueeze(1), f2_positive.unsqueeze(1)], dim=1)
            features = torch.cat([features, features_positive], dim=0)
            labels = torch.cat([labels, labels], dim=0)
        else:
            features_positive = None

        if opt.mixup_hybrid:
            if opt.intra_inter_mix_hybrid:
                f1_hybrid, f2_hybrid = mixup_hybrid_features([features1, features2], labels, alpha=opt.alpha_hybrid, beta=opt.beta_hybrid, method=opt.hybrid_method)
                features_hybrid = torch.cat([f1_hybrid.unsqueeze(1), f2_hybrid.unsqueeze(1)], dim=1)
        else:
            features_hybrid = None

        if opt.method == 'SupCon':
            loss1 = criterion1(features, labels)
            if opt.mixup_vanilla or opt.mixup_vanilla_features:
                features_mix_vanilla = torch.cat([features, features_mix_vanilla], dim=0)
                labels_mix_negative_convert = torch.cat([labels_convert, labels_mix_negative_convert], dim=0)
                loss = loss1 = loss2 = criterion2(features_mix_vanilla, labels_mix_negative_convert)
                #loss2 = criterion2(features_mix, labels_mix_negative_convert)    
                #loss = loss1 + opt.alfa * loss2
            elif opt.mixup_negative:
                loss2 = criterion1(features, labels, features_negative=features_negative)        
                loss = loss1 + opt.alfa * loss2
            elif opt.mixup_positive:
                loss2 = criterion1(features, labels, features_positive=features_positive)        
                loss = loss1 + opt.alfa * loss2
            elif opt.augmentation_method == "vanilia":
                loss = loss2 = loss1
                

        elif opt.method == 'SimCLR':

            if opt.mixup_vanilla or opt.mixup_vanilla_features:
                features_mix_vanilla = torch.cat([features, features_mix_vanilla], dim=0)
                labels_mix_negative_convert = torch.cat([labels_convert, labels_mix_negative_convert], dim=0)
                loss_sup = loss1 = criterion2(features, labels)                    #criterion2(features_mix_vanilla, labels_mix_negative_convert)
                loss_ssl0 = criterion1(features)
                loss_ssl1 = criterion1(features_mix_vanilla_mix_ori)
                loss_ssl2 = criterion1(features_mix_vanilla_ori_mix)
                loss_ssl = loss2 = loss_ssl0 + (1-lam) * loss_ssl1 + lam * loss_ssl2  #lam * loss_ssl1 + (1-lam) * loss_ssl2
            else:
                loss_sup = loss1 = criterion2(features, labels)              # TODO pay attention to criterion1 and criterion2 here
                loss_ssl = loss2 = criterion1(features)

            losses_ssl.update(loss_ssl.item())
            losses_sup.update(loss_sup.item())
            loss = loss_sup + opt.method_lam * loss_ssl

        elif opt.method == 'SimCLR_CE':

            if opt.mixup_vanilla or opt.mixup_vanilla_features:
                features_mix_vanilla = torch.cat([features, features_mix_vanilla], dim=0)
                labels_mix_negative_convert = torch.cat([labels_convert, labels_mix_negative_convert], dim=0)
                loss_ce = loss1 = criterion2(logits, labels_linear)                     #criterion2(features_mix_vanilla, labels_mix_negative_convert)
                loss_ssl0 = criterion1(features)
                loss_ssl1 = criterion1(features_mix_vanilla_mix_ori)
                loss_ssl2 = criterion1(features_mix_vanilla_ori_mix)
                loss_ssl = loss2 = loss_ssl0 + lam * loss_ssl1 + (1-lam) * loss_ssl2
            else:
                loss_ce = loss1 = criterion2(logits, labels_linear)
                loss_ssl = loss2 = criterion1(features)

            losses_ssl.update(loss_ssl.item())
            losses_sup.update(loss_ce.item())
            loss = loss_ce + opt.method_lam * loss_ssl

        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)
        losses1.update(loss1.item(), bsz)
        losses2.update(loss2.item(), bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        #plot_grad_flow(model.named_parameters(), idx, epoch)
        if opt.clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), opt.clip)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})\t'
                  "loss1 {loss1.val:.3f} ({loss1.avg:.3f})\t"
                  "loss2 {loss2.val:.3f} ({loss2.avg:.3f})\t".format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, loss1=losses1, loss2=losses2))
            if opt.method == "SimCLR":
                print("loss_ssl {loss_ssl.val:.3f} ({loss_ssl.avg:.3f})\t"
                      "loss_sup {loss_sup.val:.3f} ({loss_sup.avg:.3f})\t".format(
                        loss_ssl=losses_ssl, loss_sup=losses_sup))
            if opt.method == "SimCLR_CE":
                print("loss_ssl {loss_ssl.val:.3f} ({loss_ssl.avg:.3f})\t"
                      "loss_sup {loss_ce.val:.3f} ({loss_ce.avg:.3f})\t".format(
                        loss_ssl=losses_ssl, loss_ce=losses_sup))
            sys.stdout.flush()

    return (losses.avg, losses1.avg, losses2.avg)


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)
    print("train_loader, ", train_loader.__len__())

    # build model and criterion
    model, linear, criterion1, criterion2 = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    losses = []

    #save_file = os.path.join(opt.save_folder, 'first.pth')
    #save_model(model, optimizer, opt, opt.epochs, save_file)

    # training routine
    for epoch in range(0, opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)

        # train for one epoch
        time1 = time.time()
        loss = train(train_loader, model, linear, criterion1, criterion2, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        losses.append(loss)
        if epoch % opt.save_freq == 0:
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, linear, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, linear, optimizer, opt, opt.epochs, save_file)
    with open(os.path.join(opt.save_folder, "loss_" + str(opt.trail)), "wb") as f:
         pickle.dump(losses, f)


if __name__ == '__main__':
    main()
