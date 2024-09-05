import os
import sys
import argparse
import time
import math
import pickle

import torch
import torch.backends.cudnn as cudnn
from torchvision import transforms, datasets

from util import TwoCropTransform, AverageMeter
from util import adjust_learning_rate, warmup_learning_rate
from util import set_optimizer, save_model
from util import tau_step, tau_step_cosine, tau_step_linear, tau_step_exp
from networks.resnet_big import SupConResNet
from losses import SupConLoss
from dataUtil import get_train_datasets
from config import temperature1_scheduling_mapping, temperature2_scheduling_mapping, temperature_scheduling_epoch_mapping

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
    parser.add_argument('--save_freq', type=int, default=1,
                        help='save frequency')
    parser.add_argument('--batch_size', type=int, default=256,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=16,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.001,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument("--pretrained", type=int, default=1)
    parser.add_argument("--tau_strategy", type=str, default="fixed", choices=["fixed", "fixed_set", "fixed_set_diff", "cosine", "linear", "exp"])
    parser.add_argument("--tau_max", type=float, default=0.5)   
    parser.add_argument("--tau_min", type=float, default=0.05)     
    parser.add_argument("--cosine_period", type=float, default=1.0)
    parser.add_argument("--init_model_path", type=str, default=None)

    # model dataset
    parser.add_argument('--model', type=str, default='resnet18', choices=["resnet18", "resnet34"])
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
                        choices=['SupCon', 'SimCLR'], help='choose method')
    parser.add_argument("--trail", type=int, default=0, choices=[0,1,2,3,4,5], help="index of repeating training")
    parser.add_argument("--action", type=str, default="training_supcon",
                        choices=["training_supcon", "trainging_linear", "testing_known", "testing_unknown", "feature_reading"])
    # temperature
    parser.add_argument('--temp', type=float, default=0.05, help='temperature for loss function')
    parser.add_argument('--temp_max', type=float, default=0.05, help='temperature for loss function max')
    parser.add_argument('--temp_min', type=float, default=0.05, help='temperature for loss function min')

    # other setting
    parser.add_argument('--cosine', type=bool, default=False,
                        help='using cosine annealing')
    parser.add_argument('--syncBN', action='store_true',
                        help='using synchronized batch normalization')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

     # downsample dataset
    parser.add_argument("--downsample", type=bool, default=False)
    parser.add_argument("--downsample_ratio", type=float, default=0.5)
    parser.add_argument("--downsample_ratio_center", type=float, default=0.05)
    parser.add_argument("--downsample_middle", type=int, default=3)
    parser.add_argument("--last_feature_path", type=str, default=None)
    parser.add_argument("--last_model_path", type=str, default=None)


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

    opt.model_name = '{}_{}_{}_lr_{}_decay_{}_bsz_{}_tempmax_{}_tempmin_{}_trail_{}_tau_{}_cosine_period_{}'.\
        format(opt.method, opt.datasets, opt.model, opt.learning_rate,
               opt.weight_decay, opt.batch_size, opt.temp_max, opt.temp_min, opt.trail, opt.tau_strategy, opt.cosine_period)


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

    if opt.tau_strategy == "fixed_set" or opt.tau_strategy == "fixed_set_diff":
        opt.tau_set1 = temperature1_scheduling_mapping[opt.datasets]
        opt.tau_set2 = temperature2_scheduling_mapping[opt.datasets]
        opt.tau_epochs = temperature_scheduling_epoch_mapping[opt.datasets]
    else:
        opt.tau_set = None
        opt.tau_epochs = None

    return opt


def set_loader(opt):
    # construct data loader
    
    train_dataset =  get_train_datasets(opt)

    print("train_dataset,", len(train_dataset))
    print("train_dataset,", train_dataset[0][0][0].shape)

    train_sampler = None
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=opt.batch_size, shuffle=(train_sampler is None),
                                               num_workers=opt.num_workers, pin_memory=True, sampler=train_sampler)

    return train_loader


def set_model(opt):
    model = SupConResNet(name=opt.model)
    if opt.init_model_path is not None:
        load_model(opt, model)
    criterion = SupConLoss(temperature=opt.temp)

    # enable synchronized Batch Normalization
    if opt.syncBN:
        model = apex.parallel.convert_syncbn_model(model)

    if torch.cuda.is_available():
        if torch.cuda.device_count() > 1:
            model.encoder = torch.nn.DataParallel(model.encoder)
        model = model.cuda()
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion


def load_model(opt, model=None):
    if model is None:
        model = SupConResNet(name=opt.model)

    ckpt = torch.load(opt.init_model_path, map_location='cpu')
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


def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    similarities = AverageMeter()
    dissimilarities = AverageMeter()

    end = time.time()
    for idx, (images, labels) in enumerate(train_loader):
        data_time.update(time.time() - end)

        images = torch.cat([images[0], images[1]], dim=0)

        if torch.cuda.is_available():
            images = images.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        #warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)

        # compute loss
        features = model(images)
        f1, f2 = torch.split(features, [bsz, bsz], dim=0)
        similarity = torch.matmul(f1, f2.T).detach().cpu()
        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels, labels.T).float().detach().cpu()
        unmask = 1 - mask
        dissimilarity = similarity * unmask
        similarity = similarity * mask
        similarity = similarity.sum() / mask.sum()
        dissimilarity = dissimilarity.sum() / unmask.sum()
        features = torch.cat([f1.unsqueeze(1), f2.unsqueeze(1)], dim=1)
        if opt.method == 'SupCon':
            loss = criterion(features, labels)
        elif opt.method == 'SimCLR':
            loss = criterion(features)
        else:
            raise ValueError('contrastive method not supported: {}'.
                             format(opt.method))

        # update metric
        losses.update(loss.item(), bsz)
        similarities.update(similarity, bsz)
        dissimilarities.update(dissimilarity, bsz)

        # SGD
        optimizer.zero_grad()
        loss.backward()
        #plot_grad_flow(model.named_parameters(), idx, epoch)
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print("Temperature: ", opt.temp)
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'similarity {similar.val:.3f} ({similar.avg:.3f})\t'
                  'dissimilarity {dissimilar.val:.3f} ({dissimilar.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                   epoch, idx + 1, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, similar=similarities, dissimilar=dissimilarities))
            sys.stdout.flush()

    return losses.avg, similarities.avg, dissimilarities.avg


def main():
    opt = parse_option()

    # build data loader
    train_loader = set_loader(opt)
    print("train_loader, ", train_loader.__len__())

    # build model and criterion
    model, criterion = set_model(opt)

    # build optimizer
    optimizer = set_optimizer(opt, model)

    losses = []
    ss = []
    dss = []

    save_file = os.path.join(opt.save_folder, 'first.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)

    # training routine
    for epoch in range(0, opt.epochs):
        adjust_learning_rate(opt, optimizer, epoch)
        if opt.tau_strategy == "fixed_set" or opt.tau_strategy == "fixed_set_diff":
            tau_step(opt, epoch)                                                          
            criterion = SupConLoss(temperature=opt.temp1, temperature1=opt.temp2)
        elif opt.tau_strategy == "cosine":
            tau_step_cosine(opt, epoch) 
            criterion = SupConLoss(temperature=opt.temp)
        elif opt.tau_strategy == "linear":
            tau_step_linear(opt, epoch) 
            criterion = SupConLoss(temperature=opt.temp)
        elif opt.tau_strategy == "exp":
            tau_step_exp(opt, epoch) 
            criterion = SupConLoss(temperature=opt.temp)
        else:
            criterion = SupConLoss(temperature=opt.temp)
        
        # train for one epoch
        time1 = time.time()
        loss, similarities, dissimilarities = train(train_loader, model, criterion, optimizer, epoch, opt)
        time2 = time.time()
        print('epoch {}, total time {:.2f}'.format(epoch, time2 - time1))

        losses.append(loss)
        ss.append(similarities)
        dss.append(dissimilarities)
        if epoch % opt.save_freq == 0:                         #  or epoch in opt.tau_epochs or epoch-1 in opt.tau_epochs
            save_file = os.path.join(
                opt.save_folder, 'ckpt_epoch_{epoch}.pth'.format(epoch=epoch))
            save_model(model, optimizer, opt, epoch, save_file)

    # save the last model
    save_file = os.path.join(
        opt.save_folder, 'last.pth')
    save_model(model, optimizer, opt, opt.epochs, save_file)
    with open(os.path.join(opt.save_folder, "loss_" + str(opt.trail)), "wb") as f:
         pickle.dump((losses, ss, dss), f)


if __name__ == '__main__':
    main()
