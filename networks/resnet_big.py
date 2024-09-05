"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
#from torchvision.models import resnet18, resnet34, resnet50, resnet101             #


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channels=3, zero_init_residual=False, k=1):
        super(ResNet, self).__init__()
        self.in_planes = 64*k

        self.conv1 = nn.Conv2d(in_channels, int(64*k), kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64*k)
        self.layer1 = self._make_layer(block, int(64*k), num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, int(128*k), num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, int(256*k), num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, int(512*k), num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   # [stride, 1]
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def adaptive_pool(self, features, attn_from=None):
        # features and attn_from are paired feature maps, of same size
        if attn_from is None:
            attn_from = features
        assert features.size() == attn_from.size()
        N, C, H, W = features.size()
        assert (attn_from >= 0).float().sum() == N*C*H*W
        attention = torch.einsum('nchw,nc->nhw', [attn_from, nn.functional.adaptive_avg_pool2d(attn_from, (1, 1)).view(N, C)])
        attention = attention / attention.view(N, -1).sum(1).view(N, 1, 1).repeat(1, H, W)
        attention = attention.view(N, 1, H, W)
        # output size: N, C
        return (features * attention).view(N, C, -1).sum(2)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        #out = self.adaptive_pool(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)



model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    "simCNN": [None, 128],
    "MLP": [None, 128]
}


class remove_fc(nn.Module):
    def __init__(self, model):
        super(remove_fc, self).__init__()
        self.model = model

    def forward(self, x):
        x = self.model.conv1(x)
        x = self.model.bn1(x)
        x = self.model.relu(x)
        x = self.model.maxpool(x)

        x = self.model.layer1(x)
        x = self.model.layer2(x)
        x = self.model.layer3(x)
        x = self.model.layer4(x)
        
        x = self.model.avgpool(x)
        x = torch.flatten(x, 1)
        return x


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""
    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x


class SupConResNet(nn.Module):
    """backbone + projection head"""
    def __init__(self, name='resnet18', head='mlp', feat_dim=128, in_channels=3, wide_k=1):
        super(SupConResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        dim_in = dim_in * wide_k                        ###
        self.encoder = model_fun(in_channels=in_channels, k=wide_k)
        #self.encoder.load_state_dict(torch.load(pretrained))                 #
        #self.encoder = nn.Sequential(*list(self.encoder.children())[:-1])    #
        if head == 'linear':
            self.head = nn.Linear(dim_in, feat_dim)
        elif head == 'mlp':
            self.head = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))

    def forward(self, x):
        feat = self.encoder(x)
        #feat = torch.flatten(feat, 1)                                     #
        feat = F.normalize(self.head(feat), dim=1)
        return feat


class MoCoResNet(nn.Module):
    
    def __init__(self, opt, name="resnet", head="linear", feat_dim=128, in_channels=3):
        """
        K: size of the bank buffer
        """
        super(MoCoResNet, self).__init__()
        self.queue_size = opt.K
        self.momentum = opt.momentum_moco
        self.temp = opt.temp
        
        model_fun, dim_in = model_dict[name]
        self.encoder_q = model_fun(in_channels=in_channels)
        self.encoder_k = model_fun(in_channels=in_channels)

        for param_q, param_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            param_k.data.copy_(param_q.data)        # initialize
            param_k.requires_grad = False           # no update by gradient

        if head == "linear":
            self.head_q = nn.Linear(dim_in, feat_dim)
            self.head_k = nn.Linear(dim_in, feat_dim)
        elif head == "mlp":
            self.head_q = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
            self.head_k = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=True),
                nn.Linear(dim_in, feat_dim)
            )
        else:
            raise NotImplementedError("head not supported: {}".format(head))

        for param_q, param_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            param_k.data.copy_(param_q.data)        # initialize
            param_k.requires_grad = False           # no update by gradient

        # create the queue to store negative samples
        self.register_buffer("queue", torch.randn(self.queue_size, feat_dim))   # TODO attention the dimension here
        #self.queue = nn.functional.normalize(self.queue, dim=1)

        # create pointer to store current position in the queue when enqueue and dequeue
        self.register_buffer("queue_ptr", torch.zeros(1, dtype=torch.long))

    @torch.no_grad()
    def momentum_update(self):
        """
        update the key_encoder parameters through the momemtum updata
        
        key_parameters = momemtum * key_parameters + (1 - momentum) * query_parameters
        """

        # for each of the parameters in each encoder
        for p_q, p_k in zip(self.encoder_q.parameters(), self.encoder_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)
        
        for p_q, p_k in zip(self.head_q.parameters(), self.head_k.parameters()):
            p_k.data = p_k.data * self.momentum + p_q.detach().data * (1. - self.momentum)

    @torch.no_grad()
    def shuffled_idx(self, batch_size):

        # generate shuffled indexes
        shuffled_idxs = torch.randperm(batch_size).long().cuda()
        reverse_idxs = torch.zeros(batch_size).long().cuda()
        value = torch.arange(batch_size).long().cuda()
        reverse_idxs.index_copy_(0, shuffled_idxs, value)

        return shuffled_idxs, reverse_idxs

    @torch.no_grad()
    def update_queue(self, feat_k):

        batch_size = feat_k.size(0)
        ptr = int(self.queue_ptr)

        # replace the keys at ptr (dequeue and enqueue)
        self.queue[ptr:ptr+batch_size, :] = feat_k      # TODO attention the dimensions here !!!!

        # move pointer alone the end of current batch
        ptr = (ptr + batch_size) % self.queue_size

        # store queue pointer as register_buffer
        self.queue_ptr[0] = ptr


    def InfoNCE_logits(self, f_q, f_k):

        """
        compute the similarity logits between positive samples and
        positive to all negative in the memory
        """

        f_k = f_k.detach()

        # get queue from register_buffer
        f_mem = self.queue.clone().detach()

        # normalize the feature representation
        f_q = nn.functional.normalize(f_q, dim=1)
        f_k = nn.functional.normalize(f_k, dim=1)
        f_mem = nn.functional.normalize(f_mem, dim=1)

        # compute sim between positive views
        pos = torch.bmm(f_q.view(f_q.size(0), 1, -1), f_k.view(f_k.size(0), -1, 1)).squeeze(-1)     # bmm((bsz, 1, dim), (bsz, dim, 1)) = (bsz, 1, 1), sim between corresponding tensors => (bsz, 1)

        # compute sim between positive and all negative in memory
        neg = torch.mm(f_q, f_mem.transpose(1, 0))         # mm((bsz, dim), (dim, bsz)) = (bsz, bsz)

        logits = torch.cat((pos, neg), dim=1)              # (bsz, bsz+1), the first is the sim between f_q and f_k, the rest bsz are the sims between f_q and f_mem in the queue
        logits /= self.temp

        # create labels, first logit is posive and the rest are negative
        labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()

        return logits, labels


    def forward(self, x_q, x_k=None, mode="simclr"):

        batch_size = x_q.size(0)

        # feature of the query view from the query enoder
        feat_q = self.head_q(self.encoder_q(x_q))

        if mode == "moco":
            # get shuffled and reversed indexes for the current minibatch
            shuffled_idxs, reverse_idxs = self.shuffled_idx(batch_size)
            with torch.no_grad():
                # update the key encoder
                self.momentum_update()                    
                # shuffle minibatch
                x_k = x_k[shuffled_idxs]
                # feature representations of the shuffled key view from the key encoder
                feat_k = self.head_k(self.encoder_k(x_k))
                # reverse the shuffled samples to original position
                feat_k = feat_k[reverse_idxs]
                
                
            # compute the logits for the InfoNCE contrastive loss
            logit, labels = self.InfoNCE_logits(feat_q, feat_k)
            # updata the queue/memory with the curent key_encoder minibatch
            self.update_queue(feat_k)            

            return logit, labels

        elif mode == "simclr":
            
            assert x_k == None
            feat_q = nn.functional.normalize(feat_q, dim=1)
            
            return feat_q


class SupCEResNet(nn.Module):
    """encoder + classifier"""
    def __init__(self, name='resnet18', num_classes=10):
        super(SupCEResNet, self).__init__()
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, x):
        return self.fc(self.encoder(x))


class LinearClassifier(nn.Module):
    """Linear classifier"""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()
        _, feat_dim = model_dict[name]
        self.fc = nn.Linear(feat_dim, num_classes)

    def forward(self, features):
        return self.fc(features)
