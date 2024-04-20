"""
Author: Yonglong Tian (yonglong@mit.edu)
Date: May 07, 2020
"""
from __future__ import print_function

import torch
import torch.nn as nn


class SupConLoss_adv(nn.Module):
    """Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""
    def __init__(self, temperature=0.07,  contrast_mode='all',
                 base_temperature=0.07, old_classes=range(10), positive_lam=1.0):
        super(SupConLoss_adv, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.old_classes = old_classes
        self.positive_lam = positive_lam


    def contrast_adv(self, anchor_features, mask, epi):

        l = len(anchor_features)
        for i, anchor_feature in enumerate(anchor_features):
            anchor_feature = anchor_feature.repeat(l, 1)
            m = mask[i].repeat(l, 1)
            contrast_features = anchor_feature - epi * torch.matmul(anchor_features, m)
            

        """
        similarities = []
        contrast_features = anchor_features
        for i, (anchor_feature, label) in enumerate(zip(anchor_features, labels)):
            indices = torch.where(labels==label)
            contrast_features_i = contrast_features
            for ind in indices:
                contrast_features_i[ind] = contrast_features_i[ind] - epi * anchor_feature
            similarity = torch.div(torch.matmul(anchor_feature, contrast_features_i.T), self.temperature)
            similarities.append(similarity)

        similarities = torch.cat(similarities)
        return similarities
        """

    def forward(self, features, labels=None, mask=None, reduction='mean'):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """
        device = (torch.device('cuda') if features.is_cuda else torch.device('cpu'))

        if len(features.shape) < 3:
            raise ValueError('`features` needs to be [bsz, n_views, ...],'
                             'at least 3 dimensions are required')
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and mask is not None:
            raise ValueError('Cannot define both `labels` and `mask`')
        elif labels is None and mask is None:
            mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError('Num of labels does not match num of features')
            mask = torch.eq(labels, labels.T).float().to(device)
        else:
            mask = mask.float().to(device)
            
        # Remove the features of old classes 
        #if_new = []
        #for i, l in enumerate(labels):
        #    if l.item() in self.old_classes:
        #        if_new.append(0)
        #    else:
        #        if_new.append(1)
                
        #if_new = torch.tensor(if_new)
        #if_new = torch.unsqueeze(if_new, dim=0)
        #if_new = if_new.repeat(batch_size, 1)
        #if_new_mask = torch.logical_or(if_new.T, if_new)
        #if_new_mask = if_new_mask.float().to(device)
                
        contrast_count = features.shape[1]
        bsz = features.shape[0]
        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)

        if self.contrast_mode == 'one':
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == 'all':
            anchor_feature = contrast_feature
            anchor_count = contrast_count
            #if features_negative is not None:
            #    anchor_feature_negative = contrast_feature_negative
        else:
            raise ValueError('Unknown mode: {}'.format(self.contrast_mode))
        mask = mask.repeat(anchor_count, contrast_count)

        # compute logits
        anchor_dot_contrast = self.contrast_adv(anchor_feature, labels, epi=0.1)    

        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        #logits = (self.temperature / self.base_temperature) * logits

        
        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0
        )
        #if_new_mask = if_new_mask.repeat(anchor_count, contrast_count)

        mask = mask * logits_mask        #* if_new_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
    
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / (mask.sum(1))
        #mean_log_prob_pos[mean_log_prob_pos != mean_log_prob_pos] = 0.1
        #print("mean_log_prob_pos", torch.sum(torch.abs(mean_log_prob_pos)))

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos              # purpose of base_temperture unknown
        #print("loss", torch.sum(torch.abs(loss)))
        loss = loss.view(anchor_count, batch_size).mean()
        #print("supcon loss mean", torch.sum(torch.abs(loss)))
        

        return loss
