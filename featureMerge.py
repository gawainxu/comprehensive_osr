#!/usr/bin/env python2
# -*- coding: utf-8 -*-

import json
import pickle
import numpy as np


def featureMerge(featureList, opt):
    
    featureMaps = []
    featureMaps_backbone = []
    featureMaps_linear = []
    labels = []
    print(opt.save_path_all)

    for featurePath in featureList:
        
        with open(featurePath, "rb") as f:
            features, feature_backbone, feature_linear, labels_part = pickle.load(f)
  
        if len(feature_linear) > 0:
            featureMaps_linear = featureMaps_linear + feature_linear

        featureMaps_backbone = featureMaps_backbone + feature_backbone
        featureMaps = featureMaps + features
        labels = labels + labels_part
        
    featureMaps_backbone = np.array(featureMaps_backbone, dtype=object)
    featureMaps = np.array(featureMaps, dtype=object)
    #print("featureMaps_backbone", featureMaps_backbone.shape)
    featureMaps_backbone = np.squeeze(featureMaps_backbone)
    featureMaps = np.squeeze(featureMaps)
    featureMaps_linear = np.squeeze(np.array(featureMaps_linear))
    labels = np.squeeze(np.array(labels))
    
    with open(opt.save_path_all, 'wb') as f:
        pickle.dump((featureMaps, featureMaps_backbone, featureMaps_linear, labels), f)
        
