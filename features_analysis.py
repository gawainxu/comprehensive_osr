import pickle
import argparse
import numpy as np
from util import CKA, linear_CKA


def parse_option():

    parser = argparse.ArgumentParser('argument for feature analysis')
    
    parser.add_argument("--feature_path", type=str, default="./features1/cifar10_resnet18_temp_0.05_tau_strategy_fixed_id_0_lr_0.001_bz_256_epoch_")
    parser.add_argument("--epoch_start", type=int, default=0)
    parser.add_argument("--epoch_end", type=int, default=1)
    parser.add_argument("--epoch_step", type=int, default=1)
    parser.add_argument("--output_file", type=str, default="./ss1")

    opt = parser.parse_args()

    return opt


def main():
    
    opt = parse_option()
    ss = []

    for epoch in range(opt.epoch_start, opt.epoch_end, opt.epoch_step):
        
        epoch1 = epoch
        epoch2 = epoch + opt.epoch_step
        feature_path1 = opt.feature_path + str(epoch1)
        feature_path2 = opt.feature_path + str(epoch2)

        with open(feature_path1, "rb") as f:
            features1, _, _, labels1 = pickle.load(f)

        with open(feature_path2, "rb") as f:
            features2, _, _, labels2 = pickle.load(f)

        features1 = np.squeeze(features1)
        features2 = np.squeeze(features2)
        #features1 = features1[:-1:10]
        #features2 = features2[:-1:10]

        s = linear_CKA(features1, features2)
        print("epoch: ", epoch, s)
        ss.append((epoch, s))

    with open(opt.output_file, "wb") as f:
        pickle.dump(ss, f)



if __name__ == '__main__':
    main()
