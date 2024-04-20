import torch 
import pickle
import numpy as np
from sklearn import linear_model
from scipy.spatial.distance import cdist, squareform
from scipy.stats import pearsonr



def id_2nn(distances, fraction=0.9, verbose=False):

   """
    Args:
        distances : 2-D Matrix (n,n) where n is the number of points
        fraction : fraction of the data considered for the dimensionality
        estimation (default : fraction = 0.9)

    Returns:            
        x : log(mu)    (*)
        y : -(1-F(mu)) (*)
        reg : linear regression y ~ x structure obtained with scipy.stats.linregress
        (reg.slope is the intrinsic dimension estimate)
        r : determination coefficient of y ~ x
        pval : p-value of y ~ x: 
   """

   # sort distance matrix
   distances_sorted = np.sort(distances, axis=1, kind="quicksort")
   
   # clean data, first two closest
   k1 = distances_sorted[:, 1]
   k2 = distances_sorted[:, 2] 

   #print("k1", np.min(k1))
   #print("k2", np.min(k2))

   zeros = np.where(k1 == 0)[0]
   if verbose:
        print('Found n. {} elements for which r1 = 0'.format(zeros.shape[0]))
        print(zeros)

   degeneracies = np.where(k1 == k2)[0]
   if verbose:
        print('Found n. {} elements for which r1 = r2'.format(degeneracies.shape[0]))
        print(degeneracies)

   good = np.setdiff1d(np.arange(distances_sorted.shape[0]), np.array(zeros))
   good = np.setdiff1d(good, np.array(degeneracies))

   if verbose:
        print('Fraction good points: {}'.format(good.shape[0]/distances_sorted.shape[0]))

   npoints = int(np.floor(good.shape[0] * fraction)) 

   # define mu and Femp
   N = good.shape[0]
   mu = np.sort(np.divide(k2, k1), axis=None, kind="quicksort")
   Femd = (np.arange(1, N+1, dtype=np.float64)) / N
   
   # take logs (leave out the last element because 1-Femp is zero here)
   x = np.log(mu[:-2])
   y = -np.log(1-Femd[:-2])

   # regression
   regr = linear_model.LinearRegression(fit_intercept=False)
   regr.fit(x[0:npoints, np.newaxis], y[0:npoints, np.newaxis])
   r, pval = pearsonr(x[0:npoints], y[0:npoints])

   return x, y, regr.coef_[0][0], r, pval


def euclidean_distance(data):

    distances = cdist(data, data, 'euclidean')
    return distances



if __name__ == "__main__":
    
    feature_path = "D://projects//SupCon_Analysis//features//cifar10_resnet18_method_ce_temp_0.05_K_1_id_4_lr_0.001_bz_64_epoch_200_train"
    with open(feature_path, "rb") as f:
        feature_maps, feature_maps_backbone, label = pickle.load(f)
    npoints = feature_maps.shape[0]

    if len(feature_maps.shape) > 2:
        feature_maps = np.reshape(feature_maps, (npoints, -1))
    
    if len(feature_maps_backbone.shape) > 2:
        feature_maps_backbone = np.reshape(feature_maps_backbone, (npoints, -1))

    # TODO sort features
    for c in range(10):
        indices = [i for i in range(c*1001, (c+1)*1001)]
        feature_maps_c = feature_maps[indices]
        #feature_maps_backbone_c = feature_maps_backbone[indices]

        print(feature_maps_c.shape)
        distances = euclidean_distance(feature_maps_c)
        _, _, d, r,pval = id_2nn(distances)
        print(d)

    

# TODO List
# ID for full wide models (SupCon, CE, class-wise)
# ID for wide models without head
# 
#
