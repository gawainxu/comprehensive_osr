import pickle
import numpy as np
import matplotlib.pyplot as plt
from util import AUROC

def KNN_logits_global(testing_features, examplar_features, l):

    similarities_global = []
    for idx, testing_feature in enumerate(testing_features):

        similarities = np.matmul(examplar_features, testing_feature) #/ np.linalg.norm(examplar_features.astype(float), axis=1) / np.linalg.norm(testing_feature.astype(float))
        ind = np.argsort(similarities)[:1]
        top_k_similarities = similarities[ind]
        print(l, idx, np.mean(top_k_similarities))
        similarities_global.append(np.sum(top_k_similarities))

    return np.array(similarities_global)



with open("D://projects//comprehensive_OSR//features//cifar10//cifar10_resnet34_mixup_min_similarity_no_p_0.1_alfa_2.0_single_0.05_SupCon_trail_0_test_unknown", "rb") as f:
    features_testing_unknown, features_testing_unknown_encoding, _, labels_testing_unknown = pickle.load(f) 

with open("D://projects//comprehensive_OSR//features//cifar10//cifar10_resnet34_mixup_min_similarity_no_p_0.1_alfa_2.0_single_0.05_SupCon_trail_0_train", "rb") as f:
    features_train, features_train_encoding, _, labels_train = pickle.load(f) 


#similarities_unknown = KNN_logits_global(features_testing_unknown[::40, :], features_train, l="testing_unknown")
#similarities_train = KNN_logits_global(features_train[::40, :], features_train, l="train")


norm_train = np.linalg.norm(features_train_encoding.astype(float), axis=1)
norm_unknown = np.linalg.norm(features_testing_unknown_encoding.astype(float), axis=1)

print(np.mean(norm_train.astype(float)), np.mean(norm_unknown.astype(float)))

plt.hist(norm_train[:2000], label="Train") 
plt.hist(norm_unknown[:2000], label="unknown")
plt.legend()
plt.savefig("D://projects//comprehensive_OSR//plots//hist.pdf")

labels_binary_known = [1 for i in range(2000)]
labels_binary_unknown = [0 for i in range(2000)]
labels_binary = np.array(labels_binary_known + labels_binary_unknown)
scores = np.concatenate((norm_train[:2000], norm_unknown[:2000]))


AUROC(labels_binary, scores, opt=None)