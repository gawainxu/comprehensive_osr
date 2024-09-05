import numpy as np
from sklearn.metrics import precision_recall_curve, auc


def detection_accuracy(id_scores, od_scores):
    
    # for DTACC
    min_id_score = np.min(id_scores)
    mean_od_scores = np.mean(od_scores)
    best_acc = 0

    # for TNR, assuming od are positives
    tp = 0
    fp = 0
    tn = 0
    fn = 0

    tpr_error = 1000

    for i in np.linspace(min_id_score-2*mean_od_scores, min_id_score+2*mean_od_scores, num=500):
        tp = np.where(od_scores<=i)[0].size
        tn += np.where(id_scores>=i)[0].size
        correct = tp + tn

        fn = np.where(od_scores>i)[0].size
        fp += np.where(id_scores<i)[0].size
        wrong = fn + fp

        acc = correct * 1.0 / (correct + wrong)

        if acc > best_acc:
            best_acc = acc

        tpr = tp * 1.0 / (fp + fn)
        if np.abs(tpr - 0.95) < tpr_error:
            tpr_error = np.abs(tpr - 0.95)
            tnr = tn * 1.0 / (tp + tn)

    return best_acc, tnr


def au_in_out(labels, probs):

    precision, recall, _ = precision_recall_curve(labels, probs)
    au_in = auc(recall, precision)

    # au_out can be problematic
    labels_reverse = 1 - labels
    probs_reverse = 1 - probs
    precision_reverse, recall_reverse, _ = precision_recall_curve(labels_reverse, probs_reverse)
    au_out = auc(recall_reverse, precision_reverse)

    return au_in, au_out







