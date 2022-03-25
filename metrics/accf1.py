import numpy as np
from sklearn.metrics import f1_score, accuracy_score


def acc_f1_score(y_true, y_pred, ignore_index=None, normalize=False, average='macro', **kwargs):
    """Multi-class f1 score and accuracy"""
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    if ignore_index is not None:
        leave = y_true != ignore_index
    else:
        leave = np.ones_like(y_true)
    y_true = y_true[leave]
    y_pred = y_pred[leave]
    f1 = f1_score(y_true=y_true, y_pred=y_pred, average=average, **kwargs)
    acc = accuracy_score(y_true=y_true, y_pred=y_pred, normalize=normalize)
    return acc, f1


class AccF1Metric(object):
    def __init__(self, ignore_index, average='macro'):
        self.ignore_index = ignore_index
        self.average = average
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def clear(self):
        self.y_true = []
        self.y_pred = []

    def get(self):
        y_true = np.stack(self.y_true, axis=0).reshape(-1)
        y_pred = np.stack(self.y_pred, axis=0).reshape(-1)
        acc, f1 = acc_f1_score(y_true=y_true, y_pred=y_pred,
                               average=self.average,
                               normalize=True,
                               ignore_index=self.ignore_index)
        return acc, f1


class MultiLabelAccF1(object):
    def __init__(self, ignore_index=None, average='binary'):
        self.ignore_index = ignore_index
        self.average = average
        self.y_pred = []
        self.y_true = []

    def update(self, y_pred, y_true):
        self.y_pred.append(y_pred)
        self.y_true.append(y_true)

    def clear(self):
        self.y_true = []
        self.y_pred = []

    def get(self):
        y_true = np.vstack(self.y_true)
        y_pred = np.vstack(self.y_pred)
        total_num = y_pred.shape[0] * y_pred.shape[1]
        labeled_idx = y_true != self.ignore_index
        labeled_num = np.sum(labeled_idx)
        acc = 0
        f1 = []
        for i in range(y_pred.shape[1]):
            acc_i, f1_i = acc_f1_score(y_true=y_true[:, i], y_pred=y_pred[:, i],
                                       average=self.average,
                                       normalize=False,
                                       ignore_index=self.ignore_index)
            acc += acc_i
            f1.append(f1_i)
        acc = acc / labeled_num
        f1 = np.mean(f1)
        return acc, f1
