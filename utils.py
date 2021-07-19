from torchvision import transforms as tfs
import torch
import torch.nn as nn
import torch.nn.functional as F
import itertools
import numpy as np
import matplotlib.pyplot as plt
from scene_classification.config import *
import os


class LabelSmoothingCrossEntropy(nn.Module):
    def __init__(self, eps=0.1, reduction='mean'):
        super(LabelSmoothingCrossEntropy, self).__init__()
        self.eps = eps
        self.reduction = reduction

    def forward(self, output, target):
        c = output.size()[-1]
        log_preds = F.log_softmax(output, dim=-1)
        if self.reduction == 'sum':
            loss = -log_preds.sum()
        else:
            loss = -log_preds.sum(dim=-1)
            if self.reduction == 'mean':
                loss = loss.mean()
        return loss*self.eps/c + (1-self.eps) * F.nll_loss(log_preds, target, reduction=self.reduction)


# 训练集数据预处理 
def train_tf(x):
    # x=x.resize((RESIZE, RESIZE))
    x = x.resize((RESIZE, RESIZE))
    x = x.convert('RGB')
    im_aug = tfs.Compose([
        tfs.RandomCrop(CROP),
        tfs.RandomHorizontalFlip(),  # default 0.5
        tfs.ToTensor(),
        tfs.Normalize(MEAN, STD)
    ])
    x = im_aug(x)

    return x


# 测试集数据预处理
def test_tf(x):
    x=x.resize((RESIZE, RESIZE))
    x=x.convert('RGB')
    im_aug = tfs.Compose([
        tfs.CenterCrop(CROP),
        tfs.ToTensor(),
        tfs.Normalize(MEAN, STD)
    ])
    x = im_aug(x)
    return x


def test_tf1(x):
    x = x.resize((RESIZE, RESIZE))
    x = x.convert('RGB')
    im_aug = tfs.Compose([
        tfs.CenterCrop(CROP),
        tfs.ToTensor(),
        tfs.Normalize(MEAN, STD)
    ])
    x = im_aug(x)
    return x


def get_acc(output, label):
    """Calculate accuracy.
    Parameters:
        output         -- prediction probability(str)
        labels         -- ground truth label(tensor list)
    *****************************************************
    Modification Date:2021-05-28  By: Yunhao Cheng
    """
    _, pred_label = output.max(1) 
    num_correct = (pred_label == label).sum().item()
    return num_correct


def confusion_matrix(preds, labels, conf_matrix):
    """Statistical confusion matrix information.
    Parameters:
        preds          -- prediction label(str)
        labels         -- ground truth label(str)
        conf_matrix    -- confusion matrix(list)
    ************************************************
    Modification Date:2021-05-28  By: Yunhao Cheng
    """
    for p, t in zip(preds, labels):
        conf_matrix[t, p] += 1
    return conf_matrix


def plot_confusion_matrix(cm, classes, save_path, normalize=False, title='Confusion matrix', cmap=plt.cm.Blues):
    """Draw confusion matrix.
    Parameters:
        cm          -- confusion matrix(list)
        classes     -- dataset label(list)
        save_path   -- save path(str)
    ************************************************
    Modification Date:2021-05-28  By: Yunhao Cheng
    """
    plt.rc('font', family='Times New Roman', size='10')
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=60)
    plt.yticks(tick_marks, classes)
    # plt.tick_params(labelsize=5)

    fmt = '.2f' if normalize else '.0f'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt), ha="center", va="center", color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.tight_layout()
    plt.savefig(os.path.join(save_path, 'confusion_matrix.png'))