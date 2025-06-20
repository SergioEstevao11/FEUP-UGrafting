import math
import torch
import torch.optim as optim
import numpy as np
from ood_metrics import calc_metrics
from sklearn.metrics import roc_curve
import random
from PIL import ImageFilter, ImageOps

class Solarization(object):
    """
    Apply Solarization to the PIL image.
    """
    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img
        

class GaussianBlur(object):
    """
    Apply Gaussian Blur to the PIL image.
    """
    def __init__(self, p=0.5, radius_min=0.1, radius_max=2.):
        self.prob = p
        self.radius_min = radius_min
        self.radius_max = radius_max

    def __call__(self, img):
        do_it = random.random() <= self.prob
        if not do_it:
            return img

        return img.filter(
            ImageFilter.GaussianBlur(
                radius=random.uniform(self.radius_min, self.radius_max)
            )
        )

class TwoCropTransform:
    """Create two crops of the same image"""

    def __init__(self, transform):
        self.transform = transform

    def __call__(self, x):
        return self.transform(x), self.transform(x)


class AverageMeter(object):
    """Computes and stores the average and current value"""

    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def adjust_learning_rate(args, optimizer, epoch):
    lr = args.learning_rate
    if args.cosine:
        eta_min = lr * (args.lr_decay_rate ** 3)
        lr = eta_min + (lr - eta_min) * (
                1 + math.cos(math.pi * epoch / args.epochs)) / 2
    else:
        steps = np.sum(epoch > np.asarray(args.lr_decay_epochs))
        if steps > 0:
            lr = lr * (args.lr_decay_rate ** steps)

    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


def warmup_learning_rate(args, epoch, batch_id, total_batches, optimizer):
    if args.warm and epoch <= args.warm_epochs:
        p = (batch_id + (epoch - 1) * total_batches) / \
            (args.warm_epochs * total_batches)
        lr = args.warmup_from + p * (args.warmup_to - args.warmup_from)

        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def set_optimizer(opt, model):
    optimizer = optim.SGD(model.parameters(),
                          lr=opt.learning_rate,
                          momentum=opt.momentum,
                          weight_decay=opt.weight_decay)
    return optimizer


def calc_metrics_transformed(ind_score: np.ndarray, ood_score: np.ndarray) -> dict:
    labels = [1] * len(ind_score) + [0] * len(ood_score)
    scores = np.hstack([ind_score, ood_score])

    metric_dict = calc_metrics(scores, labels)
    fpr, tpr, _ = roc_curve(labels, scores)

    metric_dict_transformed = {
        "AUROC": 100 * metric_dict["auroc"],
        #    "TNR at TPR 95%": 100 * (1 - metric_dict["fpr_at_95_tpr"]),
        #   "Detection Acc.": 100 * 0.5 * (tpr + 1 - fpr).max(),
    }
    return metric_dict_transformed


import numpy as np

def thresholding_mechanism(predictions, variances, method='top_percent'):
    """
    Apply a thresholding mechanism to filter out predictions based on variance.
    
    Parameters:
    - variances: np.array, shape (10000, 128), the variances for each prediction
    - predictions: np.array, shape (10000, 10), the prediction probabilities for each class
    - method: str, 'average' or 'top_10_percent', the method to determine the threshold
    
    Returns:
    - filtered_predictions: np.array, filtered predictions based on the chosen threshold
    - accepted_indices: np.array, indices of the accepted predictions
    """



    


    if method == 'average':
        threshold = np.mean(variances)
    elif method == 'top_percent':
        threshold = np.percentile(variances, 100)
    else:
        raise ValueError("Method should be either 'average' or 'top_10_percent'")
    
    # Calculate the mean variance for each prediction
    mean_variances = np.mean(variances, axis=1)
    
    # Get the indices of the predictions that are below the threshold
    accepted_indices = np.where(mean_variances <= threshold)[0]
    rejected_indices = np.where(mean_variances > threshold)[0]
    
    
    return accepted_indices, rejected_indices
