import sys
import time
import torch
from utils.util import AverageMeter, warmup_learning_rate
from models.ugraft import UGraft
from utils.losses import UALoss
import torch.backends.cudnn as cudnn
from torch import nn
from torch.cuda.amp import GradScaler, autocast

scaler = GradScaler()

def train(train_loader, model, criterion, optimizer, epoch, opt):
    """one epoch training"""
    model.train()
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    stdlosses = AverageMeter()
    stdlosses2 = AverageMeter()
    end = time.time()
    for idx, ((image1, image2), labels) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # images = torch.cat([image1, image2], dim=0)
        if torch.cuda.is_available():
            image1 = image1.cuda(non_blocking=True)
            image2 = image2.cuda(non_blocking=True)
            labels = labels.cuda(non_blocking=True)
        bsz = labels.shape[0]

        # warm-up learning rate
        warmup_learning_rate(opt, epoch, idx, len(train_loader), optimizer)
        # compute features and features std
        with autocast():
            features, features_std = model(image1, image2)
            # compute loss
            loss, std_loss, std_loss2 = criterion(features, features_std, epoch)
        # update metric
        losses.update(loss.item(), bsz)
        stdlosses.update(std_loss.item(), bsz)
        stdlosses2.update(std_loss2.item(), bsz)
        # SGD
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print info
        if (idx + 1) % opt.print_freq == 0:
            print('Train: [{0}][{1}/{2}]\t'
                  'BT {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'DT {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'loss {loss.val:.3f} ({loss.avg:.3f})'.format(
                epoch, idx + 1, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))
            sys.stdout.flush()

        

    return losses.avg, stdlosses.avg, stdlosses2.avg


def set_model(model_name, temperature, syncBN=False, lamda1=1, lamda2=0.1,
              batch_size=512, nh=5, image_shape=(3, 32, 32)):
    model = UGraft(name=model_name, n_heads=nh, image_shape=image_shape)
    criterion = UALoss(temperature=temperature, lamda1=lamda1, lamda2=lamda2, batch_size=batch_size)

    # enable synchronized Batch Normalization

    if torch.cuda.is_available():
        model = model.cuda()
        if syncBN:
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if torch.cuda.device_count() > 1:
            model = torch.nn.DataParallel(model)
        criterion = criterion.cuda()
        cudnn.benchmark = True

    return model, criterion

def evaluate_uncertainty(val_loader, model):
    model.eval()  # Set the model to evaluation mode
    test_features = []
    test_features_std = []
    test_labels = []

    with torch.no_grad():  # Disable gradient calculation
        for idx, ((image1, image2), labels) in enumerate(val_loader):
            if torch.cuda.is_available():
                image1 = image1.cuda(non_blocking=True)
                image2 = image2.cuda(non_blocking=True)
                labels = labels.cuda(non_blocking=True)
            
            # Get features and uncertainties from the model
            features, features_std = model(image1, image2)
       
            
            test_features.append(features)
            test_features_std.append(features_std)
            test_labels.append(labels)

    # Concatenate the list of tensors into a single tensor
    test_features = torch.cat(test_features, dim=0)
    test_features_std = torch.cat(test_features_std, dim=0)
    test_labels = torch.cat(test_labels, dim=0)

    return test_features, test_features_std, test_labels
