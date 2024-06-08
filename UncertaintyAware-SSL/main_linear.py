from __future__ import print_function

import argparse
import time
import math
import torch
import copy
import os
import numpy as np
from Dataloader.dataloader import linear_data_loader
from utils.util import adjust_learning_rate
from utils.util import set_optimizer

from Train.linear_eval import set_model_linear, train, evaluate, validate
from torch.utils.tensorboard import SummaryWriter


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=12,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=100,
                        help='number of training epochs')

    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.01,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='60,75,90',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.2,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=0,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')
    parser.add_argument('--patience', type=int, default=10,
                        help='patience for early stopping')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn'], help='dataset')
    parser.add_argument('--data_folder', type=str, default='.',
                        help='path to dataset')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')

    parser.add_argument('--ckpt', type=str, default='./saved_models/cifar10_models/simclr_cifar10_0_epoch200.pth',
                        help='path to pre-trained model')
    parser.add_argument('--classifier_path', type=str, default='.',
                        help='path to save classifier')
    parser.add_argument('--log', type=str, default='.',
                        help='path to save tensorboard logs')
    parser.add_argument('--semi', action='store_true',
                        help='semi-supervised')
    parser.add_argument('--semi_percent', type=int, default=10,
                        help='percentage of data usage in semi-supervised')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='number of ensemble models')
    parser.add_argument('--nh', type=int, default=1,
                        help='number of heads')
    parser.add_argument('--lamda1', type=float, default=0,
                        help='uncertainty_penalty_weight')
    parser.add_argument('--lamda2', type=float, default=0.1,
                        help='uncertainty_threshold')
    
    parser.add_argument('--ugraft_probing', action='store_true',
                        help='true if linear probing was done on top of the ugraft module')
    parser.add_argument('--backbone', type=str,
                        help='backbone model (vit, resnet50, etc.)')
    parser.add_argument('--uq_method', type=str,
                        help='uq method (mlp, mc-dropout, direct-modeling)')
    parser.add_argument('--prev_epochs', type=str, default="?",
                        help='number of epochs the encoder was trained on')

    opt = parser.parse_args()

    iterations = opt.lr_decay_epochs.split(',')
    opt.lr_decay_epochs = list([])
    for it in iterations:
        opt.lr_decay_epochs.append(int(it))
    opt.tb_path = os.path.join(opt.log, '{}/linear_evaluation/'.format(opt.dataset))

    # warm-up for large-batch training,
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate


    # if opt.dataset == 'cifar10':
    #     opt.n_cls = 10
    # elif opt.dataset == 'cifar100':
    #     opt.n_cls = 100
    # elif opt.dataset == 'svhn':
    #     opt.n_cls = 10
    # else:
    #     raise ValueError('dataset not supported: {}'.format(opt.dataset))


    if opt.semi:
        opt.classifier_path = os.path.join(opt.classifier_path, '{}/semi_model/'.format(opt.dataset))
    else:
        opt.classifier_path = os.path.join(opt.classifier_path, '{}_experiments/linear_models'.format(opt.dataset))
    if not os.path.isdir(opt.classifier_path):
        os.makedirs(opt.classifier_path)
    return opt


def main():
    opt = parse_option()

    writer = SummaryWriter(log_dir=opt.tb_path)
    # build data loader
    train_loader, val_loader, test_loader, targets = linear_data_loader(dataset=opt.dataset, batch_size=opt.batch_size,
                                                           semi=opt.semi, semi_percent=opt.semi_percent)
    num_classes = np.unique(targets).shape[0]
    
    ensemble = opt.ensemble
    for i in range(ensemble):
        best_acc = 0
        best_epoch = 0
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        model, classifier, criterion = set_model_linear(number_cls=num_classes, path=opt.ckpt,
                                                        nh=opt.nh, opt=opt)
        for param in model.parameters():
            param.requires_grad = False
        # build optimizer
        optimizer = set_optimizer(opt, classifier)
        print('ensemble number is {}:'.format(i))
        # training routine


        loss_l = []
        acc_l = []
        val_acc_l = []
        val_loss_l = []
        print("EPOCHS: ", opt.epochs)
        for epoch in range(1, opt.epochs + 1):
            adjust_learning_rate(opt, optimizer, epoch)

            # train for one epoch
            time1 = time.time()
            loss, acc = train(train_loader, model, classifier, criterion,
                              optimizer, epoch, opt)
            time2 = time.time()
            print('ensemble number {}, Train epoch {}, total time {:.2f}, accuracy:{:.2f}'.format(i,
                                                                                                  epoch, time2 - time1,
                                                                                                  acc))

            # eval for one epoch
            val_loss, val_acc = validate(val_loader, model, classifier, criterion, opt)
            writer.add_scalar("Loss/train", loss, epoch)
            writer.add_scalar('train/learning_rate', optimizer.param_groups[0]['lr'], epoch)
            writer.add_scalar('accuracy', val_acc, epoch)
            writer.add_scalar("Loss/eval", val_loss, epoch)

            loss_l.append(loss)
            acc_l.append(acc)
            val_acc_l.append(val_acc)
            val_loss_l.append(val_loss)

            if val_acc > best_acc:
                best_epoch = epoch
                best_acc = val_acc
                best_classifier = copy.deepcopy(classifier)
                print('best accuracy: {:.2f}'.format(best_acc))
            if epoch - best_epoch > opt.patience:
                break
        evaluate(test_loader, model, best_classifier, opt)

        writer.flush()
        writer.close()
        # save the best model
        if opt.semi:

            opt.classifier_path = os.path.join(
                opt.classifier_path,
                '{}_{}_semi-linear_{}_epoch{}_{}heads_lamda1{}_lamda2{}.pth'.format(
                            opt.model,
                            model.head_type,
                           opt.dataset,
                           i,
                           opt.epochs,
                           opt.nh,
                           opt.lamda1,
                           opt.lamda2))
            

        else:
            file_name = '{}_{}_prevepochs{}_linearEval_{}_epoch{}_{}heads_lamda1{}_lamda2{}'.format(
                opt.model,
                model.head_type,
                opt.prev_epochs,
                opt.dataset,
                i,
                opt.epochs,
                opt.nh,
                opt.lamda1,
                opt.lamda2
            )
    
        if opt.ugraft_probing:
            file_name += '_withUQ'
            
        file_name += '.pth'
        
        opt.classifier_path = os.path.join(opt.classifier_path, file_name)
            
        print("acc_l", acc_l)
        print("val_acc_l", val_acc_l)
        print("loss_l", loss_l)
        print("val_loss_l", val_loss_l)



        torch.save(best_classifier.state_dict(),
                    opt.classifier_path)


if __name__ == '__main__':
    main()
