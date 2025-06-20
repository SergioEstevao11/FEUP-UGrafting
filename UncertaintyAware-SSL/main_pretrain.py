from __future__ import print_function

import os
import argparse
import time
import math
import pandas as pd
import numpy as np
import tensorboard_logger as tb_logger
import torch
from Train.pretrain import train, set_model, evaluate_uncertainty
from Dataloader.dataloader import set_loader_simclr, dataloader_UQ
from plotting.UQ_viz import visualize_with_tsne, visualize_with_3d_histogram, visualize_with_tsne_3d_histogram, linegraph_minmax_area

from utils.util import adjust_learning_rate
from utils.util import set_optimizer
import torch.profiler


def parse_option():
    parser = argparse.ArgumentParser('argument for training')

    parser.add_argument('--print_freq', type=int, default=10,
                        help='print frequency')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='batch_size')
    parser.add_argument('--num_workers', type=int, default=8,
                        help='num of workers to use')
    parser.add_argument('--epochs', type=int, default=800,
                        help='number of training epochs')
    parser.add_argument('--ensemble', type=int, default=1,
                        help='number of ensemble')
    # optimization
    parser.add_argument('--learning_rate', type=float, default=0.0005,
                        help='learning rate')
    parser.add_argument('--lr_decay_epochs', type=str, default='700,800,900',
                        help='where to decay lr, can be a list')
    parser.add_argument('--lr_decay_rate', type=float, default=0.1,
                        help='decay rate for learning rate')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9,
                        help='momentum')

    # model dataset
    parser.add_argument('--model', type=str, default='resnet50')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        help='dataset')
    parser.add_argument('--data_folder', type=str, default=None, help='path to custom dataset')
    parser.add_argument('--size_randomcrop', type=int, default=32, help='parameter for RandomResizedCrop')

    # hyperparameters
    parser.add_argument('--temp', type=float, default=0.07,
                        help='temperature for loss function')
    parser.add_argument('--nh', type=int, default=10,
                        help='number of heads')
    parser.add_argument('--lamda1', type=float, default=1,
                        help='uncertainty_penalty_weight')
    parser.add_argument('--lamda2', type=float, default=0.8,
                        help='uncertainty_threshold')

    # other setting
    parser.add_argument('--cosine', action='store_true',
                        help='using cosine annealing')
    parser.add_argument('--warm', action='store_true',
                        help='warm-up for large batch training')
    parser.add_argument('--saved_model', type=str, default=".",
                        help='path to save classifier')
    parser.add_argument('--log', type=str, default='.',
                        help='path to save tensorboard logs')
    parser.add_argument('--syncBN', action='store_true', 
                        help='enable synchronized batch normalization')
    

    opt = parser.parse_args()

    # set the path according to the environment
    if opt.data_folder is None:
        opt.data_folder = './DATA'
    
    print(f"THIS: {opt.saved_model}")
    opt.save_folder = os.path.join(opt.saved_model, '{}_experiments'.format(opt.dataset))
    if not os.path.isdir(opt.save_folder):
        os.makedirs(opt.save_folder)
    if opt.batch_size > 256:
        opt.warm = True
    if opt.warm:
        opt.warmup_from = 0.01
        opt.warm_epochs = 10
        if opt.cosine:
            eta_min = opt.learning_rate * (opt.lr_decay_rate ** 3)
            opt.warmup_to = eta_min + (opt.learning_rate - eta_min) * (
                    1 + math.cos(math.pi * opt.warm_epochs / opt.epochs)) / 2
        else:
            opt.warmup_to = opt.learning_rate
    return opt


def main():
    opt = parse_option()
    torch.cuda.empty_cache()
    # build data loader
    train_loader, image_shape, test_loader = dataloader_UQ(dataset=opt.dataset, batch_size=opt.batch_size, num_workers=opt.num_workers,
                                     size_randomcrop=opt.size_randomcrop)

    for i in range(opt.ensemble):
        torch.manual_seed(i)
        torch.cuda.manual_seed(i)
        model, criterion = set_model(model_name=opt.model, temperature=opt.temp, syncBN=opt.syncBN, lamda1=opt.lamda1,
                                     lamda2=opt.lamda2,
                                     batch_size=opt.batch_size, nh=opt.nh, image_shape=image_shape)

        # build optimizer
        optimizer = set_optimizer(opt, model)
        opt.tb_path = os.path.join(opt.log, '{}_{}_{}_{}_epochs{}_{}heads_lamda1{}_lamda2{}.pth'.format(opt.model, model.head_type, opt.dataset, i, epoch, opt.nh,
                                                                                opt.lamda1,
                                                                                opt.lamda2))
        # tensorboard
        opt.tb_path = os.path.join(opt.log, '{}_{}_{}_{}_epochs{}_{}heads_lamda1{}_lamda2{}.pth'.format(opt.model, model.head_type, opt.dataset, i, epoch, opt.nh,
                                                                                opt.lamda1,
                                                                                opt.lamda2))
        logger = tb_logger.Logger(logdir=opt.tb_path, flush_secs=2)

        time1 = time.time()
        l1 = []
        l2 = []
        l3 = []
        UQ_l = []
        UQ_l_file = os.path.join(
            'Uncertainty_{}_{}_{}_{}epochs_{}heads_lamda1{}_lamda2{}.pt'.format(opt.model, opt.dataset, i, opt.epochs, opt.nh,
                                                                          opt.lamda1,
                                                                          opt.lamda2))
        feats = []
        feats_l_file = os.path.join(
            'Feats_{}_{}_{}_{}epochs_{}heads_lamda1{}_lamda2{}.pt'.format(opt.model, opt.dataset, i, opt.epochs, opt.nh,
                                                                          opt.lamda1,
                                                                          opt.lamda2))
        
        std_data = []
        # training routine
        total_params = sum(p.numel() for p in model.parameters())
        print(f"Number of parameters: {total_params}")


        with torch.profiler.profile(
                schedule=torch.profiler.schedule(wait=1, warmup=1, active=3, repeat=2),
                on_trace_ready=torch.profiler.tensorboard_trace_handler('./log/memory_profile'),
                record_shapes=True,
                profile_memory=True,
                with_stack=True
        ) as prof:
            for epoch in range(1, opt.epochs + 1):
                adjust_learning_rate(opt, optimizer, epoch)
                # train for one epoch
                time3 = time.time()
                loss, std_loss, std_loss2 = train(train_loader, model, criterion, optimizer, epoch, opt)
                time4 = time.time()
                print('ensemble {}, epoch {}, total time {:.2f}'.format(i, epoch, time4 - time3))
                print(f"total loss: {loss}, std_loss: {std_loss}, std_loss2: {std_loss2}")
                l1.append(loss)
                l2.append(std_loss)
                l3.append(std_loss2)
                # tensorboard logger
                logger.log_value('std_loss1', std_loss, epoch)
                logger.log_value('std_loss2', std_loss2, epoch)
                logger.log_value('total_loss', loss, epoch)
                logger.log_value('learning_rate', optimizer.param_groups[0]['lr'], epoch)
                # logger.log_value('std', std_loss, epoch)
            
                #PLOTTING
                if epoch % 5 == 0:
                    features, std, labels = evaluate_uncertainty(test_loader, model)
                    # UQ_l.append(std)
                    # feats.append(features)
                    min = torch.min(std)
                    max = torch.max(std)
                    mean = std.mean(dim=(0, 1, 2))

                    #select 100 random values
                    rand_vals = std[torch.randint(0, len(std), (100,))] 
                
                    std_data.append((min.cpu(), max.cpu(), mean.cpu(), std.cpu()))
                    
                    loss_res = pd.DataFrame({"total_loss": l1, "stdloss1": l2, "stdloss2": l3})
                    os.makedirs("./csv_loss", exist_ok=True)
                    loss_res.to_csv(
                        "./csv_loss/{}_{}_{}_losses_{}heads_lamda1{}_lamda2{}.csv".format(opt.model, model.head_type, opt.dataset, opt.nh, opt.lamda1, opt.lamda2))
                    save_file = os.path.join(
                        opt.save_folder,
                        '{}_{}_{}_{}_epoch{}_{}heads_lamda1{}_lamda2{}.pth'.format(opt.model, model.head_type, opt.dataset, i, opt.epochs, opt.nh,
                                                                          opt.lamda1,
                                                                          opt.lamda2))
                    # print(f"len of test features: ", features.shape)
                    # print(f"len of test std: ", std.shape)
                    # print(f"len of test labels: ", labels)
                    #torch.save(UQ_l, UQ_l_file)
                
                if epoch % 10 == 0:
                    checkpoint_file = os.path.join(
                        "./checkpoints",
                        '{}_{}_{}_{}_epochs{}_{}heads_lamda1{}_lamda2{}.pth'.format(opt.model, model.head_type, opt.dataset, i, epoch, opt.nh,
                                                                                    opt.lamda1,
                                                                                    opt.lamda2))
                    torch.save(model.state_dict(), checkpoint_file)
                    

                torch.cuda.empty_cache()

        
        #linegraph_minmax_area(std_data, opt.epochs)
        
        # print(std_data)
        # with open(r'./std_mean.txt', 'w') as fp:
        #     for min_val, max_val, mean_val in std_data:
        #         fp.write(f"({min_val.item():.4f}, {max_val.item():.4f}, {mean_val.item():.4f})\n")
        #     print('Done')

        torch.save(UQ_l, UQ_l_file)
        torch.save(feats, feats_l_file)

        time2 = time.time()
        print('ensemble {}, total time {:.2f}'.format(i, time2 - time1))
        loss_res = pd.DataFrame({"total_loss": l1, "stdloss1": l2, "stdloss2": l3})
        os.makedirs("./csv_loss", exist_ok=True)
        loss_res.to_csv(
            "./csv_loss/{}_{}_{}_losses_{}heads_lamda1{}_lamda2{}.csv".format(opt.model, model.head_type, opt.dataset, opt.nh, opt.lamda1, opt.lamda2))
        save_file = os.path.join(
            opt.save_folder,
            '{}_{}_{}_{}_epoch{}_{}heads_lamda1{}_lamda2{}.pth'.format(opt.model, model.head_type, opt.dataset, i, opt.epochs, opt.nh,
                                                                          opt.lamda1,
                                                                          opt.lamda2))
        torch.save(model.state_dict(), save_file)


if __name__ == '__main__':
    main()
