import torch
import argparse
import torch.distributions as dists
from models.concatenate import MyEnsemble, MyUQEnsemble
import numpy as np
from Dataloader.dataloader import data_loader
from Train.linear_eval import set_model_linear, predict
from utils.metrics import *
from plotting.UQ_viz import *
from utils.util import thresholding_mechanism


def parse_option():
    parser = argparse.ArgumentParser('argument for training')
    parser.add_argument('--dataset', type=str, default='cifar10',
                        choices=['cifar10', 'cifar100', 'svhn'], help='dataset')
    parser.add_argument('--model_path', type=str,
                        default='t', help='model path')
    parser.add_argument('--classifier_path', type=str,
                        default='', help='classifier path')
    parser.add_argument('--semi', action='store_true',
                        help='semi-supervised')
    parser.add_argument('--semi_percent', type=int, default=10,
                        help='percentage of data usage in semi-supervised')
    parser.add_argument('--nh', type=int, default=1,
                        help='number of heads')
    parser.add_argument('--lamda1', type=float, default=0,
                        help='uncertainty_penalty_weight')
    parser.add_argument('--lamda2', type=float, default=0.1,
                        help='uncertainty_threshold')

    parser.add_argument('--ensemble', type=int, default=1,
                        help='number of ensemble models')
    
    parser.add_argument('--ugraft_probing', action='store_true',
                        help='true if linear probing was done on top of the ugraft module')

    parser.add_argument('--backbone', type=str,
                        help='backbone model (vit, resnet50, etc.)')
    
    parser.add_argument('--uq_method', type=str,
                        help='uq method (mlp, mc-dropout, direct-modeling)')
    opt = parser.parse_args()
    return opt


def ensemble(n, nh, targets, n_cls, test_loader, semi=False, model_dir=".", classifier_dir=".", opt=None):
    probs_ensemble2_model = []
    # print all the hyperparameters
    print("=====Hyperparameters=====")
    print(f"n is {n}")
    print(f"nh is {nh}")
    print(f"n_cls is {n_cls}")
    print(f"test_loader is {test_loader}")
    print(f"semi is {semi}")
    print(f"model_dir is {model_dir}")
    print(f"classifier_dir is {classifier_dir}")

    if semi == True:
        for i in range(n):
            linear_model_path =model_dir
            simclr_path =classifier_dir
            model, classifier, criterion = set_model_linear(number_cls=n_cls, path=simclr_path, nh=nh, opt=opt)
            classifier.load_state_dict(torch.load(linear_model_path))
            #linear_model = MyEnsemble(model, classifier).cuda().eval()
            probs_ensemble2_model.append(predict(test_loader, linear_model, laplace=False))

    else:
        for i in range(n):
            linear_model_path = model_dir
            simclr_path = classifier_dir
            print(f"nh is {nh}")
            model, classifier, criterion = set_model_linear(number_cls=n_cls, path=simclr_path, nh=nh, opt=opt)
            classifier.load_state_dict(torch.load(linear_model_path))

            linear_model = MyUQEnsemble(model, classifier, opt.ugraft_probing).cuda().eval()


            prediction, variation = predict(test_loader, linear_model)
            print(f"prediction is {prediction.shape}")
            print(f"variation is {variation.shape}")
            
            accepted_indices, rejected_indices = thresholding_mechanism(prediction, variation, method='average')

            filtered_predictions = prediction[accepted_indices]
            print("Filtered predictions shape:", filtered_predictions.shape)
            print("Indices of accepted predictions:", accepted_indices)
            prediction = filtered_predictions
            targets = targets[accepted_indices]


            probs_ensemble2_model.append(prediction)
    print(variation[0])
    print(len(variation))
    probs_ensemble2_model = np.array(probs_ensemble2_model)
    probs_ensemble2 = np.mean(probs_ensemble2_model, 0)
    acc_ensemble2 = (probs_ensemble2.argmax(-1) == targets).mean()
    oe = OELoss()
    sce = SCELoss()
    ace = ACELoss()
    tace = TACELoss()
    ece = ECELoss()
    mce = MCELoss()
    brier = BrierScore()
    auc_roc = AUCROC()
    oe_res = oe.loss(output=probs_ensemble2, labels=targets, logits=True)
    sce_res = sce.loss(output=probs_ensemble2, labels=targets, logits=True)
    ace_res = ace.loss(output=probs_ensemble2, labels=targets, logits=True)
    tace_res = tace.loss(output=probs_ensemble2, labels=targets, logits=True)
    ece_res = ece.loss(output=probs_ensemble2, labels=targets, logits=True, n_bins=15)
    mce_res = mce.loss(output=probs_ensemble2, labels=targets, logits=True, n_bins=5)
    nll_ensemble2 = -dists.Categorical(torch.softmax(torch.tensor(probs_ensemble2), dim=-1)).log_prob(
        torch.tensor(targets)).mean()
    

    # aditional metrics
    # brier_score_res = brier.score(targets, probs_ensemble2)
    # log_loss_res = log_loss(targets, probs_ensemble2)
    # auc_roc_res = roc_auc_score(targets, probs_ensemble2, multi_class='ovr')
    # auc_pr_res = average_precision_score(targets, probs_ensemble2, average='macro')
    # f1_res = f1_score(targets, np.argmax(probs_ensemble2, axis=1), average='macro')
    
    plot_precision_recall_curve_multiclass(targets, probs_ensemble2, n_cls) 
    plot_average_precision_recall_curve(targets, probs_ensemble2, n_cls)
    results = {
        "ACCURACY": 100 * acc_ensemble2,
        "NLL": 1 * nll_ensemble2.numpy(),
        "ECE": ece_res,
        "OE": oe_res,
        "ACE": ace_res,
        "SCE": sce_res,
        "TACE": tace_res,
        "MCE": mce_res,

    }
    # print("number of ensemble is {}".format(n))
    print(
        f'[ensemble] Acc.: {acc_ensemble2:.1%}; ECE: {ece_res:.1%}; NLL: {nll_ensemble2:.3}; '
        f'OE: {oe_res:.3}; MCE: {mce_res:.3};SCE: {sce_res:.3}; ACE: {ace_res:.3}; TACE: {tace_res:.3}')
    return results


def train():
    opt = parse_option()
    if opt.dataset == "cifar10":
        n_cls = 10
    elif opt.dataset == "cifar100":
        n_cls = 100
    elif opt.dataset == 'svhn':
        n_cls = 10

    if opt.semi:
        smi = True
    else:
        smi = False


    train_loader, val_loader, test_loader, targets = data_loader(opt.dataset, batch_size=128, semi=smi,
                                                                 semi_percent=opt.semi_percent)
    #ensemble(n, nh, targets, n_cls, test_loader, semi=False, model_dir=".", classifier_dir=".")
    ensemble(opt.ensemble, opt.nh, targets, n_cls, test_loader, smi, opt.model_path,opt.classifier_path, opt)


if __name__ == "__main__":
    train()
