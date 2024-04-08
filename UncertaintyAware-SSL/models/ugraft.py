import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.resnet import *
from models.backbones.vit import *

model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
    'vit' : [VisualTransformer, 1024]
}


class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose"""

    def __init__(self, dim, affine=True):
        super(LinearBatchNorm, self).__init__()
        self.dim = dim
        self.bn = nn.BatchNorm2d(dim, affine=affine)

    def forward(self, x):
        x = x.view(-1, self.dim, 1, 1)
        x = self.bn(x)
        x = x.view(-1, self.dim)
        return x
    

def MC_dropout(act_vec, p=0.5, mask=True):
    return F.dropout(act_vec, p=p, training=mask, inplace=False)

class UGraft(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='vit', head='mlp', feat_dim=128, n_heads=5, image_shape=(3, 32, 32)):
        super(UGraft, self).__init__()

        self.backbone_task = name
        model_fun, dim_in = model_dict[name]
        self.total_var = 0
        self.encoder = model_fun() #TODO Add arguments passing to the model
        self.proj = []
        self.n_heads = n_heads
        self.head_type = head
        self.pdrop = 0.5

        if head == 'linear':
            self.proj = nn.Linear(dim_in, feat_dim)

        elif head == 'mlp':
            self.proj = nn.ModuleList()
            for _ in range(n_heads):
                pro = nn.Sequential(
                    nn.Linear(dim_in, dim_in),
                    nn.ReLU(inplace=True),
                    nn.Linear(dim_in, feat_dim)
                )
                self.proj.append(pro)
        
        elif head == "direct-modelling":
            self.n_heads = 1
            
            self.proj_mean = nn.ModuleList([
            nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=False),
                nn.Linear(dim_in, feat_dim)
            ) for _ in range(n_heads)
            ])

            self.proj_variance = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(dim_in, dim_in),
                    nn.ReLU(inplace=False),
                    nn.Linear(dim_in, feat_dim),
                    nn.Softplus()  # to ensure variance is positive
                ) for _ in range(n_heads)
            ])

        elif head == 'mc-dropout':
            self.fc1 = nn.Linear(dim_in, dim_in)
            self.fc2 = nn.Linear(dim_in, dim_in)
            self.fc3 = nn.Linear(dim_in, feat_dim)
            self.act = nn.ReLU(inplace=False)


        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
    def apply_mc_dropout(self, x):
        # Apply MC-Dropout pattern using defined layers
        x = MC_dropout(x, p=self.pdrop, mask=True)
        x = self.act(self.fc1(x))
        x = MC_dropout(x, p=self.pdrop, mask=True)
        x = self.act(self.fc2(x))
        x = self.fc3(x)  # No dropout after the last layer
        return x

    def forward(self, x1, x2, sample=True):
        f1, f2 = self.encoder(x1), self.encoder(x2)
        res1, res2 = [], []

        if self.head_type == 'mc-dropout':
            for _ in range(self.n_heads):
                x1_dropout = self.apply_mc_dropout(f1)
                x2_dropout = self.apply_mc_dropout(f2)
                res1.append(F.normalize(x1_dropout, dim=1))
                res2.append(F.normalize(x2_dropout, dim=1))

        elif self.head_type == 'mlp':
            for proj in self.proj:
                res1.append(F.normalize(proj(f1), dim=1))
                res2.append(F.normalize(proj(f2), dim=1))
        
        elif self.head_type == 'direct-modelling':
            res1_mean, res2_mean = [], []
            res1_variance, res2_variance = [], []

            for i in range(self.n_heads):
                # Project to mean and variance
                mean1 = self.proj_mean[i](f1)
                variance1 = self.proj_variance[i](f1) + 1e-6  # Ensure non-zero variance for stability
                
                mean2 = self.proj_mean[i](f2)
                variance2 = self.proj_variance[i](f2) + 1e-6
                
                # Normalize the means for stability in contrastive learning
                res1_mean.append(F.normalize(mean1, dim=1))
                res2_mean.append(F.normalize(mean2, dim=1))
                
                res1_variance.append(variance1)
                res2_variance.append(variance2)

            feat1_mean = torch.mean(torch.stack(res1_mean), dim=0)
            feat2_mean = torch.mean(torch.stack(res2_mean), dim=0)
            features_mean = torch.cat([feat1_mean.unsqueeze(1), feat2_mean.unsqueeze(1)], dim=1)
            
            feat1_variance = torch.mean(torch.stack(res1_variance), dim=0)
            feat2_variance = torch.mean(torch.stack(res2_variance), dim=0)
            features_variance = torch.cat([feat1_variance.unsqueeze(1), feat2_variance.unsqueeze(1)], dim=1)

            return features_mean, features_variance

                
        else:  # for linear
            res1 = [F.normalize(self.proj(f1), dim=1)] * self.n_heads
            res2 = [F.normalize(self.proj(f2), dim=1)] * self.n_heads
        
        
        # Compute the mean and standard deviation of the representations
        feat1 = torch.mean(torch.stack(res1), dim=0)
        feat2 = torch.mean(torch.stack(res2), dim=0)
        
        feat1_std = torch.sqrt(torch.var(torch.stack(res1), dim=0) + 0.0001)
        feat2_std = torch.sqrt(torch.var(torch.stack(res2), dim=0) + 0.0001)
        features = torch.cat([feat1.unsqueeze(1), feat2.unsqueeze(1)], dim=1)
        features_std = torch.cat([feat1_std.unsqueeze(1), feat2_std.unsqueeze(1)], dim=1)


        return features, features_std


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__init__()

        _, dim_in = model_dict[name]
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, features):
        return self.fc(features)


