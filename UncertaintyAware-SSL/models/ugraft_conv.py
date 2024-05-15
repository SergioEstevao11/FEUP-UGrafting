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
    'vit': [VisualTransformer, 1024]
}

class LinearBatchNorm(nn.Module):
    """Implements BatchNorm1d by BatchNorm2d, for SyncBN purpose."""
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
    """UGraft model incorporating convolutional layers."""
    def __init__(self, name='vit', head='direct-modelling', feat_dim=128, n_heads=5, image_shape=(3, 32, 32)):
        super(UGraft, self).__init__()
        self.backbone_task = name
        model_fun, dim_in = model_dict[name]
        self.encoder = model_fun()
        self.n_heads = n_heads
        self.head_type = head
        self.pdrop = 0.5

        if head == 'linear':
            # Using 1x1 convolutions to simulate linear layers
            self.proj = nn.Conv2d(dim_in, feat_dim, 1)

        elif head == 'mlp':
            # Using convolutional blocks
            self.proj = nn.ModuleList()
            for _ in range(n_heads):
                self.proj.append(nn.Sequential(
                    nn.Conv2d(dim_in, dim_in, 1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(dim_in, feat_dim, 1)
                ))

        elif head == "direct-modelling":
            self.proj_mean = nn.ModuleList([nn.Sequential(
                nn.Conv2d(dim_in, dim_in, 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(dim_in, feat_dim, 1)
            ) for _ in range(n_heads)])

            self.proj_variance = nn.ModuleList([nn.Sequential(
                nn.Conv2d(dim_in, dim_in, 1),
                nn.ReLU(inplace=False),
                nn.Conv2d(dim_in, feat_dim, 1),
                nn.Softplus()  # Ensuring positive variance
            ) for _ in range(n_heads)])

        elif head == 'mc-dropout':
            self.fc1 = nn.Conv2d(dim_in, dim_in, 1)
            self.fc2 = nn.Conv2d(dim_in, dim_in, 1)
            self.fc3 = nn.Conv2d(dim_in, feat_dim, 1)
            self.act = nn.ReLU(inplace=False)

        else:
            raise NotImplementedError(f'head not supported: {head}')

    def forward(self, x1, x2, sample=True):
        f1, f2 = self.encoder(x1), self.encoder(x2)
        res1, res2 = [], []
        for i in range(self.n_heads):
            if self.head_type in ['mc-dropout', 'mlp']:
                x1_processed = self.proj[i](f1)
                x2_processed = self.proj[i](f2)
            elif self.head_type == 'direct-modelling':
                x1_processed = self.proj_mean[i](f1)
                x2_processed = self.proj_mean[i](f2)
                variance1 = self.proj_variance[i](f1)
                variance2 = self.proj_variance[i](f2)
                res1.append((x1_processed, variance1))
                res2.append((x2_processed, variance2))
            res1.append(F.normalize(x1_processed, dim=1))
            res2.append(F.normalize(x2_processed, dim=1))
        return res1, res2  # Return means and optionally variances for direct modelling

class LinearClassifier(nn.Module):
    """Linear classifier using convolution."""
    def __init__(self, name='resnet50', num_classes=10):
        super(LinearClassifier, self).__
