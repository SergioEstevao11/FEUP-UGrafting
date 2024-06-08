import torch
import torch.nn as nn
import torch.nn.functional as F
from models.backbones.resnet import *
from models.backbones.vit import *


model_dict = {
    'resnet18': [lambda image_size, channels: resnet18(in_channel=channels), 512],
    'resnet34': [lambda image_size, channels: resnet34(in_channel=channels), 512],
    'resnet50': [lambda image_size, channels: resnet50(in_channel=channels), 2048],
    'resnet101': [lambda image_size, channels: resnet101(in_channel=channels), 2048],
    'vit' : [lambda image_size, channels: VisualTransformer(image_size=image_size, channels=channels), 1024]
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

class TemperatureScaling(nn.Module):
    def __init__(self):
        super(TemperatureScaling, self).__init__()
        self.temperature = nn.Parameter(torch.ones(1) * 1.0)

    def forward(self, logits):
        return logits / self.temperature

class UGraft(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='resnet50', head='mlp', feat_dim=128, n_heads=5, image_shape=(3, 32, 32)):
        super(UGraft, self).__init__()
        print(f"Using backbone: {name}", 
              f" with head: {head}")
        self.backbone_task = name
        model_fun, dim_in = model_dict[name]
        self.total_var = 0
        self.encoder = model_fun(image_shape[1], image_shape[0])
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

            self.common_path = nn.Sequential(
                nn.Linear(dim_in, dim_in),
                nn.ReLU(inplace=False),
            )
            
            self.proj_linear = nn.Linear(dim_in, feat_dim)

            self.proj_variance = nn.Sequential(
                    nn.Linear(dim_in, feat_dim),
                    nn.Softplus()  # to ensure variance is positive
            ) 

            # nn.init.kaiming_uniform_(self.proj_variance[0].weight, a=0, mode='fan_in', nonlinearity='relu')
            # nn.init.constant_(self.proj_variance[0].bias, 0)
            
            # # Add a small positive constant to the initialized weights
            # with torch.no_grad():
            #     self.proj_variance[0].weight += 1e-6

            nn.init.constant_(self.proj_variance[0].weight, 0.1)
            nn.init.constant_(self.proj_variance[0].bias, 0.1)

        elif head == 'mc-dropout':
            self.fc1 = nn.Linear(dim_in, dim_in)
            self.fc2 = nn.Linear(dim_in, dim_in)
            self.fc3 = nn.Linear(dim_in, feat_dim)
            self.act = nn.ReLU(inplace=False)

        elif head == 'temp-scaling':
            self.temperature_scaling = TemperatureScaling()
            self.proj = nn.Linear(dim_in, feat_dim)

        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
    def apply_mc_dropout(self, x):
        # Apply MC-Dropout pattern using defined layers
        x = MC_dropout(x, p=self.pdrop, mask=True)
        x = self.act(self.fc2(x))
        x = self.fc3(x)  # No dropout after the last layer
        return x
    

    def forward(self, *inputs, sample=True):
        if len(inputs) == 1:
            return self.forward_single(inputs[0])
        elif len(inputs) == 2:
            return self.forward_dual(inputs[0], inputs[1])
        else:
            raise ValueError("Invalid number of inputs for forward pass. Expected 1 or 2 inputs.")


    def forward_dual(self, x1, x2):
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

            f1 = self.common_path(f1)
            f2 = self.common_path(f2)

            #for i in range(self.n_heads):
                # Project to mean and variance
            mean1 = self.proj_linear(f1)
            variance1 = self.proj_variance(f1) + 1e-6  # Ensure non-zero variance for stability
            
            mean2 = self.proj_linear(f2)
            variance2 = self.proj_variance(f2) + 1e-6
            
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

        elif self.head_type == 'temp-scaling':
            logits1 = self.proj(f1)
            logits2 = self.proj(f2)
            scaled_logits1 = self.temperature_scaling(logits1)
            scaled_logits2 = self.temperature_scaling(logits2)
            features = torch.cat([scaled_logits1.unsqueeze(1), scaled_logits2.unsqueeze(1)], dim=1)
            features_std = torch.sqrt(torch.var(features, dim=1) + 0.0001)
            features_mean = torch.mean(features, dim=1)
            return features_mean, features_std

                
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
    

    def forward_single(self, x):
        enconding = self.encoder(x)
        res = []

        if self.head_type == 'mc-dropout':
            for _ in range(self.n_heads):
                x_dropout = self.apply_mc_dropout(enconding)
                res.append(F.normalize(x_dropout, dim=1))

        elif self.head_type == 'mlp':
            for proj in self.proj:
                res.append(F.normalize(proj(enconding), dim=1))
        
        elif self.head_type == 'direct-modelling':
            res_mean = []
            res_variance = []

            enconding = self.common_path(enconding)

            #for i in range(self.n_heads):
                # Project to mean and variance
            mean = self.proj_linear(enconding)
            variance = self.proj_variance(enconding) + 1e-6  # Ensure non-zero variance for stability
            
            
            # Normalize the means for stability in contrastive learning
            res_mean.append(F.normalize(mean, dim=1))
            
            res_variance.append(variance)

            features_mean = torch.mean(torch.stack(res_mean), dim=0)
            features_variance = torch.mean(torch.stack(res_variance), dim=0)

            return features_mean, features_variance
        
        elif self.head_type == 'temp-scaling':
            logits = self.proj(enconding)
            scaled_logits = self.temperature_scaling(logits)
            return F.normalize(scaled_logits, dim=1)

        else:  # for linear
            res = [F.normalize(self.proj(enconding), dim=1)] * self.n_heads
        
        
        # Compute the mean and standard deviation of the representations
        features = torch.mean(torch.stack(res), dim=0)
        features_variance = torch.sqrt(torch.var(torch.stack(res), dim=0) + 0.0001)

        return features, features_variance
            


class LinearClassifier(nn.Module):
    """Linear classifier"""

    # def __init__(self, name='resnet50', num_classes=10):
    #     super(LinearClassifier, self).__init__()

    #     _, dim_in = model_dict[name]
    #     self.fc = nn.Linear(dim_in, num_classes)
    def __init__(self, embedding_dim, num_classes=10):
        super(LinearClassifier, self).__init__()

        self.fc = nn.Linear(embedding_dim, num_classes)

    def forward(self, features):
        return self.fc(features)


