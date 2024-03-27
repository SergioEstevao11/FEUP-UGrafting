"""ResNet in PyTorch.
ImageNet-Style ResNet
[1] Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun
    Deep Residual Learning for Image Recognition. arXiv:1512.03385
Adapted from: https://github.com/bearpaw/pytorch-classification
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(BasicBlock, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, is_last=False):
        super(Bottleneck, self).__init__()
        self.is_last = is_last
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, self.expansion * planes, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(self.expansion * planes)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != self.expansion * planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, self.expansion * planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = F.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        if self.is_last:
            return out, preact
        else:
            return out


class ResNet(nn.Module):
    def __init__(self, block, num_blocks, in_channel=3, zero_init_residual=False):
        super(ResNet, self).__init__()
        self.in_planes = 64

        self.conv1 = nn.Conv2d(in_channel, 64, kernel_size=3, stride=1, padding=1,
                               bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2)
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves
        # like an identity. This improves the model by 0.2~0.3% according to:
        # https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)
        layers = []
        for i in range(num_blocks):
            stride = strides[i]
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)

    def forward(self, x, layer=100):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        return out


def resnet18(**kwargs):
    return ResNet(BasicBlock, [2, 2, 2, 2], **kwargs)


def resnet34(**kwargs):
    return ResNet(BasicBlock, [3, 4, 6, 3], **kwargs)


def resnet50(**kwargs):
    return ResNet(Bottleneck, [3, 4, 6, 3], **kwargs)


def resnet101(**kwargs):
    return ResNet(Bottleneck, [3, 4, 23, 3], **kwargs)


model_dict = {
    'resnet18': [resnet18, 512],
    'resnet34': [resnet34, 512],
    'resnet50': [resnet50, 2048],
    'resnet101': [resnet101, 2048],
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

class conResNet(nn.Module):
    """backbone + projection head"""

    def __init__(self, name='resnet50', head='mc-dropout', feat_dim=128, n_heads=5):
        super(conResNet, self).__init__()

        self.backbone_task = name
        model_fun, dim_in = model_dict[name]
        self.total_var = 0
        self.encoder = model_fun()
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

        elif head == 'mc-dropout':
            self.fc1 = nn.Linear(dim_in, dim_in)
            self.fc2 = nn.Linear(dim_in, dim_in)
            self.fc3 = nn.Linear(dim_in, feat_dim)
            self.act = nn.ReLU()


        else:
            raise NotImplementedError(
                'head not supported: {}'.format(head))
        
    def apply_mc_dropout(self, x):
        # Apply MC-Dropout pattern using defined layers
        x = self.act(self.fc1(x))
        x = MC_dropout(x, p=self.pdrop, mask=True)
        x = self.act(self.fc2(x))
        x = MC_dropout(x, p=self.pdrop, mask=True)
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
        print(f"number of classes is {num_classes}")
        self.fc = nn.Linear(dim_in, num_classes)

    def forward(self, features):
        return self.fc(features)


class MultiHeadSegResNet(nn.Module):
    def __init__(self, name='resnet50', num_classes=10, n_heads=5):
        super(MultiHeadSegResNet, self).__init__()
        model_fun, _ = model_dict[name]
        self.encoder = model_fun()
        self.n_heads = n_heads
        
        # Create multiple segmentation heads
        self.segmentation_heads = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(512, 256, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(256, 128, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                nn.Conv2d(128, num_classes, kernel_size=1)
            ) for _ in range(n_heads)
        ])
    
    def forward(self, x):
        # Encode the input
        encoded_features = self.encoder(x)
        
        # Get segmentation output from each head
        seg_maps = [head(encoded_features) for head in self.segmentation_heads]
        
        # Stack outputs to compute mean and variance
        seg_maps_stack = torch.stack(seg_maps, dim=0)
        
        # Compute mean and variance across the heads for each pixel
        mean_seg_maps = torch.mean(seg_maps_stack, dim=0)
        variance_seg_maps = torch.var(seg_maps_stack, dim=0)
        
        return mean_seg_maps, variance_seg_maps, seg_maps
