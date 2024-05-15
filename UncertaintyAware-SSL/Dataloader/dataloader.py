from torch.utils.data import DataLoader, random_split
import torchvision.transforms as transforms
from torchvision import datasets
import torch
from utils.util import TwoCropTransform
import utils
from PIL import Image

# search tiny imagenet C 
# from https://github.com/clint-kristopher-morris/DINO_concise/blob/main/notebooks_/Concise_DINO-Demo.ipynb

class DataAugmentationDINO(object):
    def __init__(self, global_crops_scale, local_crops_scale, local_crops_number, image_size=64):
        flip_and_color_jitter = transforms.Compose([
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomApply(
                [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
                p=0.8
            ),
            transforms.RandomGrayscale(p=0.2),
        ])
        
        normalize = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
        ])
        # first global crop
        self.global_transfo1 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(3, 1.0),
            normalize,
        ])
        # second global crop
        self.global_transfo2 = transforms.Compose([
            transforms.RandomResizedCrop(image_size, scale=global_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(3, sigma=0.1),
            #transforms.Solarization(0.2),
            normalize,
        ])
        # transformation for the local small crops
        self.local_crops_number = local_crops_number
        self.local_transfo = transforms.Compose([
            transforms.RandomResizedCrop(int(96/(244/image_size)), scale=local_crops_scale, interpolation=Image.BICUBIC),
            flip_and_color_jitter,
            transforms.GaussianBlur(3, 0.5),
            normalize,
        ])

    def __call__(self, image):
        crops = []
        crops.append(self.global_transfo1(image))
        crops.append(self.global_transfo2(image))
        # for _ in range(self.local_crops_number):
        #     crops.append(self.local_transfo(image))
        return crops[0], crops[1]
    

def data_loader(dataset="cifar10", batch_size=512, semi=False, semi_percent=10, num_cores=12):
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "svhn":
        mean = (0.4376821, 0.4437697, 0.47280442)
        std = (0.19803012, 0.20101562, 0.19703614)

    normalize = transforms.Normalize(mean=mean, std=std)
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])
    sampler = None
    # datasets
    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root='../../DATA2/', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR10(root='../../DATA2/', train=False, download=True, transform=val_transform)

    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root='../../DATA2/', train=True, download=True, transform=train_transform)
        test_dataset = datasets.CIFAR100(root='../../DATA2/', train=False, download=True, transform=val_transform)
    elif dataset == "svhn":
        train_dataset = datasets.SVHN(
            root='../../DATA2/', split="train", download=True, transform=train_transform
        )
        test_dataset = datasets.SVHN(
            root='../../DATA2/', split="test", download=True, transform=val_transform
        )
    elif dataset == "tinyimagenet":
        pass
        # train_dataset = datasets.ImageFolder(root='../../DATA2/tiny-imagenet-200/train', transform=DataAugmentationDINO)
        # test_dataset = datasets.ImageFolder(root='../../DATA2/tiny-imagenet-200/val', transform=val_transform)
    if semi:
        per = semi_percent / 100
        x = int(per * len(train_dataset))
        y = int(len(train_dataset) - x)
        train, _ = random_split(train_dataset, [x, y])
    else:
        train = train_dataset

    train, val = random_split(train, [int(0.8 * len(train)),
                                      len(train) - int(0.8 * len(train))], generator=torch.Generator().manual_seed(42))

    train_loader = DataLoader(train,
                              batch_size=batch_size,
                              shuffle=True,
                              num_workers=num_cores,
                              drop_last=False
                              )
    val_loader = DataLoader(val,
                            batch_size=batch_size,
                            shuffle=False,
                            num_workers=num_cores,
                            drop_last=False)

    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=num_cores,
                             drop_last=False)

    targets = torch.cat([y for x, y in test_loader], dim=0).numpy()
    return train_loader, val_loader, test_loader, targets


def set_loader_simclr(dataset, batch_size, num_workers, data_dir='../../DATA2/', size_randomcrop=32):
    # construct data loader
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "svhn":
        mean = (0.4376821, 0.4437697, 0.47280442)
        std = (0.19803012, 0.20101562, 0.19703614)
    else:
        raise ValueError('dataset not supported: {}'.format(dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size_randomcrop, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == 'cifar10':
        train_dataset = datasets.CIFAR10(root=data_dir,
                                         transform=TwoCropTransform(train_transform),
                                         download=True)
    elif dataset == 'cifar100':
        train_dataset = datasets.CIFAR100(root=data_dir,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
    elif dataset == 'svhn':
        train_dataset = datasets.SVHN(
            root=data_dir, split="train", download=True, transform=TwoCropTransform(train_transform)
        )

    else:
        raise ValueError(dataset)

    print(f"train dataset length is {len(train_dataset)}")
    train_sampler = None
    image_shape = (3, size_randomcrop, size_randomcrop)
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=False, sampler=train_sampler)

    return train_loader, image_shape

def dataloader_UQ(dataset, batch_size, num_workers, data_dir='../../DATA2/', size_randomcrop=32): #still work this out
    # construct data loader
    if dataset == 'cifar10':
        mean = (0.4914, 0.4822, 0.4465)
        std = (0.2023, 0.1994, 0.2010)
    elif dataset == 'cifar100':
        mean = (0.5071, 0.4867, 0.4408)
        std = (0.2675, 0.2565, 0.2761)
    elif dataset == "svhn":
        mean = (0.4376821, 0.4437697, 0.47280442)
        std = (0.19803012, 0.20101562, 0.19703614)
    elif dataset == "tinyimagenet":
        mean = (0.4802, 0.4481, 0.3975) #dummy
        std = (0.2770, 0.2691, 0.2821) #dummy
    else:
        raise ValueError('dataset not supported: {}'.format(dataset))
    normalize = transforms.Normalize(mean=mean, std=std)

    #data transformations
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size=size_randomcrop, scale=(0.2, 1.)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomApply([
            transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)
        ], p=0.8),
        transforms.RandomGrayscale(p=0.2),
        transforms.ToTensor(),
        normalize,
    ])

    val_transform = transforms.Compose([
        transforms.ToTensor(),
        normalize,
    ])

    if dataset == "cifar10":
        train_dataset = datasets.CIFAR10(root=data_dir,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        test_dataset = datasets.CIFAR10(root=data_dir, train=False, download=True, transform=TwoCropTransform(val_transform))

    elif dataset == "cifar100":
        train_dataset = datasets.CIFAR100(root=data_dir,
                                          transform=TwoCropTransform(train_transform),
                                          download=True)
        test_dataset = datasets.CIFAR100(root=data_dir, train=False, download=True, transform=TwoCropTransform(val_transform))
    elif dataset == "svhn":
        train_dataset = datasets.SVHN(
            root=data_dir, split="train", download=True, transform=TwoCropTransform(train_transform)
        )
        test_dataset = datasets.SVHN(
            root=data_dir, split="test", download=True, transform=TwoCropTransform(val_transform)
        )

    elif dataset == "tinyimagenet":
        data_transform = DataAugmentationDINO(
            global_crops_scale=(0.4, 1.0), 
            local_crops_scale=(0.05, 0.4), 
            local_crops_number=6
        )

        # train_dataset = datasets.ImageFolder(root='../../DATA2/tiny-imagenet-200/train', transform=data_transform)
        # test_dataset = datasets.ImageFolder(root='../../DATA2/tiny-imagenet-200/val', transform=val_transform)
        train_dataset = datasets.ImageNet(root='../../DATA2/', train=True, download=True, transform=data_transform)
        test_dataset = datasets.ImageNet(root='../../DATA2/', train=False, download=True, transform=val_transform)

    else:
        raise ValueError(dataset)

    print(f"train dataset length is {len(train_dataset)}")
    train_sampler = None
    image_shape = (3, size_randomcrop, size_randomcrop)


   
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=(train_sampler is None),
        num_workers=num_workers, pin_memory=False, sampler=train_sampler)
    
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             num_workers=num_workers,
                             drop_last=False)

    return train_loader, image_shape, test_loader

