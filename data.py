import torch
import torchvision
import torchvision.transforms as transforms
from autoaugment import *


def get_dataloaders():

    train_transform = transforms.Compose([transforms.RandomCrop(32, padding=4, fill=128),  # fill parameter needs torchvision installed from source
                                          transforms.RandomHorizontalFlip(), CIFAR10Policy(),
                                          transforms.ToTensor(),
                                          Cutout(n_holes=1, length=16),
                                          transforms.Normalize(
        (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    test_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465),
                             (0.2023, 0.1994, 0.2010)),
    ])

    batch_size = 256

    train_set = torchvision.datasets.CIFAR10(root='../_Datasets', train=True,
                                             download=True, transform=train_transform)
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                               shuffle=True, num_workers=2)

    test_set = torchvision.datasets.CIFAR10(root='../_Datasets', train=False,
                                            download=True, transform=test_transform)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                              shuffle=False, num_workers=2)

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_loader, test_loader, classes
