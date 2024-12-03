from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor, Normalize, Compose
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import torch


def get_mnist(data_path: str = './data'):

    # tr = Compose([ToTensor(),Normalize((0.1307,),(0.3081,))])
    tr = transforms.Compose([transforms.Resize((224, 224)),
                                transforms.ToTensor(),
                                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # testset = MNIST(data_path, train=False, download=True, transform=tr)
    testset= datasets.ImageFolder(root='D:\Fifth_sem\MINI PROJECT\Trial-code\Validation', transform=tr)

    return testset

def prepare_dataset(num_partitions: int,
                     batch_size: int,
                       val_ratio: float = 0.1):
    
    testset = get_mnist()

    #split trainsert into 'num_partitions' trainset

    #create dataloaders

    # trainloaders = []
    # valloaders = []
    # for trainset_ in trainsets:
    #     num_total = len(trainset_)
    #     num_val = int(val_ratio * num_total)
    #     num_train = num_total - num_val

    #     for_train, for_val = random_split(trainset_,[num_train,num_val], torch.Generator().manual_seed(2023))

    #     trainloaders.append(DataLoader(for_train, batch_size=batch_size, shuffle=True, num_workers=2))
    #     valloaders.append(DataLoader(for_val, batch_size=batch_size, shuffle=False, num_workers=2))
    
    testloaders = DataLoader(testset, batch_size=128)

    return testloaders







