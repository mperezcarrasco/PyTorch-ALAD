import torch
import numpy as np
from torch.utils import data
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from PIL import Image


class SVHN_loader(data.Dataset):
    """This class is needed to processing batches for the dataloader."""
    def __init__(self, data, target, transform=False):
        self.data = data
        self.target = target
        self.transform = transform

    def __getitem__(self, index):
        """return transformed items."""
        x = self.data[index]
        y = self.target[index]
        if self.transform:
            x = Image.fromarray(x)
            x = self.transform(x)
        return x, y

    def __len__(self):
        """number of samples."""
        return len(self.data)


def get_svhn(args, data_dir='./data/svhn/'):
    """get dataloders"""

    transform = transforms.Compose([transforms.ToTensor(),
                                    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    train = datasets.SVHN(root=data_dir, split='train', download=True)

    data = train.data.transpose(0,2,3,1)
    labels = train.labels
    
    normal_data = data[labels==args.normal_class]
    normal_labels = labels[labels==args.normal_class]
    anormal_data = data[labels!=args.normal_class]
    anormal_labels = labels[labels!=args.normal_class]
    
    N_train = int(normal_data.shape[0]*0.8)
    
    x_train = normal_data[:N_train]
    y_train = normal_labels[:N_train]
    data_train = SVHN_loader(x_train, y_train, transform=transform)
    dataloader_train = DataLoader(data_train, batch_size=args.batch_size, 
                                  shuffle=True, num_workers=0)
    
    x_test = np.concatenate((anormal_data, normal_data[N_train:]), axis=0) 
    y_test = np.concatenate((anormal_labels, normal_labels[N_train:]), axis=0)
    y_test = np.where(y_test==args.normal_class, 0, 1)
    data_test = SVHN_loader(x_test, y_test, transform=transform)
    dataloader_test = DataLoader(data_test, batch_size=args.batch_size, 
                                 shuffle=True, num_workers=0)
    return dataloader_train, dataloader_test


