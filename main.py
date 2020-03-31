import numpy as np
import argparse 
import torch

from train import ALADTrainer
from preprocess import get_svhn


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_epochs", type=int, default=100,
                        help="number of epochs")
    parser.add_argument('--lr', type=float, default=0.0002,
                        help='learning rate')
    parser.add_argument("--batch_size", type=int, default=32, 
                        help="Batch size")
    parser.add_argument('--latent_dim', type=int, default=100,
                        help='Dimension of the latent variable z')
    parser.add_argument('--anormal_class', type=int, default=0,
                        help='Class to be treated as normal class.')
    parser.add_argument('--pretrained', type=bool, default=False,
                        help='If is there a pretrained model.')
    parser.add_argument('--spec_norm', type=bool, default=False,
                        help='If is there a pretrained model.')
    #parsing arguments.
    args = parser.parse_args() 

    #check if cuda is available.
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    data = get_svhn(args)

    egbad = ALADTrainer(args, data, device)
    egbad.train()

