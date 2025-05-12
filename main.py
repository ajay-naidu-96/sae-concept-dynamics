import torch
import torchvision
from torchvision import transforms
from torch.utils.data import DataLoader
import numpy as np
from utils import create_argparser, set_seed
from model import MnistCNN
from train_util import TrainLoop
import os
import torch.nn as nn


def main():

    args = create_argparser().parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    set_seed(args.seed)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    train_dataset = torchvision.datasets.MNIST(root=args.data_dir, train=True, transform=transform, download=True)
    loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

    model = MnistCNN()
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    TrainLoop(model, loader, optimizer, criterion, device, args).run()


if __name__ == "__main__":
    main()

