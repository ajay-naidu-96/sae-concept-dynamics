import torch
import numpy as np
from utils import create_argparser, set_seed
from model import MnistCNN
from train_util import TrainLoop, ActivationLogger
import os
import torch.nn as nn
from loader import setup_mnist_loader

def main():

    args = create_argparser().parse_args()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    set_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    loader = setup_mnist_loader(args)

    model = MnistCNN()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    TrainLoop(model, loader, optimizer, criterion, device, args).run()
    
    best_model = MnistCNN()

    best_model.load_state_dict(torch.load(os.path.join(args.log_dir, "best_model.pt")))

    ActivationLogger(best_model, loader, device, args).run()


if __name__ == "__main__":
    main()

