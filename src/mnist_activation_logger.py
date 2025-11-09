import torch
import numpy as np
from script_utils.utils import create_argparser, set_seed
from models.oracle import MnistCNN, ResNet18CIFAR10, ViTWithActivations
from script_utils.train_util import TrainLoop, ActivationLogger
import os
import torch.nn as nn
from script_utils.loader import setup_mnist_loader, setup_cifar_loader


def main():

    args = create_argparser()

    os.makedirs(args.data_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.dataset == "mnist":
        loader = setup_mnist_loader(args)
        model = MnistCNN()

    elif args.dataset == "cifar10":
        loader = setup_cifar_loader(args)   
        model = ResNet18CIFAR10()   

    elif args.dataset == "vit":
            loader = setup_cifar_loader(args)   
            print("Using Vision Transformer (ViT) model for CIFAR-10.")
            
            model = ViTWithActivations(num_classes=10) 
            state_dict = torch.hub.load_state_dict_from_url(
                    "https://huggingface.co/edadaltocg/vit_base_patch16_224_in21k_ft_cifar10/resolve/main/pytorch_model.bin",
                    map_location="cpu",
                    file_name="vit_base_patch16_224_in21k_ft_cifar10.pth",
                )
            model.load_state_dict(state_dict)

    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    # optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    optimizer = torch.optim.SGD(model.parameters(), lr=args.lr, momentum=0.9, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    # criterion = nn.CrossEntropyLoss()
    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    TrainLoop(model, loader, optimizer, criterion, device, args, scheduler).run()

    if args.dataset == "mnist":
        best_model = MnistCNN()

    elif args.dataset == "cifar10": 
        best_model = ResNet18CIFAR10()

    elif args.dataset == "vit":
        best_model = model
    
    best_model.load_state_dict(torch.load(os.path.join(args.log_dir, "oracle.pt")))
    best_model = best_model.to(device)

    ActivationLogger(best_model, loader, device, args).run()

if __name__ == "__main__":
    main()
