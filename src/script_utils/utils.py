import argparse
import random
import torch
import numpy as np


def str2bool(v):
    """
    https://stackoverflow.com/questions/15008758/parsing-boolean-values-with-argparse
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "true", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "false", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("boolean value expected")


def add_dict_to_argparser(parser, default_dict):
    for k, v in default_dict.items():
        v_type = type(v)
        if v is None:
            v_type = str
        elif isinstance(v, bool):
            v_type = str2bool
        parser.add_argument(f"--{k}", default=v, type=v_type)


def create_argparser():

    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, choices=["mnist", "cifar10", "vit"], default="mnist")

    parser.add_argument("--data_dir", type=str, default="./Data/")
    parser.add_argument("--log_dir", type=str, default="./logs/")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--name", type=str, default="experiment")

    parser.add_argument("--lr", type=float, default=None)
    parser.add_argument("--batch_size", type=int, default=None)
    parser.add_argument("--epochs", type=int, default=None)

    args = parser.parse_args()

    if args.dataset == "mnist":
        args.lr = args.lr or 5e-3
        args.batch_size = args.batch_size or 256
        args.epochs = args.epochs or 25
        args.name = args.name if args.name != "experiment" else "mnist"
        
    elif args.dataset == "cifar10":
        args.lr = args.lr or 1e-1
        args.batch_size = args.batch_size or 256
        args.epochs = args.epochs or 125
        args.name = args.name if args.name != "experiment" else "cifar10"

    elif args.dataset == "vit":
        args.lr = args.lr or 1e-1
        args.batch_size = args.batch_size or 8
        args.epochs = args.epochs or 50
        args.name = args.name if args.name != "experiment" else "ViTcifar10"


    return args


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
