from .loss import mse_criterion, reanimate_criterion, mse_l1_criterion, reanim_mse_l1_criterion
from overcomplete.sae.base import SAE
from overcomplete.sae import TopKSAE, BatchTopKSAE, train_sae
from functools import partial
import torch
import torch.nn as nn
import os
import numpy as np
import random
from collections import defaultdict

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def vanilla_sae_rand(input_dim, args):
    init_methods = {
        "xavier": lambda m: torch.nn.init.xavier_uniform_(m.weight),
        "kaiming": lambda m: torch.nn.init.kaiming_uniform_(m.weight, nonlinearity="relu"),
        "orthogonal": lambda m: torch.nn.init.orthogonal_(m.weight)
    }

    for init_name, init_func in init_methods.items():
        seed = args.seed + hash(init_name) % 1000  # unique, reproducible seed per init
        set_seed(seed)

        for nb_concepts in args.nb_concepts_list:
            sae = SAE(input_dim, nb_concepts=nb_concepts, device=args.device)

            def init_weights(m):
                if isinstance(m, nn.Linear):
                    init_func(m)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

            sae.apply(init_weights)

            expt_name = f"{init_name}_sae_{nb_concepts}_seed{seed}"
            os.makedirs(args.log_dir, exist_ok=True)
            model_save_path = os.path.join(args.log_dir, expt_name + ".pt")
            torch.save(sae, model_save_path)

            print(f"[{init_name}] Saved SAE with {nb_concepts} concepts (seed={seed}) → {model_save_path}")


def vanilla_sae_trainer(input_dim, args, res, dataloader, penalties, 
                        reanim=False):

    for nb_concepts in args.nb_concepts_list:

        for i, pen in enumerate(penalties[str(nb_concepts)]):
            
            criterion_fn = None
            expt_name = ""

            if reanim:
                criterion_fn = partial(reanim_mse_l1_criterion, l1_coefficient=pen)
                expt_name = "reanim_"
            else:
                criterion_fn = partial(mse_l1_criterion, l1_coefficient=pen)

            sae = SAE(input_dim, nb_concepts=nb_concepts, device=args.device)
                
            optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

            expt_name = expt_name + f"vanilla_sae_{nb_concepts}_{i}"

            print(f"Experiment: Model: {expt_name}, Concepts: {nb_concepts}, Pen: {pen}")

            model_save_path = os.path.join(args.log_dir, expt_name)

            logs = train_sae(sae, dataloader, criterion_fn, optimizer, 
                nb_epochs=args.epochs, device=args.device, save_best=True, log_dir=model_save_path)
            
            res[expt_name] = logs


def top_k_trainer(input_dim, args, res, dataloader, reanim=False):
            
    for nb_concepts in args.nb_concepts_list:

        for rate in args.top_k_ratios:

            top_k = max(1, int(rate * nb_concepts))

            criterion_fn = None
            expt_name = ""

            if reanim: 
                criterion_fn = reanimate_criterion
                expt_name = "reanim_"
            else:
                criterion_fn = mse_criterion

            expt_name = expt_name + f"topk_sae_{nb_concepts}_{top_k}"

            print(f"Experiment: Model: {expt_name}, Concepts: {nb_concepts}, Rate: {rate}, {top_k}")

            sae = TopKSAE(input_dim, nb_concepts=nb_concepts, top_k=top_k, device=args.device)

            optimizer = torch.optim.Adam(sae.parameters(), lr=args.lr)

            model_save_path = os.path.join(args.log_dir, expt_name)

            logs = train_sae(sae, dataloader, criterion_fn, optimizer, 
                nb_epochs=args.epochs, device=args.device, save_best=True, log_dir=model_save_path)

            res[expt_name] = logs