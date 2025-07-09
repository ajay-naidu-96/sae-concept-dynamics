from .loss import mse_criterion, reanimate_criterion, mse_l1_criterion, reanim_mse_l1_criterion
from overcomplete.sae.base import SAE
from overcomplete.sae import TopKSAE, BatchTopKSAE, train_sae
from functools import partial
import torch
import os

def vanilla_sae_trainer(input_dim, args, res, dataloader, penalties, reanim=False):

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