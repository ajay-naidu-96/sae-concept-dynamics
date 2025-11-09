import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
from script_utils.model_trainer import vanilla_sae_trainer, top_k_trainer, vanilla_sae_rand
from script_utils.loader import create_dataloader_from_chunks
from script_utils.utils import set_seed
import glob

def main():

    args = create_argparser().parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    args.device = device 

    print(f"Using device: {args.device}")

    print(f"Loading data from {args.data_dir}...")

    set_seed(args.seed)

    dataloader = create_dataloader_from_chunks(
                    args.data_dir, 
                    device, 
                    args.batch_size)

    first_chunk = sorted(glob.glob(os.path.join(args.data_dir, "*.pt")))[0]
    ckpt = torch.load(first_chunk, map_location=device)
    sample_activations = ckpt.get("fc1_activations_norm", ckpt.get("fc1"))
    input_dim = sample_activations.shape[-1]

    print(f"Input dimension: {input_dim}")
    print("Data loaded successfully.")

    results = {}

    trainer_kwargs = {
        'input_dim': input_dim,
        'args': args,
        'res': results,
        'dataloader': dataloader,
        'reanim': args.reanim
    }

    trainer_func = None

    if args.sae_type == 'random':
        print("Saving randomly initialized SAE's")
        vanilla_sae_rand(input_dim, args)

        return 

    if args.sae_type == 'vanilla':
        print("Training Vanilla SAE...")
        trainer_func = vanilla_sae_trainer
        if args.l1_penalties:
            penalties_dict = {str(c): args.l1_penalties for c in args.nb_concepts_list}
        else:
            print("No L1 penalties provided via --l1_penalties, using default logspace penalties.")
            '''
                str(c): [round(val, 6) for val in np.logspace(-5, 0, num=20)] # 1e-5 up to 1
                str(c): [round(val, 6) for val in np.logspace(-5, 1, num=30)] # 1e−5 up to 10.
                str(c): [round(val, 6) for val in np.logspace(-4, 2, num=30)] # 1e−4 up to 100.
            '''

            penalties_dict = {
                str(c): [round(val, 6) for val in np.logspace(-5, 1, num=30)] for c in args.nb_concepts_list
            }
        trainer_kwargs['penalties'] = penalties_dict

    elif args.sae_type == 'top_k':
        print("Training Top-K SAE...")
        trainer_func = top_k_trainer

    # --- Execute Trainer ---
    if trainer_func:
        trainer_func(**trainer_kwargs)
    else:
        print(f"Error: Unknown SAE type '{args.sae_type}'")
        return

    reanim_str = "_reanim" if args.reanim else ""
    save_path = os.path.join(args.log_dir, f"{args.sae_type}_sae_results{reanim_str}.pt")
    
    print(f"\nSaving results to {save_path}")
    torch.save(results, save_path)
    print("Training complete.")


def create_argparser():
    parser = argparse.ArgumentParser(description="Train a Sparse Autoencoder.")
    
    parser.add_argument('--sae_type', type=str, required=True, choices=['vanilla', 'top_k', 'random'],
                        help="The type of SAE to train.")
    parser.add_argument('--data_dir', type=str, default="./logs/train_activation.pt",
                        help="Path to the file containing the input activations.")
    parser.add_argument('--log_dir', type=str, default="./logs/sae_models/",
                        help="Directory to save trained models and results.")

    parser.add_argument('--lr', type=float, default=1e-3, help="Learning rate.")
    parser.add_argument('--batch_size', type=int, default=1024, help="Batch size for training.")
    parser.add_argument('--epochs', type=int, default=10, help="Number of training epochs.")
    parser.add_argument('--reanim', action='store_true', help="Enable the reanimation of dead neurons during training.")
    parser.add_argument('--device', type=str, default="cuda", help="Device to use for training (e.g., 'cuda', 'cpu').")

    parser.add_argument('--nb_concepts_list', type=int, nargs='+', default=[256, 512],
                        help="List of hidden dimension sizes (number of concepts) to train.")
    parser.add_argument('--l1_penalties', type=float, nargs='+',
                        help="List of L1 penalty coefficients (for vanilla SAE).")
    parser.add_argument('--top_k_ratios', type=float, nargs='+', default = [ 0.01, 0.02, 0.03, 0.04, 0.05, 0.07, 0.09, 0.12, 0.15, 0.18, 0.22, 
                                                                            0.26, 0.32, 0.38, 0.45, 0.55, 0.66, 0.78, 0.87, 0.96],
                        help="List of top-k ratios for activation (for top_k SAE).")
    parser.add_argument('--seed', type=int, default=42, help="Random seed for reproducibility.")


    return parser

if __name__ == "__main__":
    main()
