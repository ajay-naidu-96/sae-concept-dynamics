import torch
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
import argparse
import os
from script_utils.model_trainer import vanilla_sae_trainer, top_k_trainer


def main():

    args = create_argparser().parse_args()

    os.makedirs(args.log_dir, exist_ok=True)

    device = torch.device(args.device if args.device == "cuda" and torch.cuda.is_available() else "cpu")
    args.device = device 

    print(f"Using device: {args.device}")

    print(f"Loading data from {args.data_dir}...")

    try:
        ckpt = torch.load(args.data_dir, map_location=args.device)
        activations = ckpt.get("fc1_activations_norm", ckpt.get("fc1"))

        if activations is None:
            raise KeyError("Data file must contain 'fc1' or 'fc1_activations_norm' key.")

    except FileNotFoundError:
        print(f"Error: Data file not found at {args.data_dir}")
        return

    except KeyError as e:
        print(f"Error: {e}")
        return

    dataloader = DataLoader(TensorDataset(activations), batch_size=args.batch_size, shuffle=True)
    print("Data loaded successfully.")

    results = {}
    input_dim = activations.shape[-1]

    trainer_kwargs = {
        'input_dim': input_dim,
        'args': args,
        'res': results,
        'dataloader': dataloader,
        'reanim': args.reanim
    }

    trainer_func = None

    if args.sae_type == 'vanilla':
        print("Training Vanilla SAE...")
        trainer_func = vanilla_sae_trainer
        if args.l1_penalties:
            penalties_dict = {str(c): args.l1_penalties for c in args.nb_concepts_list}
        else:
            print("No L1 penalties provided via --l1_penalties, using default logspace penalties.")
            penalties_dict = {
                str(c): [round(val, 6) for val in np.logspace(-5, 0, num=20)]
                for c in args.nb_concepts_list
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
    
    parser.add_argument('--sae_type', type=str, required=True, choices=['vanilla', 'top_k'],
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
    parser.add_argument('--top_k_ratios', type=float, nargs='+', default=[0.02, 0.05, 0.1, 0.2, 0.5, 0.75, 0.95],
                        help="List of top-k ratios for activation (for top_k SAE).")

    return parser

if __name__ == "__main__":
    main()
