import torch
import numpy as np
import argparse
import os
import matplotlib.pyplot as plt
from torchvision.utils import make_grid
from tqdm import tqdm
import pandas as pd
from script_utils.loader import setup_mnist_loader, setup_cifar_loader
from models.oracle import MnistCNN, ResNet18CIFAR10

def get_all_activations(oracle, sae, loader, device):
    """
    Gets all oracle and SAE activations for a given data loader.
    """
    oracle.eval()
    sae.eval()

    all_oracle_activations = []
    all_sae_activations = []
    all_images = []

    with torch.no_grad():
        for images, _ in tqdm(loader, desc="Getting Activations"):
            images = images.to(device)
            
            # Get oracle activations
            _, _, oracle_acts = oracle(images)
            
            # Get SAE activations (codes)
            _, sae_acts = sae.encode(oracle_acts)

            all_oracle_activations.append(oracle_acts.cpu())
            all_sae_activations.append(sae_acts.cpu())
            all_images.append(images.cpu())

    return torch.cat(all_images), torch.cat(all_oracle_activations), torch.cat(all_sae_activations)

def analyze_neuron_behavior(sae_activations):
    """
    Analyzes the activation behavior of each neuron across the dataset.
    """
    num_neurons = sae_activations.shape[1]
    
    # Calculate statistics
    max_activations = torch.max(sae_activations, dim=0).values
    min_activations = torch.min(sae_activations, dim=0).values
    mean_activations = torch.mean(sae_activations, dim=0)
    std_activations = torch.std(sae_activations, dim=0)
    
    # Calculate sparsity (percentage of zero or near-zero activations)
    sparsity = (sae_activations.abs() < 1e-6).float().mean(dim=0) * 100
    
    # Count how many samples each neuron activates on (non-zero)
    num_active_samples = (sae_activations.abs() >= 1e-6).sum(dim=0)
    
    neuron_stats = {
        "Neuron Index": list(range(num_neurons)),
        "Min Activation": min_activations.tolist(),
        "Max Activation": max_activations.tolist(),
        "Mean Activation": mean_activations.tolist(),
        "Std Activation": std_activations.tolist(),
        "Sparsity (%)": sparsity.tolist(),
        "Active Samples": num_active_samples.tolist()
    }
    
    df = pd.DataFrame(neuron_stats)
    
    # Save statistics to CSV
    print("\n--- Neuron Activation Statistics ---")
    print(df.head(20).to_string())
    print(f"\n... (showing first 20 of {num_neurons} neurons)")
    
    return df

def select_neurons_to_visualize(neuron_stats_df, num_neurons, method='top_max_activation'):
    """
    Selects a subset of neurons to visualize based on a selection method.
    """
    if method == 'top_max_activation':
        # Sort by 'Max Activation' in descending order and pick the top N
        selected_df = neuron_stats_df.sort_values(by="Max Activation", ascending=False).head(num_neurons)
    elif method == 'top_mean_activation':
        # Sort by mean activation
        selected_df = neuron_stats_df.sort_values(by="Mean Activation", ascending=False).head(num_neurons)
    elif method == 'least_sparse':
        # Select neurons that fire most frequently (least sparse)
        selected_df = neuron_stats_df.sort_values(by="Sparsity (%)", ascending=True).head(num_neurons)
    elif method == 'most_sparse':
        # Select neurons that fire most rarely but still activate
        # Filter out completely dead neurons first
        active_neurons = neuron_stats_df[neuron_stats_df["Active Samples"] > 0]
        selected_df = active_neurons.sort_values(by="Sparsity (%)", ascending=False).head(num_neurons)
    else:
        raise ValueError(f"Unknown neuron selection method: {method}")

    print(f"\nSelected {len(selected_df)} neurons to visualize (method: {method}):")
    print(selected_df.to_string())

    return selected_df

def visualize_top_activating_samples(images, sae_activations, selected_neurons_df, num_samples=16, log_dir="visualizations"):
    """
    Visualizes and saves a grid of top activating samples for specified neurons.
    """
    os.makedirs(log_dir, exist_ok=True)
    
    # Save the statistics CSV
    stats_path = os.path.join(log_dir, "neuron_statistics.csv")
    selected_neurons_df.to_csv(stats_path, index=False)
    print(f"\nSaved neuron statistics to {stats_path}")

    for _, row in selected_neurons_df.iterrows():
        neuron_idx = int(row["Neuron Index"])
        min_act = row["Min Activation"]
        max_act = row["Max Activation"]
        mean_act = row["Mean Activation"]
        sparsity = row["Sparsity (%)"]
        active_samples = int(row["Active Samples"])
        
        print(f"\nVisualizing neuron {neuron_idx}...")
        print(f"  Stats: min={min_act:.4f}, max={max_act:.4f}, mean={mean_act:.4f}, sparsity={sparsity:.2f}%")

        # Get the activations for the current neuron across all samples
        neuron_acts = sae_activations[:, neuron_idx]

        # Ensure we don't request more samples than available
        available_samples = (neuron_acts > 1e-6).sum().item()
        actual_num_samples = min(num_samples, available_samples, len(neuron_acts))
        
        if actual_num_samples == 0:
            print(f"  Warning: Neuron {neuron_idx} has no active samples, skipping.")
            continue

        # Get the indices of the top activating samples
        top_indices = torch.topk(neuron_acts, k=actual_num_samples).indices
        top_activations = neuron_acts[top_indices]

        # Get the corresponding images
        top_images = images[top_indices]

        # Create a grid of images
        nrow = min(4, actual_num_samples)
        grid = make_grid(top_images, nrow=nrow, normalize=True, pad_value=0.5)
        
        # Plot and save the grid
        fig, ax = plt.subplots(figsize=(12, 10))
        ax.imshow(grid.permute(1, 2, 0).numpy())
        
        title_text = (f"Neuron {neuron_idx}: Top {actual_num_samples} Activating Samples\n"
                     f"Activation Range: [{min_act:.4f}, {max_act:.4f}] | "
                     f"Mean: {mean_act:.4f} | Sparsity: {sparsity:.2f}%\n"
                     f"Active on {active_samples} / {len(neuron_acts)} samples")
        ax.set_title(title_text, fontsize=10, pad=10)
        ax.axis("off")
        
        # Add activation values as text below the grid
        activation_text = "Activation values: " + ", ".join([f"{act:.3f}" for act in top_activations[:8]])
        if len(top_activations) > 8:
            activation_text += ", ..."
        fig.text(0.5, 0.02, activation_text, ha='center', fontsize=8)
        
        save_path = os.path.join(log_dir, f"neuron_{neuron_idx:04d}_top_samples.png")
        plt.savefig(save_path, bbox_inches='tight', dpi=150)
        plt.close()
        print(f"  Saved visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Visualize top activating samples for SAE neurons.")
    parser.add_argument("--dataset", type=str, required=True, choices=["mnist", "cifar10"], 
                       help="Dataset to use.")
    parser.add_argument("--oracle_path", type=str, required=True, 
                       help="Path to the trained oracle model.")
    parser.add_argument("--sae_path", type=str, required=True, 
                       help="Path to the trained SAE model.")
    parser.add_argument("--data_dir", type=str, default="./data", 
                       help="Directory for storing the dataset.")
    parser.add_argument("--log_dir", type=str, default="./visualizations", 
                       help="Directory to save the output images.")
    parser.add_argument("--num_neurons", type=int, default=10, 
                       help="Number of neurons to visualize.")
    parser.add_argument("--selection_method", type=str, default="top_max_activation", 
                       choices=["top_max_activation", "top_mean_activation", "least_sparse", "most_sparse"],
                       help="Method to select which neurons to visualize.")
    parser.add_argument("--num_samples", type=int, default=16, 
                       help="Number of top samples to visualize per neuron.")
    parser.add_argument("--batch_size", type=int, default=256, 
                       help="Batch size for processing the data.")
    parser.add_argument("--device", type=str, 
                       default="cuda" if torch.cuda.is_available() else "cpu", 
                       help="Device to use.")
    parser.add_argument("--is_train", action="store_true",
                       help="Use training set instead of test set.")
    args = parser.parse_args()

    # Load dataset
    print(f"\nLoading {args.dataset} dataset...")
    if args.dataset == "mnist":
        loader = setup_mnist_loader(args)
        oracle = MnistCNN()
    elif args.dataset == "cifar10":
        loader = setup_cifar_loader(args)
        oracle = ResNet18CIFAR10()
    else:
        raise ValueError(f"Unsupported dataset: {args.dataset}")

    # Determine which split to use
    split = 'train' if args.is_train else 'test'
    print(f"Using {split} split for visualization")

    # Load models
    print(f"\nLoading oracle model from {args.oracle_path}...")
    oracle.load_state_dict(torch.load(args.oracle_path, map_location=args.device))
    oracle.to(args.device)

    print(f"Loading SAE model from {args.sae_path}...")
    sae = torch.load(args.sae_path, map_location=args.device)
    sae.to(args.device)

    # Get activations
    print(f"\nExtracting activations from {split} set...")
    images, oracle_acts, sae_activations = get_all_activations(
        oracle, sae, loader[split], args.device
    )
    print(f"Extracted activations: {sae_activations.shape}")
    
    # Analyze and select neurons
    neuron_stats_df = analyze_neuron_behavior(sae_activations)
    selected_neurons_df = select_neurons_to_visualize(
        neuron_stats_df, 
        args.num_neurons, 
        args.selection_method
    )

    # Visualize
    print(f"\nGenerating visualizations...")
    visualize_top_activating_samples(
        images, 
        sae_activations, 
        selected_neurons_df,
        args.num_samples, 
        args.log_dir
    )
    
    print(f"\n✓ All visualizations saved to {args.log_dir}")

if __name__ == "__main__":
    main()