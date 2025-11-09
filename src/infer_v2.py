import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import argparse
from script_utils.utils import add_dict_to_argparser
import os
import pandas as pd
from metrics.metrics_calc import extract_purity_scores
from metrics.logistic_regression_classifier import evaluate_sae_activations, evaluate_per_neuron, evaluate_per_neuron_1vsall_new
from metrics.sae_stable_rank import evaluate_sae_stable_rank
from metrics.concept_path import ConceptPathExtractor
import pickle
from script_utils.loader import create_dataloader_with_labels
import glob


def run_regression_evaluation(sae, sae_name, dataloader, device):
    """Run logistic regression evaluation on SAE activations."""
    print(f"Running regression evaluation for {sae_name}...")
    
    all_activations = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 2:
                fc1_batch, labels_batch = batch_data
            else:
                fc1_batch = batch_data[0]
                labels_batch = None
            
            pre_codes, codes = sae.encode(fc1_batch)
            all_activations.append(codes.cpu())
            
            if labels_batch is not None:
                all_labels.append(labels_batch.cpu())
    
    # Concatenate all batches
    activations = torch.cat(all_activations, dim=0)
    
    if all_labels:
        labels = torch.cat(all_labels, dim=0)
        results = evaluate_per_neuron_1vsall_new(activations, labels, sae_name)
        return results
    else:
        print(f"Warning: No labels found for {sae_name}, skipping regression evaluation")
        return None


def run_stable_rank_evaluation(sae, sae_name, args):
    """Run stable rank evaluation on SAE."""
    print(f"Running stable rank evaluation for {sae_name}...")
    
    stable_rank_results = evaluate_sae_stable_rank(
        sae, 
        sae_name,
        thresholds=args.stable_rank_thresholds, 
        include_both_matrices=args.include_both_matrices, 
        use_frobenius_definition=args.use_frobenius_definition
    )
    
    return stable_rank_results


def run_concept_path_evaluation(sae, sae_name, dataloader, device, args, model_flavor, log_dir):
    """Run concept path evaluation on SAE activations."""
    print(f"Running concept path evaluation for {sae_name}...")
    
    concept_paths_dir = os.path.join(log_dir, model_flavor, "concept_paths")
    os.makedirs(concept_paths_dir, exist_ok=True)
    
    all_activations = []
    all_labels = []
    
    with torch.no_grad():
        for batch_data in dataloader:
            if len(batch_data) == 2:
                fc1_batch, labels_batch = batch_data
            else:
                fc1_batch = batch_data[0]
                labels_batch = None
            
            pre_codes, codes = sae.encode(fc1_batch)
            all_activations.append(codes.cpu())
            
            if labels_batch is not None:
                all_labels.append(labels_batch.cpu())
    
    # Concatenate all batches
    activations = torch.cat(all_activations, dim=0)
    
    if not all_labels:
        print(f"Warning: No labels found for {sae_name}, skipping concept path evaluation")
        return None
    
    labels = torch.cat(all_labels, dim=0)
    
    concept_paths_configs = [
        {'method': 'top_k', 'value': 10, 'min_act': 1e-5},
        {'method': 'top_k', 'value': 20, 'min_act': 1e-5},
        {'method': 'percentile', 'value': 90, 'min_act': 1e-5},
        {'method': 'std_threshold', 'value': 1.5, 'min_act': 1e-5}
    ]
    
    sae_concept_paths = {}
    
    for config in concept_paths_configs:
        config_name = f"{config['method']}_{config['value']}"
        print(f"\nExtracting concept paths with {config_name}...")
        
        extractor = ConceptPathExtractor(
            threshold_method=config['method'],
            threshold_value=config['value'],
            min_activation=config['min_act']
        )
        
        concept_paths_data = extractor.extract_concept_paths(
            activations, labels, f"{sae_name}_{config_name}"
        )
        
        # Save individual config results
        save_path = os.path.join(concept_paths_dir, f"{sae_name}_{config_name}.pkl")
        extractor.save_concept_paths(concept_paths_data, save_path)
        
        # Analyze concept overlap between classes
        print("  Analyzing between-class overlap...")
        overlap_analysis = extractor.analyze_concept_overlap(concept_paths_data)
        
        # Analyze random pair overlap
        print("  Analyzing random pair overlap...")
        random_overlap_analysis = extractor.analyze_random_pair_overlap(concept_paths_data, n_samples=10000)
        
        # Analyze within-class pair overlap
        print("  Analyzing within-class pair overlap...")
        within_class_overlap_analysis = extractor.analyze_within_class_pair_overlap(concept_paths_data, n_samples=10000)
        
        # Store all analyses
        concept_paths_data['overlap_analysis'] = overlap_analysis
        concept_paths_data['random_pair_overlap_analysis'] = random_overlap_analysis
        concept_paths_data['within_class_overlap_analysis'] = within_class_overlap_analysis
        
        sae_concept_paths[config_name] = concept_paths_data
        
        # Print summary
        print(f"  Extracted {len(concept_paths_data['concept_paths'])} concept paths")
        print(f"  Mean active neurons: {concept_paths_data['summary']['mean_active_neurons']:.2f}")
        print(f"  Total unique neurons used: {concept_paths_data['summary']['total_unique_neurons_used']}")
        
        # Print overlap summaries
        if 'random_pair_overlap_analysis' in concept_paths_data and 'error' not in random_overlap_analysis:
            print(f"  Random pair overlap:")
            print(f"    Avg shared neurons: {random_overlap_analysis['avg_shared_neurons']:.2f}")
            print(f"    Avg Jaccard similarity: {random_overlap_analysis['avg_jaccard_similarity']:.4f}")
        
        if 'within_class_overlap_analysis' in concept_paths_data and 'error' not in within_class_overlap_analysis:
            print(f"  Within-class overlap:")
            print(f"    Avg shared neurons: {within_class_overlap_analysis['overall']['avg_shared_neurons']:.2f}")
            print(f"    Avg Jaccard similarity: {within_class_overlap_analysis['overall']['avg_jaccard_similarity']:.4f}")
        
        if 'overlap_analysis' in concept_paths_data and 'summary' in overlap_analysis:
            print(f"  Between-class overlap:")
            print(f"    Mean Jaccard similarity: {overlap_analysis['summary']['mean_jaccard']:.4f}")
            print(f"    Mean shared neurons: {overlap_analysis['summary']['mean_shared_neurons']:.2f}")
    
    return sae_concept_paths


def main():
    args = create_argparser().parse_args()
    device = torch.device("cuda")

    # Check if at least one evaluation type is selected
    if not any([args.run_regression, args.run_stable_rank, args.run_concept_path]):
        print("Error: At least one evaluation type must be selected (--run_regression, --run_stable_rank, or --run_concept_path)")
        return

    # Only create dataloader if needed for regression or concept path evaluation
    dataloader = None
    if args.run_regression or args.run_concept_path:
        dataloader = create_dataloader_with_labels(
            args.data_dir, 
            device, 
            args.batch_size
        )

    # Initialize result dictionaries
    all_results = {}
    all_stable_rank_results = {}
    all_concept_paths = {}

    os.makedirs(args.log_dir, exist_ok=True)

    for i, model_flavor in enumerate(os.listdir(args.log_dir)):
        
        if not os.path.isdir(os.path.join(args.log_dir, model_flavor)):
            continue

        best_loss = os.path.join(args.log_dir, model_flavor, "best_loss.pth")

        if not os.path.isfile(best_loss):
            continue

        model_paths = [best_loss]

        for each_path in model_paths:
            
            sae_name = each_path.split('/')[-2] + "_" + each_path.split('/')[-1].split('.')[0]

            print(f"Loading: {model_flavor}, {sae_name}")

            sae = torch.load(each_path, map_location=device)
            sae.eval()

            # Run selected evaluations
            if args.run_regression:
                regression_results = run_regression_evaluation(sae, sae_name, dataloader, device)
                if regression_results is not None:
                    all_results[sae_name] = regression_results

            if args.run_stable_rank:
                stable_rank_results = run_stable_rank_evaluation(sae, sae_name, args)
                all_stable_rank_results[sae_name] = stable_rank_results

            if args.run_concept_path:
                concept_path_results = run_concept_path_evaluation(
                    sae, sae_name, dataloader, device, args, model_flavor, args.log_dir
                )
                if concept_path_results is not None:
                    all_concept_paths[sae_name] = concept_path_results

            # Clean up memory
            del sae
            if args.run_regression or args.run_concept_path:
                # Only delete these variables if they were created
                if 'activations' in locals():
                    del activations
                if 'all_activations' in locals():
                    del all_activations

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # Save results based on what was run
    if args.run_regression and all_results:
        regression_output_path = os.path.join(args.log_dir, f"regression_{args.output_path}")
        with open(regression_output_path, "wb") as f:
            pickle.dump(all_results, f)
        print(f"Regression results saved to: {regression_output_path}")

    if args.run_stable_rank and all_stable_rank_results:
        stable_rank_output_path = os.path.join(args.log_dir, f"stable_rank_{args.output_path}")
        with open(stable_rank_output_path, "wb") as f:
            pickle.dump(all_stable_rank_results, f)
        print(f"Stable rank results saved to: {stable_rank_output_path}")

    if args.run_concept_path and all_concept_paths:
        concept_path_output_path = os.path.join(args.log_dir, f"rand_concept_path_{args.output_path}")
        with open(concept_path_output_path, "wb") as f:
            pickle.dump(all_concept_paths, f)
        print(f"Concept path results saved to: {concept_path_output_path}")


def create_argparser():
    # Only include arguments that won't conflict with explicit parser.add_argument calls
    defaults = dict(
        data_dir="./logs/activation.pt",
        log_dir="./logs_topk_sae_/",
        output_path="results.pkl",
        batch_size=1024,
    )

    parser = argparse.ArgumentParser(
        description="Modular SAE evaluation script with selective evaluation types"
    )
    
    # Add evaluation type flags
    parser.add_argument('--run_regression', action='store_true', 
                       help='Run logistic regression evaluation')
    parser.add_argument('--run_stable_rank', action='store_true', 
                       help='Run stable rank evaluation')
    parser.add_argument('--run_concept_path', action='store_true', 
                       help='Run concept path evaluation')
    
    # Add stable rank specific arguments
    parser.add_argument('--stable_rank_thresholds', nargs='+', type=float, 
                       default=[0.001, 0.01, 0.05, 0.1],
                       help='Thresholds for stable rank evaluation')
    parser.add_argument('--include_both_matrices', action='store_true', 
                       default=True,
                       help='Include both encoder and decoder matrices in stable rank')
    parser.add_argument('--use_frobenius_definition', action='store_true', 
                       default=True,
                       help='Use Frobenius definition for stable rank')

    # Add remaining basic arguments from defaults
    add_dict_to_argparser(parser, defaults)
    return parser


if __name__ == "__main__":
    main()