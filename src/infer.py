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

def main():

    args = create_argparser().parse_args()
    device = torch.device("cuda")

    dataloader = create_dataloader_with_labels(
                    args.data_dir, 
                    device, 
                    args.batch_size)

    # all_results = {}
    all_stable_rank_results = {}
    # all_concept_paths = {}

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

            concept_paths_dir = os.path.join(args.log_dir, model_flavor, "concept_paths")
            os.makedirs(concept_paths_dir, exist_ok=True)

            # Collect all activations and labels from the dataloader
            all_activations = []
            all_labels = []
            
            with torch.no_grad():
                for batch_data in dataloader:
                    if len(batch_data) == 2:
                        fc1_batch, labels_batch = batch_data
                    else:
                        # If dataloader only returns activations
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
                else:
                    If no labels available, you might need to handle this case
                    depending on your data structure
                    labels = None

                concept_paths_configs = [
                    {'method': 'top_k', 'value': 10, 'min_act': 0.1},
                    {'method': 'top_k', 'value': 20, 'min_act': 0.1},
                    {'method': 'percentile', 'value': 90, 'min_act': 0.1},
                    {'method': 'std_threshold', 'value': 1.5, 'min_act': 0.1}
                ]

                if labels is not None:
                    results = evaluate_per_neuron_1vsall_new(activations, labels, sae_name)
                    all_results[sae_name] = results
                else:
                    print(f"Warning: No labels found for {sae_name}, skipping evaluation")

                stable_rank_results = evaluate_sae_stable_rank(
                    sae, 
                    sae_name,
                    thresholds=[0.001, 0.01, 0.05, 0.1], 
                    include_both_matrices=True, 
                    use_frobenius_definition = True
                )
                
                all_stable_rank_results[sae_name] = stable_rank_results

                sae_concept_paths = {}
                
                for config in concept_paths_configs:
                    config_name = f"{config['method']}_{config['value']}"
                    print(f"Extracting concept paths with {config_name}...")
                    
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
                    overlap_analysis = extractor.analyze_concept_overlap(concept_paths_data)
                    concept_paths_data['overlap_analysis'] = overlap_analysis
                    
                    sae_concept_paths[config_name] = concept_paths_data
                    
                    print(f"  Extracted {len(concept_paths_data['concept_paths'])} concept paths")
                    print(f"  Mean active neurons: {concept_paths_data['summary']['mean_active_neurons']:.2f}")
                    print(f"  Total unique neurons used: {concept_paths_data['summary']['total_unique_neurons_used']}")
                
                all_concept_paths[sae_name] = sae_concept_paths

            del sae, activations, all_activations

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

        output_path = os.path.join(args.log_dir, args.output_path)

        with open(output_path, "wb") as f:
            pickle.dump(all_results, f)

def create_argparser():

    defaults = dict(
        data_dir="./logs/activation.pt",
        log_dir= "./logs_topk_sae_/",
        output_path="concept_path_results.pkl",
        batch_size=1024
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser

if __name__ == "__main__":
    main()