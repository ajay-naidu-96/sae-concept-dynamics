import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import argparse
from script_utils.utils import add_dict_to_argparser
import os
import pandas as pd
from metrics.metrics_calc import extract_purity_scores
from metrics.logistic_regression_classifier import evaluate_sae_activations, evaluate_per_neuron, evaluate_per_neuron_1vsall_new
import pickle

def results_to_dataframe(all_results, save_path=None):

    records = []
    for sae_name, results in all_results.items():
        record = {
            'SAE_Name': sae_name,
            'Train_Accuracy': results['train_accuracy'],
            'Test_Accuracy': results['test_accuracy'],
            'CV_Mean': results['cv_mean'],
            'CV_Std': results['cv_std'],
            'N_Features': results['n_features'],
            'N_Samples': results['n_samples']
        }
        records.append(record)
    
    df = pd.DataFrame(records)
    
    df = df.sort_values('Test_Accuracy', ascending=False).reset_index(drop=True)
    
    df['Rank'] = df.index + 1
    
    column_order = ['Rank', 'SAE_Name', 'Test_Accuracy', 'CV_Mean', 'CV_Std', 
                   'Train_Accuracy', 'N_Features', 'N_Samples']

    df = df[column_order]
    
    numerical_cols = ['Test_Accuracy', 'CV_Mean', 'CV_Std', 'Train_Accuracy']
    df[numerical_cols] = df[numerical_cols].round(4)
    
    if save_path:
        df.to_csv(save_path, index=False)
        print(f"\nResults saved to: {save_path}")

def main():

    args = create_argparser().parse_args()
    device = torch.device("cuda")
    ckpt = torch.load(args.data_dir)

    fc1 = ckpt["fc1"]
    fc1 = fc1.to(device)
    # fc1 = ckpt["fc1"]
    # fc1 = torch.randn_like(c1).to(device)
    fc1_activations = ckpt["fc1_activations_norm"]
    labels = ckpt["labels"]

    dataloader = torch.utils.data.DataLoader(TensorDataset(fc1), batch_size=1024, shuffle=False)
    all_results = {}

    os.makedirs(args.log_dir, exist_ok=True)

    for i in range(1):
        
        model_paths = {
            'xavier_256': '/home/ag4077/Playground/sae-concept-dynamics/logs_random_sae_new/xavier_sae_256_seed203.pt',
            'xavier_512': '/home/ag4077/Playground/sae-concept-dynamics/logs_random_sae_new/xavier_sae_512_seed203.pt',

            # 'kaiming_256': '/home/ag4077/Playground/sae-concept-dynamics/logs_random_sae_new/kaiming_sae_256_seed405.pt',
            # 'kaiming_512': '/home/ag4077/Playground/sae-concept-dynamics/logs_random_sae_new/kaiming_sae_512_seed405.pt',

            # 'orthogonal_256': '/home/ag4077/Playground/sae-concept-dynamics/logs_random_sae_new/orthogonal_sae_256_seed862.pt',
            # 'orthogonal_512': '/home/ag4077/Playground/sae-concept-dynamics/logs_random_sae_new/orthogonal_sae_512_seed862.pt',
        }

        for sae_name, each_path in model_paths.items():
            
            print(f"Loading: {sae_name}")

            sae = torch.load(each_path, map_location=device)
            sae.eval()

            with torch.no_grad():

                pre_codes, codes = sae.encode(fc1)
                activations = codes.cpu()

                evaluate_sae_activations(
                    activations, 
                    labels.cpu(), 
                    sae_name
                )

            del sae, pre_codes, codes, activations

            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def create_argparser():

    defaults = dict(
        data_dir="./logs/activation.pt",
        log_dir= "./logs_vanilla_sae_/",
        batch_size=1024
    )

    parser = argparse.ArgumentParser()
    add_dict_to_argparser(parser, defaults)

    return parser

if __name__ == "__main__":
    main()