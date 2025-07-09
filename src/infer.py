import torch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
import argparse
from utils import add_dict_to_argparser
import os
import pandas as pd
from metrics_calc import extract_purity_scores
from logistic_regression_classifier import evaluate_sae_activations, evaluate_per_neuron, evaluate_per_neuron_1vsall
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
    fc1_activations = ckpt["fc1_activations"]
    labels = ckpt["labels"]

    dataloader = torch.utils.data.DataLoader(TensorDataset(fc1), batch_size=1024, shuffle=False)
    all_results = {}

    for model_flavor in os.listdir(args.log_dir):
        
        if model_flavor.endswith('.pt'):
            continue

        best_loss = os.path.join(args.log_dir, model_flavor, "best_loss.pth")
        best_both = os.path.join(args.log_dir, model_flavor, "best_both.pth")

        dead_ones = [model_type for model_type in os.listdir(os.path.join(args.log_dir, model_flavor)) if model_type.startswith("best_dead")]

        sorted(dead_ones)

        best_dead = os.path.join(args.log_dir, model_flavor, dead_ones[-1])

        model_paths = [best_loss, best_both, best_dead]

        for each_path in model_paths:
            
            sae_name = each_path.split('/')[-2] + "_" + each_path.split('/')[-1].split('.')[0]

            sae = torch.load(each_path)
            sae.to(device)
            sae.eval()

            with torch.no_grad():

                pre_codes, codes = sae.encode(fc1.to(device))
                activations = codes.cpu()

                results = evaluate_per_neuron_1vsall(activations, labels.cpu(), sae_name)
                all_results[sae_name] = results
    
    # results_to_dataframe(all_results, 'sae_lr_results_per_neuron.csv')
    output_path = os.path.join(args.log_dir, "all_results_vanilla_sae.pkl")
    with open(output_path, "wb") as f:
        pickle.dump(all_results, f)
    
    # results = []

    # for idx, model_name in enumerate(os.listdir(args.log_dir)):

    #     print(f"Running: {idx}, {model_name}")

    #     checkpoint = torch.load(args.log_dir+model_name)

    #     sae = checkpoint['full_model']
        
    #     sae.to(device)
    #     sae = sae.eval()

    #     with torch.no_grad():
    #         pre_codes, codes = sae.encode(fc1.to(device))
            
    #         activations = codes.cpu()
            
    #         res = extract_purity_scores(activations, labels)

    #         res['reanimate'] = 1 if "reanimate" in model_name else 0

    #         if model_name.startswith("top_k_sae"):
    #             res['model_name'] = "top_k_sae"
    #             res['nb_concepts'] = model_name.split('.')[0].split('_')[-2]
    #             res['top_k'] = model_name.split('.')[0].split('_')[-1]
    #         elif model_name.startswith("batch_top_k"):
    #             res['model_name'] = "batch_top_k"
    #             res['nb_concepts'] = model_name.split('.')[0].split('_')[-3]
    #             res['top_k'] = model_name.split('.')[0].split('_')[-2]

    #         res['avg_final_loss'] = checkpoint['logs']['avg_loss'][-1]
    #         res['final_dead_features'] = checkpoint['logs']['dead_features'][-1]

    #         results.append(res)

    # df_res = pd.DataFrame(results)
    # df_res.to_csv("results_norm.csv")

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