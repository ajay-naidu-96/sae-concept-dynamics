import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from typing import Dict, List
import pandas as pd
from sklearn.metrics import roc_auc_score
from scipy.stats import entropy

def evaluate_per_neuron_1vsall(activations, labels, sae_name="SAE", selective_auc_threshold=0.85):

    if hasattr(activations, 'cpu'):
        activations = activations.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()

    n_neurons = activations.shape[1]
    n_classes = len(np.unique(labels))
    per_neuron_digit_auc = []

    for neuron_idx in range(n_neurons):
        z_i = activations[:, neuron_idx:neuron_idx+1]
        scaler = StandardScaler()
        z_i_scaled = scaler.fit_transform(z_i)

        for digit in range(n_classes):
            binary_labels = (labels == digit).astype(int)

            try:
                clf = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
                clf.fit(z_i_scaled, binary_labels)
                prob = clf.predict_proba(z_i_scaled)[:, 1]
                auc = roc_auc_score(binary_labels, prob)
            except Exception:
                auc = 0.5  

            per_neuron_digit_auc.append({
                "neuron_idx": neuron_idx,
                "digit": digit,
                "auc": auc
            })

    auc_df = pd.DataFrame(per_neuron_digit_auc)
    auc_matrix = auc_df.pivot(index='neuron_idx', columns='digit', values='auc').fillna(0)

    best_digit = auc_matrix.idxmax(axis=1)
    best_auc = auc_matrix.max(axis=1)

    neuron_stats = pd.DataFrame({
        "neuron_idx": auc_matrix.index,
        "best_digit": best_digit,
        "best_auc": best_auc,
        "is_selective": best_auc > selective_auc_threshold
    })

    digit_counts = neuron_stats['best_digit'].value_counts()

    sae_summary = {
        "sae_name": sae_name,
        "total_neurons": len(neuron_stats),
        "digit_coverage": digit_counts.count(),
        "redundancy_score": digit_counts.max(),
        "fraction_selective": neuron_stats['is_selective'].mean(),
        "mean_best_auc": best_auc.mean(),
        "top_10_auc_mean": best_auc.sort_values(ascending=False).head(10).mean(),
        "entropy_best_class_assignments": entropy(digit_counts.values + 1e-6)
    }

    return {
        "auc_matrix": auc_matrix,
        "per_neuron_df": neuron_stats,
        "sae_summary": sae_summary
    }


def evaluate_per_neuron_1vsall_test(activations, activations_test, labels, labels_test):
    if hasattr(activations, 'cpu'):
        activations = activations.cpu().numpy()
        activations_test = activations_test.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
        labels_test = labels_test.cpu().numpy()  # Fixed: was overwriting labels
    
    n_neurons = activations.shape[1]
    n_classes = len(np.unique(labels))
    per_neuron_digit_auc = []
    
    for neuron_idx in range(n_neurons):
        z_i = activations[:, neuron_idx:neuron_idx+1]
        z_i_test = activations_test[:, neuron_idx:neuron_idx+1]
        
        # Fit scaler on training data, transform both train and test
        scaler = StandardScaler()
        z_i_scaled = scaler.fit_transform(z_i)
        z_i_test_scaled = scaler.transform(z_i_test)  # Fixed: use transform, not fit_transform
        
        for digit in range(n_classes):
            binary_labels = (labels == digit).astype(int)
            binary_labels_test = (labels_test == digit).astype(int)
            
            try:
                clf = LogisticRegression(solver='lbfgs', max_iter=1000, class_weight='balanced')
                clf.fit(z_i_scaled, binary_labels)
                
                # Evaluate on TRAINING data
                prob_train = clf.predict_proba(z_i_scaled)[:, 1]
                auc_train = roc_auc_score(binary_labels, prob_train)
                
                # Evaluate on TEST data
                prob_test = clf.predict_proba(z_i_test_scaled)[:, 1]
                auc_test = roc_auc_score(binary_labels_test, prob_test)
            except Exception:
                auc_train = 0.5
                auc_test = 0.5
            
            per_neuron_digit_auc.append({
                "neuron_idx": neuron_idx,
                "digit": digit,
                "auc_train": auc_train,
                "auc_test": auc_test
            })
    
    df = pd.DataFrame(per_neuron_digit_auc)
    return df


def evaluate_per_neuron(activations, labels, sae_name="SAE", test_size=0.2, random_state=42, top_k=10):
    
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )
    
    per_neuron_results = []
    
    for neuron_idx in range(activations.shape[1]):
        neuron_train = X_train[:, neuron_idx:neuron_idx+1]  
        neuron_test = X_test[:, neuron_idx:neuron_idx+1]
        
        scaler = StandardScaler()
        neuron_train_scaled = scaler.fit_transform(neuron_train)
        neuron_test_scaled = scaler.transform(neuron_test)
        
        classifier = LogisticRegression(
            random_state=random_state,
            max_iter=1000,
            solver='lbfgs'
        )
        
        try:
            classifier.fit(neuron_train_scaled, y_train)
            
            train_acc = classifier.score(neuron_train_scaled, y_train)
            test_acc = classifier.score(neuron_test_scaled, y_test)
            
            cv_scores = cross_val_score(
                classifier, neuron_train_scaled, y_train, 
                cv=3, scoring='accuracy'
            )
            
            neuron_stats = {
                'neuron_idx': neuron_idx,
                'test_accuracy': test_acc,
                'train_accuracy': train_acc,
                'cv_mean': cv_scores.mean(),
                'cv_std': cv_scores.std(),
                'activation_mean': neuron_train.mean(),
                'activation_std': neuron_train.std(),
                'sparsity': (neuron_train == 0).mean(),  
                'max_activation': neuron_train.max()
            }
            
        except Exception as e:
            neuron_stats = {
                'neuron_idx': neuron_idx,
                'test_accuracy': 0.0,
                'train_accuracy': 0.0,
                'cv_mean': 0.0,
                'cv_std': 0.0,
                'activation_mean': neuron_train.mean(),
                'activation_std': neuron_train.std(),
                'sparsity': (neuron_train == 0).mean(),
                'max_activation': neuron_train.max()
            }
        
        per_neuron_results.append(neuron_stats)
    
    neuron_df = pd.DataFrame(per_neuron_results)
    neuron_df = neuron_df.sort_values('test_accuracy', ascending=False).reset_index(drop=True)
    
    summary_stats = {
        'sae_name': sae_name,
        'n_neurons': len(per_neuron_results),
        'top_neuron_accuracy': neuron_df['test_accuracy'].iloc[0],
        'mean_neuron_accuracy': neuron_df['test_accuracy'].mean(),
        'median_neuron_accuracy': neuron_df['test_accuracy'].median(),
        'std_neuron_accuracy': neuron_df['test_accuracy'].std(),
        'neurons_above_random': (neuron_df['test_accuracy'] > 0.1).sum(),  
        'neurons_above_50pct': (neuron_df['test_accuracy'] > 0.5).sum(),
        'top_10_mean': neuron_df['test_accuracy'].head(10).mean(),
        'mean_sparsity': neuron_df['sparsity'].mean()
    }

    return {
        'summary': summary_stats,
        'per_neuron_df': neuron_df,
        'per_neuron_list': per_neuron_results
    }

def evaluate_sae_activations(activations, labels, sae_name="SAE", test_size=0.2, random_state=42):
    """
    Evaluate SAE concept encoding using pre-extracted activations.
    
    Args:
        activations: torch.Tensor or np.array of shape [N_samples, N_features]
        labels: torch.Tensor or np.array of shape [N_samples] with class labels
        sae_name: Name for this SAE (for display)
        test_size: Fraction for test split
        random_state: Random seed
    
    Returns:
        dict: Evaluation results including test accuracy
    """
    
    # Convert to numpy if needed
    if isinstance(activations, torch.Tensor):
        activations = activations.cpu().numpy()
    if isinstance(labels, torch.Tensor):
        labels = labels.cpu().numpy()
    
    print(f"Evaluating {sae_name}")
    print(f"Activations shape: {activations.shape}")
    print(f"Labels shape: {labels.shape}")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        activations, labels, 
        test_size=test_size, 
        random_state=random_state, 
        stratify=labels
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Train logistic regression
    classifier = LogisticRegression(
        random_state=random_state,
        max_iter=1000,
        solver='lbfgs'
    )
    classifier.fit(X_train_scaled, y_train)
    
    # Evaluate
    train_acc = classifier.score(X_train_scaled, y_train)
    test_acc = classifier.score(X_test_scaled, y_test)
    
    # Cross-validation for robustness
    cv_scores = cross_val_score(
        classifier, X_train_scaled, y_train, 
        cv=5, scoring='accuracy'
    )
    
    # Predictions for detailed analysis
    y_pred = classifier.predict(X_test_scaled)
    
    results = {
        'sae_name': sae_name,
        'train_accuracy': train_acc,
        'test_accuracy': test_acc,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'n_features': activations.shape[1],
        'n_samples': activations.shape[0]
    }
    
    print(f"  Train Accuracy: {train_acc:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f}")
    print(f"  CV Score: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
    print(f"  Features: {activations.shape[1]}")
    print()
    
    return results