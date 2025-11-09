import numpy as np
import torch
import pandas as pd
from sklearn.preprocessing import StandardScaler
from typing import Dict, List, Optional, Tuple
from scipy.linalg import svd


def compute_svd_metrics(activations: np.ndarray, 
                       normalize: bool = True,
                       center: bool = True) -> Dict:
    """
    Compute SVD-based metrics for SAE activations.
    
    Args:
        activations: Array of shape [N_samples, N_neurons]
        normalize: Whether to standardize features
        center: Whether to center the data
    
    Returns:
        Dictionary with SVD metrics
    """
    # Convert to numpy if needed
    if hasattr(activations, 'cpu'):
        activations = activations.cpu().numpy()
    
    # Preprocessing
    X = activations.copy()
    
    if center:
        X = X - X.mean(axis=0)
    
    if normalize:
        scaler = StandardScaler(with_mean=False)  # Already centered if center=True
        X = scaler.fit_transform(X)
    
    # Compute SVD
    U, s, Vt = svd(X, full_matrices=False)
    
    # Basic metrics
    rank = np.linalg.matrix_rank(X)
    stable_rank = (s**2).sum() / (s[0]**2)  # ||A||_F^2 / ||A||_2^2
    
    # Effective rank (90% of variance)
    cumsum_var = np.cumsum(s**2)
    total_var = cumsum_var[-1]
    effective_rank_90 = np.argmax(cumsum_var >= 0.9 * total_var) + 1
    effective_rank_95 = np.argmax(cumsum_var >= 0.95 * total_var) + 1
    effective_rank_99 = np.argmax(cumsum_var >= 0.99 * total_var) + 1
    
    # Condition number
    condition_number = s[0] / s[-1] if s[-1] > 1e-12 else np.inf
    
    # Participation ratio (inverse of sum of squared normalized singular values)
    normalized_s = s**2 / (s**2).sum()
    participation_ratio = 1.0 / (normalized_s**2).sum()
    
    # Entropy of singular values
    entropy_sv = -np.sum(normalized_s * np.log(normalized_s + 1e-12))
    
    return {
        'matrix_rank': rank,
        'stable_rank': stable_rank,
        'effective_rank_90': effective_rank_90,
        'effective_rank_95': effective_rank_95,
        'effective_rank_99': effective_rank_99,
        'condition_number': condition_number,
        'participation_ratio': participation_ratio,
        'entropy_singular_values': entropy_sv,
        'singular_values': s,
        'explained_variance_ratio': s**2 / (s**2).sum(),
        'n_samples': X.shape[0],
        'n_features': X.shape[1],
        'frobenius_norm': np.linalg.norm(X, 'fro'),
        'spectral_norm': s[0]
    }


def analyze_rank_vs_sparsity(activations_dict: Dict[str, np.ndarray],
                           sparsity_levels: Optional[List[float]] = None,
                           normalize: bool = True) -> pd.DataFrame:
    """
    Analyze how rank metrics change with sparsity levels.
    
    Args:
        activations_dict: Dictionary mapping sparsity level names to activation arrays
        sparsity_levels: List of sparsity values (if None, will compute from activations)
        normalize: Whether to normalize activations
    
    Returns:
        DataFrame with rank metrics for each sparsity level
    """
    results = []
    
    for name, activations in activations_dict.items():
        # Compute sparsity if not provided
        if sparsity_levels is None:
            sparsity = (activations == 0).mean()
        else:
            # Assume names correspond to sparsity levels
            try:
                sparsity = float(name.split('_')[-1])  # Assumes format like "sae_0.1"
            except:
                sparsity = (activations == 0).mean()
        
        # Compute SVD metrics
        metrics = compute_svd_metrics(activations, normalize=normalize)
        
        # Add identifiers
        metrics['sae_name'] = name
        metrics['sparsity'] = sparsity
        
        results.append(metrics)
    
    return pd.DataFrame(results)


def compute_rank_by_class(activations: np.ndarray, 
                         labels: np.ndarray,
                         normalize: bool = True) -> Dict:
    """
    Compute rank metrics for each class separately.
    
    Args:
        activations: SAE activations [N_samples, N_neurons]
        labels: Class labels [N_samples]
        normalize: Whether to normalize activations
    
    Returns:
        Dictionary with per-class rank metrics
    """
    if hasattr(activations, 'cpu'):
        activations = activations.cpu().numpy()
    if hasattr(labels, 'cpu'):
        labels = labels.cpu().numpy()
    
    unique_labels = np.unique(labels)
    class_metrics = {}
    
    for label in unique_labels:
        mask = labels == label
        class_activations = activations[mask]
        
        metrics = compute_svd_metrics(class_activations, normalize=normalize)
        metrics['n_samples_class'] = class_activations.shape[0]
        
        class_metrics[f'class_{label}'] = metrics
    
    return class_metrics


def analyze_feature_correlations(activations: np.ndarray,
                                threshold: float = 0.8) -> Dict:
    """
    Analyze correlations between SAE features.
    
    Args:
        activations: SAE activations [N_samples, N_neurons]
        threshold: Correlation threshold for identifying redundant features
    
    Returns:
        Dictionary with correlation analysis results
    """
    if hasattr(activations, 'cpu'):
        activations = activations.cpu().numpy()
    
    # Remove zero-variance features
    non_zero_var = activations.var(axis=0) > 1e-12
    active_features = activations[:, non_zero_var]
    
    # Compute correlation matrix
    corr_matrix = np.corrcoef(active_features.T)
    
    # Find highly correlated pairs
    high_corr_pairs = []
    n_features = corr_matrix.shape[0]
    
    for i in range(n_features):
        for j in range(i+1, n_features):
            if abs(corr_matrix[i, j]) > threshold:
                high_corr_pairs.append({
                    'feature_i': i,
                    'feature_j': j,
                    'correlation': corr_matrix[i, j]
                })
    
    # Compute effective number of independent features
    eigenvals = np.linalg.eigvals(corr_matrix)
    eigenvals = eigenvals[eigenvals > 1e-12]  # Remove near-zero eigenvalues
    effective_features = len(eigenvals)
    
    return {
        'correlation_matrix': corr_matrix,
        'high_correlation_pairs': high_corr_pairs,
        'n_high_corr_pairs': len(high_corr_pairs),
        'max_correlation': np.abs(corr_matrix - np.eye(n_features)).max(),
        'mean_abs_correlation': np.abs(corr_matrix - np.eye(n_features)).mean(),
        'effective_n_features': effective_features,
        'redundancy_ratio': 1 - (effective_features / n_features),
        'n_zero_variance': (~non_zero_var).sum(),
        'correlation_threshold': threshold
    }


def plot_rank_analysis(rank_df: pd.DataFrame, 
                      save_path: Optional[str] = None):
    """
    Create visualization plots for rank analysis.
    
    Args:
        rank_df: DataFrame from analyze_rank_vs_sparsity
        save_path: Path to save plot (optional)
    """
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    
    # Plot 1: Stable rank vs sparsity
    axes[0, 0].scatter(rank_df['sparsity'], rank_df['stable_rank'])
    axes[0, 0].set_xlabel('Sparsity')
    axes[0, 0].set_ylabel('Stable Rank')
    axes[0, 0].set_title('Stable Rank vs Sparsity')
    
    # Plot 2: Matrix rank vs sparsity
    axes[0, 1].scatter(rank_df['sparsity'], rank_df['matrix_rank'])
    axes[0, 1].set_xlabel('Sparsity')
    axes[0, 1].set_ylabel('Matrix Rank')
    axes[0, 1].set_title('Matrix Rank vs Sparsity')
    
    # Plot 3: Effective rank (90%) vs sparsity
    axes[0, 2].scatter(rank_df['sparsity'], rank_df['effective_rank_90'])
    axes[0, 2].set_xlabel('Sparsity')
    axes[0, 2].set_ylabel('Effective Rank (90%)')
    axes[0, 2].set_title('Effective Rank vs Sparsity')
    
    # Plot 4: Condition number vs sparsity
    axes[1, 0].scatter(rank_df['sparsity'], np.log10(rank_df['condition_number']))
    axes[1, 0].set_xlabel('Sparsity')
    axes[1, 0].set_ylabel('Log10(Condition Number)')
    axes[1, 0].set_title('Condition Number vs Sparsity')
    
    # Plot 5: Participation ratio vs sparsity
    axes[1, 1].scatter(rank_df['sparsity'], rank_df['participation_ratio'])
    axes[1, 1].set_xlabel('Sparsity')
    axes[1, 1].set_ylabel('Participation Ratio')
    axes[1, 1].set_title('Participation Ratio vs Sparsity')
    
    # Plot 6: Entropy vs sparsity
    axes[1, 2].scatter(rank_df['sparsity'], rank_df['entropy_singular_values'])
    axes[1, 2].set_xlabel('Sparsity')
    axes[1, 2].set_ylabel('Entropy of Singular Values')
    axes[1, 2].set_title('SV Entropy vs Sparsity')
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()


def plot_singular_value_spectrum(activations: np.ndarray,
                                name: str = "SAE",
                                top_k: int = 50,
                                save_path: Optional[str] = None):
    """
    Plot the singular value spectrum.
    
    Args:
        activations: SAE activations
        name: Name for the plot
        top_k: Number of top singular values to plot
        save_path: Path to save plot (optional)
    """
    metrics = compute_svd_metrics(activations)
    s = metrics['singular_values']
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Plot 1: Singular values
    ax1.plot(range(1, min(len(s), top_k) + 1), s[:top_k], 'bo-')
    ax1.set_xlabel('Singular Value Index')
    ax1.set_ylabel('Singular Value')
    ax1.set_title(f'{name}: Top {top_k} Singular Values')
    ax1.set_yscale('log')
    
    # Plot 2: Cumulative explained variance
    cumvar = np.cumsum(s**2) / np.sum(s**2)
    ax2.plot(range(1, len(cumvar) + 1), cumvar, 'r-')
    ax2.axhline(y=0.9, color='gray', linestyle='--', alpha=0.7, label='90%')
    ax2.axhline(y=0.95, color='gray', linestyle='--', alpha=0.7, label='95%')
    ax2.axhline(y=0.99, color='gray', linestyle='--', alpha=0.7, label='99%')
    ax2.set_xlabel('Number of Components')
    ax2.set_ylabel('Cumulative Explained Variance')
    ax2.set_title(f'{name}: Cumulative Explained Variance')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
    
    plt.show()
    
    # Print key metrics
    print(f"\n{name} Rank Metrics:")
    print(f"  Matrix Rank: {metrics['matrix_rank']}")
    print(f"  Stable Rank: {metrics['stable_rank']:.2f}")
    print(f"  Effective Rank (90%): {metrics['effective_rank_90']}")
    print(f"  Effective Rank (95%): {metrics['effective_rank_95']}")
    print(f"  Condition Number: {metrics['condition_number']:.2e}")


def compare_saes_rank_analysis(sae_activations_dict: Dict[str, np.ndarray],
                              normalize: bool = True) -> pd.DataFrame:
    """
    Compare rank metrics across multiple SAEs.
    
    Args:
        sae_activations_dict: Dictionary mapping SAE names to activation arrays
        normalize: Whether to normalize activations
    
    Returns:
        DataFrame comparing rank metrics across SAEs
    """
    comparison_results = []
    
    for sae_name, activations in sae_activations_dict.items():
        metrics = compute_svd_metrics(activations, normalize=normalize)
        metrics['sae_name'] = sae_name
        comparison_results.append(metrics)
    
    comparison_df = pd.DataFrame(comparison_results)
    
    # Sort by stable rank for easier comparison
    comparison_df = comparison_df.sort_values('stable_rank', ascending=False)
    
    return comparison_df


# Example usage and testing function
def compute_and_save_rank_metrics(activations_dict: Dict[str, np.ndarray],
                                 labels: np.ndarray,
                                 save_path: str,
                                 normalize: bool = True) -> Dict:
    """
    Compute rank metrics for multiple SAEs and save results to pickle file.
    
    Args:
        activations_dict: Dictionary mapping SAE names to activation arrays
        labels: Class labels for per-class analysis
        save_path: Path to save the results pickle file
        normalize: Whether to normalize activations
    
    Returns:
        Dictionary containing all computed metrics
    """
    all_metrics = {}
    
    for sae_name, activations in activations_dict.items():
        print(f"Computing rank metrics for {sae_name}...")
        
        # Overall metrics
        overall_metrics = compute_svd_metrics(activations, normalize=normalize)
        
        # Per-class metrics
        class_metrics = compute_rank_by_class(activations, labels, normalize=normalize)
        
        # Correlation analysis
        correlation_metrics = analyze_feature_correlations(activations)
        
        # Additional statistics
        sparsity = (activations == 0).mean() if hasattr(activations, 'mean') else (activations == 0).mean()
        activation_stats = {
            'mean_activation': float(activations.mean()),
            'std_activation': float(activations.std()),
            'max_activation': float(activations.max()),
            'min_activation': float(activations.min()),
            'sparsity': float(sparsity),
            'active_neurons': int((activations.sum(axis=0) != 0).sum())
        }
        
        # Combine all metrics for this SAE
        all_metrics[sae_name] = {
            'overall_metrics': overall_metrics,
            'class_metrics': class_metrics,
            'correlation_metrics': correlation_metrics,
            'activation_stats': activation_stats,
            'sae_name': sae_name
        }
    
    # Save to pickle file
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(all_metrics, f)
    
    print(f"Rank metrics saved to {save_path}")
    return all_metrics


def integrate_with_classification_results(classification_results: Dict,
                                        activations_dict: Dict[str, np.ndarray],
                                        labels: np.ndarray,
                                        save_path: str,
                                        normalize: bool = True) -> Dict:
    """
    Integrate rank analysis with existing classification results and save combined results.
    
    Args:
        classification_results: Results from evaluate_per_neuron_1vsall_new
        activations_dict: Dictionary mapping SAE names to activation arrays
        labels: Class labels
        save_path: Path to save combined results
        normalize: Whether to normalize activations
    
    Returns:
        Combined results dictionary
    """
    combined_results = {}
    
    # Copy classification results
    for sae_name, class_results in classification_results.items():
        combined_results[sae_name] = class_results.copy()
    
    # Add rank metrics
    for sae_name, activations in activations_dict.items():
        if sae_name in combined_results:
            print(f"Adding rank metrics to {sae_name}...")
            
            # Overall metrics
            overall_metrics = compute_svd_metrics(activations, normalize=normalize)
            
            # Per-class metrics
            class_metrics = compute_rank_by_class(activations, labels, normalize=normalize)
            
            # Correlation analysis
            correlation_metrics = analyze_feature_correlations(activations)
            
            # Add to existing results
            combined_results[sae_name]['rank_metrics'] = {
                'overall': overall_metrics,
                'per_class': class_metrics,
                'correlations': correlation_metrics
            }
        else:
            print(f"Warning: {sae_name} not found in classification results")
    
    # Save combined results
    import pickle
    with open(save_path, 'wb') as f:
        pickle.dump(combined_results, f)
    
    print(f"Combined results saved to {save_path}")
    return combined_results


def create_rank_summary_table(results_dict: Dict) -> pd.DataFrame:
    """
    Create a summary table of rank metrics across all SAEs.
    
    Args:
        results_dict: Results from compute_and_save_rank_metrics or integrate_with_classification_results
    
    Returns:
        DataFrame with summary statistics
    """
    summary_data = []
    
    for sae_name, results in results_dict.items():
        # Handle different result structures
        if 'rank_metrics' in results:
            # From integrate_with_classification_results
            rank_data = results['rank_metrics']['overall']
            class_data = results['sae_summary'] if 'sae_summary' in results else {}
        elif 'overall_metrics' in results:
            # From compute_and_save_rank_metrics
            rank_data = results['overall_metrics']
            class_data = {}
        else:
            continue
        
        summary_row = {
            'sae_name': sae_name,
            'stable_rank': rank_data.get('stable_rank', np.nan),
            'matrix_rank': rank_data.get('matrix_rank', np.nan),
            'effective_rank_90': rank_data.get('effective_rank_90', np.nan),
            'effective_rank_95': rank_data.get('effective_rank_95', np.nan),
            'condition_number': rank_data.get('condition_number', np.nan),
            'participation_ratio': rank_data.get('participation_ratio', np.nan),
            'entropy_sv': rank_data.get('entropy_singular_values', np.nan),
            'n_features': rank_data.get('n_features', np.nan),
            'sparsity': results.get('activation_stats', {}).get('sparsity', np.nan),
            'mean_best_auc': class_data.get('mean_best_auc_test', np.nan),
            'fraction_selective': class_data.get('fraction_selective', np.nan)
        }
        
        summary_data.append(summary_row)
    
    return pd.DataFrame(summary_data)


def example_usage():
    """
    Example of how to use the rank analysis functions.
    """
    # Generate synthetic SAE activations for demonstration
    np.random.seed(42)
    
    # Simulate different sparsity levels
    n_samples, n_neurons = 1000, 500
    
    # Dense activations
    dense_activations = np.random.randn(n_samples, n_neurons)
    
    # Sparse activations (90% sparsity)
    sparse_activations = np.random.randn(n_samples, n_neurons)
    sparse_mask = np.random.rand(n_samples, n_neurons) < 0.9
    sparse_activations[sparse_mask] = 0
    
    # Very sparse activations (95% sparsity)
    very_sparse_activations = np.random.randn(n_samples, n_neurons)
    very_sparse_mask = np.random.rand(n_samples, n_neurons) < 0.95
    very_sparse_activations[very_sparse_mask] = 0
    
    # Create synthetic labels
    labels = np.random.randint(0, 10, n_samples)
    
    # Example activations dict
    sae_dict = {
        'dense': dense_activations,
        'sparse_90': sparse_activations,
        'sparse_95': very_sparse_activations
    }
    
    # Compute and save rank metrics
    results = compute_and_save_rank_metrics(
        sae_dict, 
        labels, 
        'example_rank_metrics.pkl'
    )
    
    # Create summary table
    summary_df = create_rank_summary_table(results)
    print("Summary Table:")
    print(summary_df[['sae_name', 'stable_rank', 'effective_rank_90', 'sparsity']])


if __name__ == "__main__":
    example_usage()