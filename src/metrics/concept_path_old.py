import torch
import numpy as np
import pickle
import os
from typing import Dict, List, Tuple, Any, Optional
from collections import defaultdict
import pandas as pd
from .logistic_regression_classifier import evaluate_sae_activations

class ConceptPathExtractor:
    """
    Extract concept paths from SAE activations.
    A concept path is the set of neurons that strongly activate for a given input.
    """
    
    def __init__(self, threshold_method='top_k', threshold_value=10, min_activation=0.1):
        """
        Initialize the concept path extractor.
        
        Args:
            threshold_method: Method to determine active neurons
                - 'top_k': Select top k most active neurons
                - 'percentile': Select neurons above certain percentile (0-100)
                - 'absolute': Select neurons above absolute threshold
                - 'std_threshold': Select neurons above mean + threshold*std
            threshold_value: Value for the threshold method
            min_activation: Minimum activation value to consider (filters out very small activations)
        """
        self.threshold_method = threshold_method
        self.threshold_value = threshold_value
        self.min_activation = min_activation
        
    def extract_active_neurons(self, activations: torch.Tensor, sample_idx: int = None) -> Dict[str, Any]:
        """
        Extract active neurons for a single sample.
        
        Args:
            activations: Tensor of shape [n_neurons] for single sample
            sample_idx: Index of the sample (for tracking)
            
        Returns:
            Dict containing active neuron indices, values, and metadata
        """
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()
        
        # Filter out very small activations
        valid_mask = activations >= self.min_activation
        valid_activations = activations[valid_mask]
        valid_indices = np.where(valid_mask)[0]
        
        if len(valid_activations) == 0:
            return {
                'sample_idx': sample_idx,
                'active_neurons': [],
                'activation_values': [],
                'num_active': 0,
                'total_neurons': len(activations),
                'activation_sum': 0.0,
                'activation_mean': 0.0
            }
        
        # Apply threshold method
        if self.threshold_method == 'top_k':
            k = min(self.threshold_value, len(valid_activations))
            top_k_indices = np.argpartition(valid_activations, -k)[-k:]
            selected_indices = valid_indices[top_k_indices]
            selected_values = valid_activations[top_k_indices]
            
        elif self.threshold_method == 'percentile':
            threshold = np.percentile(valid_activations, self.threshold_value)
            mask = valid_activations >= threshold
            selected_indices = valid_indices[mask]
            selected_values = valid_activations[mask]
            
        elif self.threshold_method == 'absolute':
            mask = valid_activations >= self.threshold_value
            selected_indices = valid_indices[mask]
            selected_values = valid_activations[mask]
            
        elif self.threshold_method == 'std_threshold':
            mean_act = np.mean(valid_activations)
            std_act = np.std(valid_activations)
            threshold = mean_act + self.threshold_value * std_act
            mask = valid_activations >= threshold
            selected_indices = valid_indices[mask]
            selected_values = valid_activations[mask]
            
        else:
            raise ValueError(f"Unknown threshold method: {self.threshold_method}")
        
        # Sort by activation strength (descending)
        sort_order = np.argsort(selected_values)[::-1]
        selected_indices = selected_indices[sort_order]
        selected_values = selected_values[sort_order]
        
        return {
            'sample_idx': sample_idx,
            'active_neurons': selected_indices.tolist(),
            'activation_values': selected_values.tolist(),
            'num_active': len(selected_indices),
            'total_neurons': len(activations),
            'activation_sum': float(np.sum(selected_values)),
            'activation_mean': float(np.mean(selected_values)) if len(selected_values) > 0 else 0.0,
            'max_activation': float(np.max(selected_values)) if len(selected_values) > 0 else 0.0
        }
    
    def extract_concept_paths(self, activations: torch.Tensor, labels: torch.Tensor = None, 
                            sae_name: str = "SAE") -> Dict[str, Any]:
        """
        Extract concept paths for all samples in the dataset.
        
        Args:
            activations: Tensor of shape [n_samples, n_neurons]
            labels: Optional tensor of shape [n_samples] with class labels
            sae_name: Name of the SAE for tracking
            
        Returns:
            Dict containing all concept paths and summary statistics
        """
        if isinstance(activations, torch.Tensor):
            activations = activations.cpu().numpy()
        if labels is not None and isinstance(labels, torch.Tensor):
            labels = labels.cpu().numpy()
        
        n_samples, n_neurons = activations.shape
        concept_paths = []
        
        print(f"Extracting concept paths for {n_samples} samples using {self.threshold_method} method...")
        
        for i in range(n_samples):
            path = self.extract_active_neurons(activations[i], sample_idx=i)
            if labels is not None:
                path['label'] = int(labels[i])
            concept_paths.append(path)
            
            # if (i + 1) % 1000 == 0:
                # print(f"Processed {i + 1}/{n_samples} samples")
        
        # Compute summary statistics
        num_active_neurons = [path['num_active'] for path in concept_paths]
        activation_sums = [path['activation_sum'] for path in concept_paths]
        neuron_set = self._get_all_active_neurons(concept_paths)

        summary = {
            'sae_name': sae_name,
            'n_samples': n_samples,
            'n_neurons': n_neurons,
            'threshold_method': self.threshold_method,
            'threshold_value': self.threshold_value,
            'min_activation': self.min_activation,
            'mean_active_neurons': np.mean(num_active_neurons),
            'std_active_neurons': np.std(num_active_neurons),
            'median_active_neurons': np.median(num_active_neurons),
            'min_active_neurons': np.min(num_active_neurons),
            'max_active_neurons': np.max(num_active_neurons),
            'mean_activation_sum': np.mean(activation_sums),
            'total_unique_neurons_used': len(neuron_set)
        }

        evaluate_sae_activations(
            activations, 
            labels, 
            sae_name, 
            sorted(list(neuron_set))
        )
        
        # Per-class statistics if labels provided
        if labels is not None:
            class_stats = self._compute_class_statistics(concept_paths, labels)
            summary['class_statistics'] = class_stats
        
        return {
            'concept_paths': concept_paths,
            'summary': summary,
            'extraction_config': {
                'threshold_method': self.threshold_method,
                'threshold_value': self.threshold_value,
                'min_activation': self.min_activation
            }
        }
    
    def _get_all_active_neurons(self, concept_paths: List[Dict]) -> set:
        """Get set of all neurons that were active across all samples."""
        all_neurons = set()
        for path in concept_paths:
            all_neurons.update(path['active_neurons'])
        return all_neurons
    
    def _compute_class_statistics(self, concept_paths: List[Dict], labels: np.ndarray) -> Dict:
        """Compute per-class statistics for concept paths."""
        class_stats = {}
        unique_labels = np.unique(labels)
        
        for label in unique_labels:
            class_paths = [path for path in concept_paths if path.get('label') == label]
            if not class_paths:
                continue
                
            num_active = [path['num_active'] for path in class_paths]
            activation_sums = [path['activation_sum'] for path in class_paths]
            
            # Get most frequently active neurons for this class
            neuron_counts = defaultdict(int)
            for path in class_paths:
                for neuron_idx in path['active_neurons']:
                    neuron_counts[neuron_idx] += 1
            
            # Top 10 most frequent neurons for this class
            top_neurons = sorted(neuron_counts.items(), key=lambda x: x[1], reverse=True)[:10]
            
            class_stats[int(label)] = {
                'n_samples': len(class_paths),
                'mean_active_neurons': np.mean(num_active),
                'std_active_neurons': np.std(num_active),
                'mean_activation_sum': np.mean(activation_sums),
                'unique_neurons_used': len(set().union(*[path['active_neurons'] for path in class_paths])),
                'top_frequent_neurons': top_neurons
            }
        
        return class_stats
    
    def save_concept_paths(self, concept_paths_data: Dict, save_path: str):
        """Save concept paths to disk."""
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        with open(save_path, 'wb') as f:
            pickle.dump(concept_paths_data, f)
        print(f"Concept paths saved to: {save_path}")
    
    def load_concept_paths(self, load_path: str) -> Dict:
        """Load concept paths from disk."""
        with open(load_path, 'rb') as f:
            return pickle.load(f)
    
    def analyze_concept_overlap(self, concept_paths_data: Dict, class_pairs: List[Tuple] = None) -> Dict:
        """
        Analyze overlap between concept paths of different classes.
        
        Args:
            concept_paths_data: Output from extract_concept_paths
            class_pairs: List of class pairs to analyze, if None analyzes all pairs
            
        Returns:
            Dict with overlap analysis results
        """
        concept_paths = concept_paths_data['concept_paths']
        
        # Group paths by class
        class_paths = defaultdict(list)
        for path in concept_paths:
            if 'label' in path:
                class_paths[path['label']].append(path)
        
        if not class_paths:
            return {'error': 'No class labels found in concept paths'}
        
        classes = sorted(class_paths.keys())
        
        if class_pairs is None:
            class_pairs = [(i, j) for i in classes for j in classes if i < j]
        
        overlap_results = {}
        
        for class_a, class_b in class_pairs:
            if class_a not in class_paths or class_b not in class_paths:
                continue
                
            # Get all neurons used by each class
            neurons_a = set()
            neurons_b = set()
            
            for path in class_paths[class_a]:
                neurons_a.update(path['active_neurons'])
            for path in class_paths[class_b]:
                neurons_b.update(path['active_neurons'])
            
            # Compute overlap metrics
            intersection = neurons_a & neurons_b
            union = neurons_a | neurons_b
            
            jaccard = len(intersection) / len(union) if union else 0
            overlap_a = len(intersection) / len(neurons_a) if neurons_a else 0
            overlap_b = len(intersection) / len(neurons_b) if neurons_b else 0
            
            overlap_results[f'class_{class_a}_vs_{class_b}'] = {
                'jaccard_similarity': jaccard,
                'overlap_fraction_a': overlap_a,
                'overlap_fraction_b': overlap_b,
                'shared_neurons': len(intersection),
                'unique_to_a': len(neurons_a - neurons_b),
                'unique_to_b': len(neurons_b - neurons_a),
                'total_neurons_a': len(neurons_a),
                'total_neurons_b': len(neurons_b)
            }
        
        return {
            'overlap_analysis': overlap_results,
            'summary': {
                'mean_jaccard': np.mean([r['jaccard_similarity'] for r in overlap_results.values()]),
                'mean_shared_neurons': np.mean([r['shared_neurons'] for r in overlap_results.values()])
            }
        }

def extract_and_save_concept_paths(sae, activations, labels, sae_name, save_dir, 
                                 threshold_method='top_k', threshold_value=10, min_activation=0.1):
    """
    Convenience function to extract and save concept paths for a single SAE.
    
    Args:
        sae: The SAE model
        activations: Input activations [n_samples, n_neurons] 
        labels: Class labels [n_samples]
        sae_name: Name for the SAE
        save_dir: Directory to save results
        threshold_method: Method for selecting active neurons
        threshold_value: Threshold value
        min_activation: Minimum activation threshold
        
    Returns:
        Dict with concept paths data
    """
    extractor = ConceptPathExtractor(
        threshold_method=threshold_method,
        threshold_value=threshold_value, 
        min_activation=min_activation
    )
    
    concept_paths_data = extractor.extract_concept_paths(activations, labels, sae_name)
    
    # Save the results
    save_path = os.path.join(save_dir, f"{sae_name}_concept_paths.pkl")
    extractor.save_concept_paths(concept_paths_data, save_path)
    
    # Also save summary as CSV for easy viewing
    summary_df = pd.DataFrame([concept_paths_data['summary']])
    summary_csv_path = os.path.join(save_dir, f"{sae_name}_concept_paths_summary.csv")
    summary_df.to_csv(summary_csv_path, index=False)
    
    return concept_paths_data