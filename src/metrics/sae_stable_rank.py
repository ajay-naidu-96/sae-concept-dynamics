import torch
import numpy as np
from typing import Dict, Any, Optional, Union

def compute_stable_rank(matrix: torch.Tensor, 
                       threshold: float = 0.01, 
                       method: str = 'frobenius') -> Dict[str, Any]:

    matrix_cpu = matrix.cpu().float().detach()  
    
    U, S, V = torch.svd(matrix_cpu)
    
    frobenius_norm_squared = torch.sum(matrix_cpu * matrix_cpu).item()
    max_singular_value_squared = (S.max() ** 2).item() if len(S) > 0 else 0
    
    if max_singular_value_squared > 1e-12:
        stable_rank_frobenius = frobenius_norm_squared / max_singular_value_squared
    else:
        stable_rank_frobenius = 0
    
    stable_rank_threshold = 0
    if method in ['relative', 'absolute']:
        if method == 'relative':
            if S.max() > 0:
                significant_mask = S >= threshold * S.max()
                stable_rank_threshold = significant_mask.sum().item()
            else:
                stable_rank_threshold = 0
        elif method == 'absolute':
            significant_mask = S >= threshold
            stable_rank_threshold = significant_mask.sum().item()
    
    total_rank = torch.linalg.matrix_rank(matrix_cpu).item()
    condition_number = (S.max() / S.min()).item() if S.min() > 1e-10 else float('inf')
    
    effective_dim_ratio = stable_rank_frobenius / matrix.shape[1] if matrix.shape[1] > 0 else 0
    
    spectral_decay = None
    if len(S) > 1:
        log_s = torch.log(S[S > 1e-10])
        if len(log_s) > 1:
            spectral_decay = -(log_s[-1] - log_s[0]) / (len(log_s) - 1)
            spectral_decay = spectral_decay.item()
    
    participation_ratio = (torch.sum(S) ** 2 / torch.sum(S ** 2)).item() if len(S) > 0 else 0
    
    effective_rank = 0
    spectral_entropy = 0
    singular_value_distribution = None
    
    if len(S) > 0:
        nonzero_S = S[S > 1e-12]
        
        if len(nonzero_S) > 0:
            l1_norm = torch.sum(nonzero_S).item()
            if l1_norm > 0:
                # Fix: Add detach() before converting to numpy
                singular_value_distribution = (nonzero_S / l1_norm).detach().numpy()

                log_probs = torch.log(nonzero_S / l1_norm)
                spectral_entropy = -torch.sum((nonzero_S / l1_norm) * log_probs).item()
                
                effective_rank = np.exp(spectral_entropy)
    
    return {
        'stable_rank': stable_rank_frobenius, 
        'effective_rank': effective_rank, 
        'stable_rank_threshold': stable_rank_threshold,  
        'total_rank': total_rank,
        'matrix_shape': tuple(matrix.shape),
        'condition_number': condition_number,
        'effective_dim_ratio': effective_dim_ratio,
        'participation_ratio': participation_ratio,
        'spectral_entropy': spectral_entropy,
        'singular_value_distribution': singular_value_distribution,
        'spectral_decay': spectral_decay,
        'frobenius_norm_squared': frobenius_norm_squared,
        'max_singular_value_squared': max_singular_value_squared,
        'singular_values': S.detach().numpy(),  # Also add detach() here
        'threshold_used': threshold,
        'method_used': method
    }

def get_weight_tensor(obj):
    """Extract weight tensor from various possible object types."""
    if isinstance(obj, torch.Tensor):
        return obj
    elif hasattr(obj, 'weight') and isinstance(obj.weight, torch.Tensor):
        return obj.weight
    elif isinstance(obj, torch.nn.Linear):
        return obj.weight
    elif isinstance(obj, torch.nn.Module):
        for param in obj.parameters():
            if isinstance(param, torch.Tensor):
                return param
    return None

def evaluate_sae_stable_rank(sae: torch.nn.Module, 
                           sae_name: str,
                           thresholds: list = [0.001, 0.01, 0.05, 0.1],
                           include_both_matrices: bool = True,
                           use_frobenius_definition: bool = True) -> Dict[str, Any]:

    results = {
        'sae_name': sae_name,
        'encoder_analysis': {},
        'decoder_analysis': {},
        'summary': {}
    }
    
    encoder_weight = None
    decoder_weight = None
    
    # Try common attribute names
    encoder_attrs = ['encoder', 'W_enc', 'encode_weight', 'encoder_weight']
    decoder_attrs = ['dictionary', 'decoder', 'W_dec', 'decode_weight', 'decoder_weight']
    
    for attr_name in encoder_attrs:
        if hasattr(sae, attr_name):
            encoder_weight = get_weight_tensor(getattr(sae, attr_name))
            if encoder_weight is not None:
                break
    
    for attr_name in decoder_attrs:
        if hasattr(sae, attr_name):
            attr_obj = getattr(sae, attr_name)
            if attr_name == 'dictionary' and hasattr(attr_obj, '_weights'):
                decoder_weight = attr_obj._weights
            else:
                decoder_weight = get_weight_tensor(attr_obj)
            if decoder_weight is not None:
                break
    
    # Fallback: use Linear layers or parameters
    if encoder_weight is None or decoder_weight is None:
        # Try Linear layers first
        linear_layers = [(name, module) for name, module in sae.named_modules() 
                        if isinstance(module, torch.nn.Linear)]
        
        if encoder_weight is None and len(linear_layers) >= 1:
            encoder_weight = linear_layers[0][1].weight
        if decoder_weight is None and len(linear_layers) >= 2:
            decoder_weight = linear_layers[1][1].weight
        
        # Final fallback: use 2D parameters (skip bias)
        if encoder_weight is None or decoder_weight is None:
            weight_params = [(name, param) for name, param in sae.named_parameters() 
                           if param.dim() >= 2]
            
            if encoder_weight is None and len(weight_params) >= 1:
                encoder_weight = weight_params[0][1]
            if decoder_weight is None and len(weight_params) >= 2:
                decoder_weight = weight_params[1][1]
    
    # Analyze encoder weights
    if encoder_weight is not None and isinstance(encoder_weight, torch.Tensor):
        if use_frobenius_definition:
            # Use true stable rank definition
            results['encoder_analysis']['frobenius'] = compute_stable_rank(
                encoder_weight, method='frobenius'
            )
        
        # Also compute threshold-based versions for comparison
        for threshold in thresholds:
            results['encoder_analysis'][f'threshold_{threshold}'] = compute_stable_rank(
                encoder_weight, threshold=threshold, method='relative'
            )
    
    # Analyze decoder weights
    if include_both_matrices and decoder_weight is not None and isinstance(decoder_weight, torch.Tensor):
        if use_frobenius_definition:
            # Use true stable rank definition
            results['decoder_analysis']['frobenius'] = compute_stable_rank(
                decoder_weight, method='frobenius'
            )
        
        # Also compute threshold-based versions for comparison
        for threshold in thresholds:
            results['decoder_analysis'][f'threshold_{threshold}'] = compute_stable_rank(
                decoder_weight, threshold=threshold, method='relative'
            )
    
    # Summary statistics
    if encoder_weight is not None and isinstance(encoder_weight, torch.Tensor):
        if use_frobenius_definition and 'frobenius' in results['encoder_analysis']:
            enc_result = results['encoder_analysis']['frobenius']
        else:
            mid_threshold = thresholds[len(thresholds)//2]
            key = f'threshold_{mid_threshold}'
            enc_result = results['encoder_analysis'].get(key, {})
        
        if enc_result:
            results['summary'] = {
                'encoder_stable_rank': enc_result['stable_rank'],
                'encoder_effective_rank': enc_result.get('effective_rank', 'N/A'),
                'encoder_stable_rank_threshold': enc_result.get('stable_rank_threshold', 'N/A'),
                'encoder_total_rank': enc_result['total_rank'],
                'encoder_participation_ratio': enc_result.get('participation_ratio', 'N/A'),
                'encoder_spectral_entropy': enc_result.get('spectral_entropy', 'N/A'),
                'encoder_utilization_ratio': enc_result['stable_rank'] / enc_result['total_rank'] if enc_result['total_rank'] > 0 else 0,
                'encoder_shape': enc_result['matrix_shape'],
                'encoder_condition_number': enc_result['condition_number']
            }
        
        if (include_both_matrices and decoder_weight is not None and 
            isinstance(decoder_weight, torch.Tensor)):
            
            if use_frobenius_definition and 'frobenius' in results['decoder_analysis']:
                dec_result = results['decoder_analysis']['frobenius']
            else:
                mid_threshold = thresholds[len(thresholds)//2]
                key = f'threshold_{mid_threshold}'
                dec_result = results['decoder_analysis'].get(key, {})
            
            if dec_result:
                results['summary'].update({
                    'decoder_stable_rank': dec_result['stable_rank'],
                    'decoder_effective_rank': dec_result.get('effective_rank', 'N/A'),
                    'decoder_stable_rank_threshold': dec_result.get('stable_rank_threshold', 'N/A'),
                    'decoder_total_rank': dec_result['total_rank'],
                    'decoder_participation_ratio': dec_result.get('participation_ratio', 'N/A'),
                    'decoder_spectral_entropy': dec_result.get('spectral_entropy', 'N/A'),
                    'decoder_utilization_ratio': dec_result['stable_rank'] / dec_result['total_rank'] if dec_result['total_rank'] > 0 else 0,
                    'decoder_shape': dec_result['matrix_shape'],
                    'decoder_condition_number': dec_result['condition_number']
                })
    
    return results

# def compare_stable_rank_definitions(matrix: torch.Tensor) -> Dict[str, Any]:
#     """
#     Compare the mathematical stable rank definition with threshold-based approximations.
#     """
#     # True stable rank
#     frobenius_result = compute_stable_rank(matrix, method='frobenius')
    
#     threshold_results = {}
#     for threshold in [0.001, 0.01, 0.05, 0.1, 0.2]:
#         threshold_results[f'threshold_{threshold}'] = compute_stable_rank(
#             matrix, threshold=threshold, method='relative'
#         )
    
#     return {
#         'true_stable_rank': frobenius_result,
#         'threshold_approximations': threshold_results,
#         'comparison': {
#             'frobenius_value': frobenius_result['stable_rank'],
#             'effective_rank': frobenius_result['effective_rank'],
#             'threshold_values': {k: v['stable_rank_threshold'] for k, v in threshold_results.items()},
#             'total_rank': frobenius_result['total_rank'],
#             'participation_ratio': frobenius_result['participation_ratio'],
#             'spectral_entropy': frobenius_result['spectral_entropy']
#         }
#     }