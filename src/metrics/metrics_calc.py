import pandas as pd
import numpy as np


def analyze_sae(df):
    
    res = df.classification.value_counts().to_dict()

    res['csi_mean'] = df.csi.mean()
    res['entropy_purity_mean'] = df.entropy_purity.mean()
    res['gini_purity'] = df.gini_purity.mean()
    res['total_neurons'] = len(df)

    weights = df.class_frequencies.apply(sum)

    res['weighted_csi'] = np.average(df.csi, weights=weights)
    res['weighted_entropy_purity_mean'] = np.average(df.entropy_purity, weights=weights)
    res['weighted_gini_purity'] = np.average(df.gini_purity, weights=weights)
        
    return res


def calculate_neuron_metrics(df):

    class_cols = [i for i in range(10)]
    class_frequencies = df[class_cols].values
    
    metrics = []
    
    for idx, row in df.iterrows():
        neuron_id = row['neuron_id']
        freqs = class_frequencies[idx]
        
        if freqs.sum() > 0:
            freqs = freqs / freqs.sum()
        else:
            freqs = np.ones(10) / 10  
        
        max_freq = np.max(freqs)
        other_freqs = freqs[freqs != max_freq]
        mean_other = np.mean(other_freqs) if len(other_freqs) > 0 else 0
        csi = (max_freq - mean_other) / max_freq if max_freq > 0 else 0
        
        freqs_safe = freqs + 1e-10
        entropy = -np.sum(freqs_safe * np.log(freqs_safe))
        max_entropy = np.log(10)  
        entropy_purity = 1 - (entropy / max_entropy)
        
        gini = 1 - np.sum(freqs**2)
        gini_purity = 1 - gini
        
        sorted_freqs = np.sort(freqs)[::-1]  
        top2_ratio = sorted_freqs[1] / sorted_freqs[0] if sorted_freqs[0] > 0 else 0
        
        threshold = 0.1
        active_classes = np.sum(freqs > threshold)
        
        dominant_class = np.argmax(freqs)
        dominant_freq = max_freq
                
        if csi > 0.7:
            classification = "Pure"
        elif csi > 0.3:
            classification = "Compositional"
        else:
            classification = "Distributed"
        
        metrics.append({
            'neuron_id': neuron_id,
            'max_activation': row['max_activation'],
            'min_activation': row['min_activation'],
            'dominant_class': dominant_class,
            'dominant_frequency': dominant_freq,
            'csi': csi,
            'entropy_purity': entropy_purity,
            'gini_purity': gini_purity,
            'top2_ratio': top2_ratio,
            'active_classes_count': active_classes,
            'classification': classification,
            'class_frequencies': class_frequencies[idx] 
        })
    
    return pd.DataFrame(metrics)


def get_frequency_counts(activations, labels):

    max_vals = activations.max(dim=0).values  
    min_vals = activations.min(dim=0).values 
    
    df = pd.DataFrame({
            'neuron_id': range(activations.shape[1]),
            'max_activation': max_vals.numpy(),
            'min_activation': min_vals.numpy()
        })

    filtered_labels_per_active_neuron = {}

    for _, row in df.iterrows():
        neuron_id = int(row['neuron_id'])

        max_val = row['max_activation']
        lower_bound = max_val * 0.5
        upper_bound = max_val

        mask = (activations[:, neuron_id] >= lower_bound) & (activations[:, neuron_id] <= upper_bound)

        filtered_labels = labels[mask]

        filtered_labels_per_active_neuron[neuron_id] = filtered_labels.tolist()

    summary_rows = []

    for neuron_id, label_list in filtered_labels_per_active_neuron.items():

        label_counts = pd.Series(label_list).value_counts()

        row = {'neuron_id': neuron_id}

        for class_label in range(10):

            row[class_label] = label_counts.get(class_label, 0)

        summary_rows.append(row)
    
    summary_df = pd.DataFrame(summary_rows)

    df_merged = pd.merge(df, summary_df, on='neuron_id', how='inner')

    return df_merged


def extract_purity_scores(activations, labels):

    summary = get_frequency_counts(activations, labels)
    
    neuron_metrics = calculate_neuron_metrics(summary)
        
    return analyze_sae(neuron_metrics)
