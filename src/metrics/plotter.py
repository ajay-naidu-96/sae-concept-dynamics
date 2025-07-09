import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

class SAEResultsVisualizer:
    def __init__(self):
        self.per_neuron_results = None
        self.per_neuron_1vsall_results = None
        
    def load_results(self, per_neuron_pkl_path=None, per_neuron_1vsall_pkl_path=None):
        """Load results from pickle files"""
        if per_neuron_pkl_path:
            with open(per_neuron_pkl_path, 'rb') as f:
                self.per_neuron_results = pickle.load(f)
            print(f"Loaded per_neuron results with {len(self.per_neuron_results)} SAEs")
            
        if per_neuron_1vsall_pkl_path:
            with open(per_neuron_1vsall_pkl_path, 'rb') as f:
                self.per_neuron_1vsall_results = pickle.load(f)
            print(f"Loaded per_neuron_1vsall results with {len(self.per_neuron_1vsall_results)} SAEs")
    
    def plot_per_neuron_overview(self, figsize=(15, 10)):
        """Plot overview of per-neuron results across all SAEs"""
        if not self.per_neuron_results:
            print("No per_neuron results loaded!")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('Per-Neuron Classification Results Overview', fontsize=16, fontweight='bold')
        
        # Collect summary stats
        sae_names = []
        test_accs = []
        mean_accs = []
        top_accs = []
        neurons_above_random = []
        neurons_above_50 = []
        sparsities = []
        
        for sae_name, results in self.per_neuron_results.items():
            summary = results['summary']
            sae_names.append(sae_name)
            test_accs.append(summary.get('top_neuron_accuracy', 0))
            mean_accs.append(summary.get('mean_neuron_accuracy', 0))
            top_accs.append(summary.get('top_10_mean', 0))
            neurons_above_random.append(summary.get('neurons_above_random', 0) / summary.get('n_neurons', 1))
            neurons_above_50.append(summary.get('neurons_above_50', 0) / summary.get('n_neurons', 1))
            sparsities.append(summary.get('mean_sparsity', 0))
        
        # 1. Best neuron accuracy per SAE
        axes[0,0].bar(range(len(sae_names)), test_accs, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Best Neuron Accuracy per SAE')
        axes[0,0].set_ylabel('Test Accuracy')
        axes[0,0].set_xticks(range(len(sae_names)))
        axes[0,0].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Mean neuron accuracy
        axes[0,1].bar(range(len(sae_names)), mean_accs, color='lightgreen', alpha=0.7)
        axes[0,1].set_title('Mean Neuron Accuracy per SAE')
        axes[0,1].set_ylabel('Test Accuracy')
        axes[0,1].set_xticks(range(len(sae_names)))
        axes[0,1].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Top-10 mean accuracy
        axes[0,2].bar(range(len(sae_names)), top_accs, color='orange', alpha=0.7)
        axes[0,2].set_title('Top-10 Neurons Mean Accuracy')
        axes[0,2].set_ylabel('Test Accuracy')
        axes[0,2].set_xticks(range(len(sae_names)))
        axes[0,2].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Fraction of neurons above random
        axes[1,0].bar(range(len(sae_names)), neurons_above_random, color='coral', alpha=0.7)
        axes[1,0].set_title('Fraction of Neurons > Random (0.1)')
        axes[1,0].set_ylabel('Fraction')
        axes[1,0].set_xticks(range(len(sae_names)))
        axes[1,0].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Fraction above 50%
        axes[1,1].bar(range(len(sae_names)), neurons_above_50, color='purple', alpha=0.7)
        axes[1,1].set_title('Fraction of Neurons > 50%')
        axes[1,1].set_ylabel('Fraction')
        axes[1,1].set_xticks(range(len(sae_names)))
        axes[1,1].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Mean sparsity
        axes[1,2].bar(range(len(sae_names)), sparsities, color='gold', alpha=0.7)
        axes[1,2].set_title('Mean Neuron Sparsity')
        axes[1,2].set_ylabel('Sparsity')
        axes[1,2].set_xticks(range(len(sae_names)))
        axes[1,2].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_per_neuron_distributions(self, figsize=(15, 5)):
        """Plot distributions of neuron accuracies"""
        if not self.per_neuron_results:
            print("No per_neuron results loaded!")
            return
            
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        fig.suptitle('Neuron Accuracy Distributions', fontsize=16, fontweight='bold')
        
        all_accuracies = []
        sae_labels = []
        
        for sae_name, results in self.per_neuron_results.items():
            df = results['per_neuron_df']
            all_accuracies.extend(df['test_accuracy'].values)
            sae_labels.extend([sae_name] * len(df))
        
        # 1. Overall histogram
        axes[0].hist(all_accuracies, bins=50, alpha=0.7, color='skyblue', edgecolor='black')
        axes[0].axvline(0.1, color='red', linestyle='--', label='Random (0.1)')
        axes[0].axvline(0.5, color='orange', linestyle='--', label='50%')
        axes[0].set_title('Overall Neuron Accuracy Distribution')
        axes[0].set_xlabel('Test Accuracy')
        axes[0].set_ylabel('Count')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # 2. Box plot by SAE
        df_all = pd.DataFrame({'accuracy': all_accuracies, 'sae': sae_labels})
        sns.boxplot(data=df_all, x='sae', y='accuracy', ax=axes[1])
        axes[1].set_title('Accuracy Distribution by SAE')
        axes[1].set_xlabel('SAE')
        axes[1].set_ylabel('Test Accuracy')
        axes[1].tick_params(axis='x', rotation=45)
        axes[1].grid(True, alpha=0.3)
        
        # 3. Top neurons comparison
        top_neurons_data = []
        for sae_name, results in self.per_neuron_results.items():
            df = results['per_neuron_df']
            top_10 = df.head(10)['test_accuracy'].values
            for i, acc in enumerate(top_10):
                top_neurons_data.append({'sae': sae_name, 'rank': i+1, 'accuracy': acc})
        
        top_df = pd.DataFrame(top_neurons_data)
        for sae_name in self.per_neuron_results.keys():
            sae_data = top_df[top_df['sae'] == sae_name]
            axes[2].plot(sae_data['rank'], sae_data['accuracy'], marker='o', label=sae_name)
        
        axes[2].set_title('Top-10 Neurons Comparison')
        axes[2].set_xlabel('Neuron Rank')
        axes[2].set_ylabel('Test Accuracy')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_1vsall_overview(self, figsize=(15, 10)):
        """Plot overview of 1-vs-all results"""
        if not self.per_neuron_1vsall_results:
            print("No per_neuron_1vsall results loaded!")
            return
            
        fig, axes = plt.subplots(2, 3, figsize=figsize)
        fig.suptitle('1-vs-All Classification Results Overview', fontsize=16, fontweight='bold')
        
        # Collect summary stats
        sae_names = []
        total_neurons = []
        digit_coverage = []
        fraction_selective = []
        mean_best_auc = []
        top_10_auc = []
        entropy_scores = []
        
        for sae_name, results in self.per_neuron_1vsall_results.items():
            summary = results.get('sae_summary', {})
            sae_names.append(sae_name)
            total_neurons.append(summary.get('total_neurons', 0))
            digit_coverage.append(summary.get('digit_coverage', 0))
            fraction_selective.append(summary.get('fraction_selective', 0))
            mean_best_auc.append(summary.get('mean_best_auc', 0))
            top_10_auc.append(summary.get('top_10_auc_mean', 0))
            entropy_scores.append(summary.get('entropy_best_class_assignments', 0))
        
        # 1. Total neurons
        axes[0,0].bar(range(len(sae_names)), total_neurons, color='skyblue', alpha=0.7)
        axes[0,0].set_title('Total Neurons per SAE')
        axes[0,0].set_ylabel('Number of Neurons')
        axes[0,0].set_xticks(range(len(sae_names)))
        axes[0,0].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Digit coverage
        axes[0,1].bar(range(len(sae_names)), digit_coverage, color='lightgreen', alpha=0.7)
        axes[0,1].set_title('Digit Coverage (Unique Digits Detected)')
        axes[0,1].set_ylabel('Number of Digits')
        axes[0,1].set_xticks(range(len(sae_names)))
        axes[0,1].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[0,1].set_ylim(0, 10)
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Fraction selective
        axes[0,2].bar(range(len(sae_names)), fraction_selective, color='orange', alpha=0.7)
        axes[0,2].set_title('Fraction of Selective Neurons (AUC > 0.85)')
        axes[0,2].set_ylabel('Fraction')
        axes[0,2].set_xticks(range(len(sae_names)))
        axes[0,2].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[0,2].grid(True, alpha=0.3)
        
        # 4. Mean best AUC
        axes[1,0].bar(range(len(sae_names)), mean_best_auc, color='coral', alpha=0.7)
        axes[1,0].set_title('Mean Best AUC per Neuron')
        axes[1,0].set_ylabel('AUC')
        axes[1,0].set_xticks(range(len(sae_names)))
        axes[1,0].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[1,0].grid(True, alpha=0.3)
        
        # 5. Top-10 AUC mean
        axes[1,1].bar(range(len(sae_names)), top_10_auc, color='purple', alpha=0.7)
        axes[1,1].set_title('Top-10 Neurons Mean AUC')
        axes[1,1].set_ylabel('AUC')
        axes[1,1].set_xticks(range(len(sae_names)))
        axes[1,1].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[1,1].grid(True, alpha=0.3)
        
        # 6. Entropy (diversity of digit assignments)
        axes[1,2].bar(range(len(sae_names)), entropy_scores, color='gold', alpha=0.7)
        axes[1,2].set_title('Entropy of Digit Assignments')
        axes[1,2].set_ylabel('Entropy')
        axes[1,2].set_xticks(range(len(sae_names)))
        axes[1,2].set_xticklabels(sae_names, rotation=45, ha='right')
        axes[1,2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def plot_1vsall_heatmaps(self, max_saes=4, figsize=(20, 5)):
        """Plot AUC heatmaps for each SAE"""
        if not self.per_neuron_1vsall_results:
            print("No per_neuron_1vsall results loaded!")
            return
            
        sae_names = list(self.per_neuron_1vsall_results.keys())[:max_saes]
        
        fig, axes = plt.subplots(1, len(sae_names), figsize=figsize)
        if len(sae_names) == 1:
            axes = [axes]
            
        fig.suptitle('AUC Heatmaps: Neuron vs Digit Selectivity', fontsize=16, fontweight='bold')
        
        for i, sae_name in enumerate(sae_names):
            results = self.per_neuron_1vsall_results[sae_name]
            auc_matrix = results.get('auc_matrix')
            neuron_stats = results.get('per_neuron_df')
            
            if auc_matrix is None or neuron_stats is None:
                axes[i].text(0.5, 0.5, f'No data for {sae_name}', 
                           ha='center', va='center', transform=axes[i].transAxes)
                axes[i].set_title(f'{sae_name}\n(No Data)')
                continue
                
            # Take top 50 neurons for visualization
            top_neurons_df = neuron_stats.sort_values('best_auc', ascending=False)
            top_neurons = top_neurons_df.head(50)['neuron_idx'].values
            
            if len(top_neurons) > 0:
                heatmap_data = auc_matrix.loc[top_neurons]
                
                sns.heatmap(heatmap_data, 
                           cmap='viridis', 
                           cbar_kws={'label': 'AUC'},
                           ax=axes[i],
                           xticklabels=True,
                           yticklabels=False)
                axes[i].set_title(f'{sae_name}\n(Top 50 Neurons)')
                axes[i].set_xlabel('Digit')
                if i == 0:
                    axes[i].set_ylabel('Neuron Index')
        
        plt.tight_layout()
        plt.show()
    
    def plot_digit_specialization(self, figsize=(12, 8)):
        """Plot how neurons specialize for different digits"""
        if not self.per_neuron_1vsall_results:
            print("No per_neuron_1vsall results loaded!")
            return
            
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        fig.suptitle('Digit Specialization Analysis', fontsize=16, fontweight='bold')
        
        # Collect data across all SAEs
        all_digit_counts = {}
        all_best_aucs = []
        all_sae_labels = []
        
        for sae_name, results in self.per_neuron_1vsall_results.items():
            neuron_stats = results.get('per_neuron_df')
            if neuron_stats is None:
                continue
                
            # Digit distribution
            digit_counts = neuron_stats['best_digit'].value_counts().sort_index()
            all_digit_counts[sae_name] = digit_counts
            
            # Best AUCs
            all_best_aucs.extend(neuron_stats['best_auc'].values)
            all_sae_labels.extend([sae_name] * len(neuron_stats))
        
        # 1. Digit specialization counts
        digit_df = pd.DataFrame(all_digit_counts).fillna(0)
        digit_df.plot(kind='bar', ax=axes[0,0], alpha=0.7)
        axes[0,0].set_title('Neurons Specialized per Digit')
        axes[0,0].set_xlabel('Digit')
        axes[0,0].set_ylabel('Number of Neurons')
        axes[0,0].legend(title='SAE', bbox_to_anchor=(1.05, 1), loc='upper left')
        axes[0,0].grid(True, alpha=0.3)
        
        # 2. Best AUC distribution
        auc_df = pd.DataFrame({'best_auc': all_best_aucs, 'sae': all_sae_labels})
        sns.boxplot(data=auc_df, x='sae', y='best_auc', ax=axes[0,1])
        axes[0,1].axhline(0.85, color='red', linestyle='--', label='Selective Threshold')
        axes[0,1].set_title('Best AUC Distribution by SAE')
        axes[0,1].set_xlabel('SAE')
        axes[0,1].set_ylabel('Best AUC')
        axes[0,1].tick_params(axis='x', rotation=45)
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)
        
        # 3. Selectivity comparison
        selectivity_data = []
        for sae_name, results in self.per_neuron_1vsall_results.items():
            summary = results.get('sae_summary', {})
            selectivity_data.append({
                'SAE': sae_name,
                'Fraction_Selective': summary.get('fraction_selective', 0),
                'Total_Neurons': summary.get('total_neurons', 0)
            })
        
        sel_df = pd.DataFrame(selectivity_data)
        scatter = axes[1,0].scatter(sel_df['Total_Neurons'], sel_df['Fraction_Selective'], 
                                   s=100, alpha=0.7, c=range(len(sel_df)), cmap='viridis')
        
        for i, row in sel_df.iterrows():
            axes[1,0].annotate(row['SAE'], (row['Total_Neurons'], row['Fraction_Selective']),
                              xytext=(5, 5), textcoords='offset points', fontsize=8)
        
        axes[1,0].set_title('Selectivity vs SAE Size')
        axes[1,0].set_xlabel('Total Neurons')
        axes[1,0].set_ylabel('Fraction Selective')
        axes[1,0].grid(True, alpha=0.3)
        
        # 4. Entropy comparison (diversity)
        entropy_data = [results.get('sae_summary', {}).get('entropy_best_class_assignments', 0) 
                       for results in self.per_neuron_1vsall_results.values()]
        sae_names_list = list(self.per_neuron_1vsall_results.keys())
        
        axes[1,1].bar(range(len(sae_names_list)), entropy_data, alpha=0.7, color='gold')
        axes[1,1].set_title('Digit Assignment Diversity (Entropy)')
        axes[1,1].set_xlabel('SAE')
        axes[1,1].set_ylabel('Entropy')
        axes[1,1].set_xticks(range(len(sae_names_list)))
        axes[1,1].set_xticklabels(sae_names_list, rotation=45, ha='right')
        axes[1,1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.show()
    
    def generate_summary_report(self):
        """Generate a text summary of all results"""
        print("="*60)
        print("SAE EVALUATION SUMMARY REPORT")
        print("="*60)
        
        if self.per_neuron_results:
            print("\n📊 PER-NEURON (Multi-class) RESULTS:")
            print("-" * 40)
            
            for sae_name, results in self.per_neuron_results.items():
                summary = results['summary']
                print(f"\n🧠 {sae_name}:")
                print(f"  • Total neurons: {summary.get('n_neurons', 'N/A')}")
                print(f"  • Best neuron accuracy: {summary.get('top_neuron_accuracy', 0):.3f}")
                print(f"  • Mean neuron accuracy: {summary.get('mean_neuron_accuracy', 0):.3f}")
                print(f"  • Top-10 mean accuracy: {summary.get('top_10_mean', 0):.3f}")
                n_neurons = summary.get('n_neurons', 1)
                above_random = summary.get('neurons_above_random', 0)
                above_50 = summary.get('neurons_above_50', 0)
                print(f"  • Neurons > random (0.1): {above_random}/{n_neurons} ({above_random/n_neurons*100:.1f}%)")
                print(f"  • Neurons > 50%: {above_50}/{n_neurons} ({above_50/n_neurons*100:.1f}%)")
                print(f"  • Mean sparsity: {summary.get('mean_sparsity', 0):.3f}")
        
        if self.per_neuron_1vsall_results:
            print("\n🎯 PER-NEURON (1-vs-All) RESULTS:")
            print("-" * 40)
            
            for sae_name, results in self.per_neuron_1vsall_results.items():
                summary = results.get('sae_summary', {})
                print(f"\n🧠 {sae_name}:")
                print(f"  • Total neurons: {summary.get('total_neurons', 'N/A')}")
                print(f"  • Digit coverage: {summary.get('digit_coverage', 0)}/10 digits")
                print(f"  • Fraction selective (AUC>0.85): {summary.get('fraction_selective', 0):.3f}")
                print(f"  • Mean best AUC: {summary.get('mean_best_auc', 0):.3f}")
                print(f"  • Top-10 mean AUC: {summary.get('top_10_auc_mean', 0):.3f}")
                print(f"  • Assignment entropy: {summary.get('entropy_best_class_assignments', 0):.3f}")
        
        print("\n" + "="*60)
