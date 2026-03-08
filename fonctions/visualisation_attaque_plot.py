import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score, recall_score, f1_score, confusion_matrix
import os

def plot_fgsm_results_comparison(results_dict, model_names, dataset_name, output_dir=None):
    """
    Crée des visualisations complètes pour comparer les résultats des attaques FGSM
    
    Args:
        results_dict: Dictionnaire avec les résultats pour chaque modèle
        model_names: Liste des noms des modèles
        dataset_name: Nom du dataset (CICIDS/UNSW)
        output_dir: Dossier pour sauvegarder les graphiques
    """
    
    # Configuration des styles
    plt.style.use('default')
    sns.set_palette("husl")
    
    # Création des figures
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle(f'Analyse des Attaques FGSM - {dataset_name}', fontsize=16, fontweight='bold')
    
    # Données pour les graphiques
    epsilons = [0.01, 0.05, 0.1]
    metrics_clean = {'accuracy': [], 'recall': [], 'f1': []}
    metrics_adv = {'accuracy': [], 'recall': [], 'f1': []}
    fn_data = {'clean': [], 'adv': [], 'increase': []}
    
    # Extraction des données
    for model_name in model_names:
        model_results = results_dict[model_name]
        
        acc_clean = model_results['accuracy_clean']
        acc_adv = model_results['accuracy_adv']
        rec_clean = model_results['recall_clean']
        rec_adv = model_results['recall_adv']
        f1_clean = model_results['f1_clean']
        f1_adv = model_results['f1_adv']
        fn_clean = model_results['fn_clean']
        fn_adv = model_results['fn_adv']
        
        metrics_clean['accuracy'].extend([acc_clean] * 3)
        metrics_clean['recall'].extend([rec_clean] * 3)
        metrics_clean['f1'].extend([f1_clean] * 3)
        
        metrics_adv['accuracy'].extend(acc_adv)
        metrics_adv['recall'].extend(rec_adv)
        metrics_adv['f1'].extend(f1_adv)
        
        fn_data['clean'].extend([fn_clean] * 3)
        fn_data['adv'].extend(fn_adv)
        fn_data['increase'].extend([adv - clean for clean, adv in zip([fn_clean]*3, fn_adv)])
    
    # 1. Évolution de l'Accuracy avec epsilon
    ax = axes[0, 0]
    for i, model_name in enumerate(model_names):
        acc_clean = results_dict[model_name]['accuracy_clean']
        acc_adv = results_dict[model_name]['accuracy_adv']
        
        ax.plot(epsilons, acc_adv, marker='o', linewidth=2.5, label=f'{model_name} (après attaque)')
        ax.axhline(y=acc_clean, color=f'C{i}', linestyle='--', alpha=0.7, label=f'{model_name} (clean)')
    
    ax.set_xlabel('Epsilon (ε)')
    ax.set_ylabel('Accuracy')
    ax.set_title('Impact de FGSM sur l\'Accuracy', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epsilons)
    
    # 2. Évolution du Recall avec epsilon
    ax = axes[0, 1]
    for i, model_name in enumerate(model_names):
        rec_clean = results_dict[model_name]['recall_clean']
        rec_adv = results_dict[model_name]['recall_adv']
        
        ax.plot(epsilons, rec_adv, marker='s', linewidth=2.5, label=f'{model_name} (après attaque)')
        ax.axhline(y=rec_clean, color=f'C{i}', linestyle='--', alpha=0.7, label=f'{model_name} (clean)')
    
    ax.set_xlabel('Epsilon (ε)')
    ax.set_ylabel('Recall')
    ax.set_title('Impact de FGSM sur le Recall', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epsilons)
    
    # 3. Évolution du F1-Score avec epsilon
    ax = axes[0, 2]
    for i, model_name in enumerate(model_names):
        f1_clean = results_dict[model_name]['f1_clean']
        f1_adv = results_dict[model_name]['f1_adv']
        
        ax.plot(epsilons, f1_adv, marker='^', linewidth=2.5, label=f'{model_name} (après attaque)')
        ax.axhline(y=f1_clean, color=f'C{i}', linestyle='--', alpha=0.7, label=f'{model_name} (clean)')
    
    ax.set_xlabel('Epsilon (ε)')
    ax.set_ylabel('F1-Score')
    ax.set_title('Impact de FGSM sur le F1-Score', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epsilons)
    
    # 4. Augmentation des Faux Négatifs
    ax = axes[1, 0]
    bar_width = 0.25
    x_pos = np.arange(len(epsilons))
    
    for i, model_name in enumerate(model_names):
        fn_increase = results_dict[model_name]['fn_increase']
        ax.bar(x_pos + i * bar_width, fn_increase, bar_width, label=model_name)
    
    ax.set_xlabel('Epsilon (ε)')
    ax.set_ylabel('Augmentation des Faux Négatifs')
    ax.set_title('Augmentation des Faux Négatifs après FGSM', fontweight='bold')
    ax.set_xticks(x_pos + bar_width)
    ax.set_xticklabels(epsilons)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # 5. Heatmap de robustesse (accuracy après attaque / accuracy clean)
    ax = axes[1, 1]
    robustness_data = []
    for model_name in model_names:
        model_results = results_dict[model_name]
        robustness = [adv / model_results['accuracy_clean'] for adv in model_results['accuracy_adv']]
        robustness_data.append(robustness)
    
    robustness_df = pd.DataFrame(robustness_data, index=model_names, columns=epsilons)
    sns.heatmap(robustness_df, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                cbar_kws={'label': 'Ratio de Robustesse'}, ax=ax)
    ax.set_title('Robustesse des Modèles (Accuracy après/avant)', fontweight='bold')
    ax.set_xlabel('Epsilon (ε)')
    
    # 6. Performance relative drop
    ax = axes[1, 2]
    performance_drop = []
    for model_name in model_names:
        model_results = results_dict[model_name]
        drop = [(clean - adv) / clean * 100 for clean, adv in 
                zip([model_results['accuracy_clean']]*3, model_results['accuracy_adv'])]
        performance_drop.append(drop)
    
    x = np.arange(len(epsilons))
    for i, drops in enumerate(performance_drop):
        ax.plot(epsilons, drops, marker='D', linewidth=2.5, label=model_names[i])
    
    ax.set_xlabel('Epsilon (ε)')
    ax.set_ylabel('Baisse de Performance (%)')
    ax.set_title('Baisse Relative de Performance', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xticks(epsilons)
    
    plt.tight_layout()
    
    # Sauvegarde
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f'{output_dir}/fgsm_analysis_{dataset_name}.png', dpi=300, bbox_inches='tight')
        plt.savefig(f'{output_dir}/fgsm_analysis_{dataset_name}.pdf', bbox_inches='tight')
    
    plt.show()
    
    # Rapport texte
    print_general_report(results_dict, model_names, dataset_name)

def print_general_report(results_dict, model_names, dataset_name):
    """Génère un rapport texte des résultats"""
    print("="*70)
    print(f"RAPPORT D'ANALYSE FGSM - {dataset_name}")
    print("="*70)
    
    for model_name in model_names:
        model_results = results_dict[model_name]
        print(f"\n📊 {model_name}:")
        print(f"   Accuracy: {model_results['accuracy_clean']:.4f} → {model_results['accuracy_adv'][-1]:.4f} "
              f"(Δ: {model_results['accuracy_clean'] - model_results['accuracy_adv'][-1]:.4f})")
        print(f"   Recall:   {model_results['recall_clean']:.4f} → {model_results['recall_adv'][-1]:.4f} "
              f"(Δ: {model_results['recall_clean'] - model_results['recall_adv'][-1]:.4f})")
        print(f"   F1-Score: {model_results['f1_clean']:.4f} → {model_results['f1_adv'][-1]:.4f} "
              f"(Δ: {model_results['f1_clean'] - model_results['f1_adv'][-1]:.4f})")
        print(f"   Faux Négatifs: {model_results['fn_clean']} → {model_results['fn_adv'][-1]} "
              f"(+{model_results['fn_increase'][-1]})")
        
        robustness = model_results['accuracy_adv'][-1] / model_results['accuracy_clean']
        print(f"   Robustesse: {robustness:.2%}")

def create_fgsm_results_dict():
    """
    Crée le dictionnaire de résultats à partir de vos exécutions
    Adaptez cette fonction selon vos variables réelles
    """
    results_dict = {
        'CICIDS_noeq': {
            'accuracy_clean': 0.9952,
            'accuracy_adv': [0.9886, 0.9806, 0.9806],  # ε=0.01, 0.05, 0.1
            'recall_clean': 0.9901,
            'recall_adv': [0.9845, 0.9561, 0.9562],
            'f1_clean': 0.9875,
            'f1_adv': [0.9706, 0.9497, 0.9498],
            'fn_clean': 741,
            'fn_adv': [1154, 3270, 3262],
            'fn_increase': [413, 2529, 2521]
        },
        'CICIDS_smote': {
            'accuracy_clean': 0.9608,
            'accuracy_adv': [0.9566, 0.9504, 0.9502],
            'recall_clean': 0.9956,
            'recall_adv': [0.9951, 0.9946, 0.9944],
            'f1_clean': 0.9068,
            'f1_adv': [0.8979, 0.8849, 0.8844],
            'fn_clean': 331,
            'fn_adv': [363, 404, 414],
            'fn_increase': [32, 73, 83]
        },
        'UNSW_noeq': {
            'accuracy_clean': 0.9172,
            'accuracy_adv': [0.8940, 0.8808, 0.8797],
            'recall_clean': 0.9274,
            'recall_adv': [0.9077, 0.8978, 0.8970],
            'f1_clean': 0.9350,
            'f1_adv': [0.9166, 0.9063, 0.9055],
            'fn_clean': 1775,
            'fn_adv': [2256, 2499, 2518],
            'fn_increase': [481, 724, 743]
        }
    }
    return results_dict

# Exemple d'utilisation pour CICIDS
results_dict = create_fgsm_results_dict()
model_names = ['CICIDS_noeq', 'CICIDS_smote']
plot_fgsm_results_comparison(
    results_dict=results_dict,
    model_names=model_names,
    dataset_name='CICIDS',
    output_dir='../analysis/FGSM'
)

# Exemple d'utilisation pour UNSW
model_names_unsw = ['UNSW_noeq']
plot_fgsm_results_comparison(
    results_dict=results_dict,
    model_names=model_names_unsw,
    dataset_name='UNSW',
    output_dir='../analysis/FGSM'
)