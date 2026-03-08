import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from matplotlib.patches import Patch
import os

def visualize_fgsm_results(results_unsw, results_cicids, model_unsw, model_cicids,
                          X_test_sample_unsw, X_adv_unsw, y_test_sample_unsw,
                          X_test_sample_cicids, X_adv_cicids, y_test_sample_cicids,
                          feature_names_unsw, feature_names_cicids, save_path="../results/fgsm_visualizations"):
    """
    Visualisation complète des résultats FGSM avec graphiques avancés
    """
    
    print("\n📊 CRÉATION DES VISUALISATIONS FGSM...")
    os.makedirs(save_path, exist_ok=True)
    
    # Style des graphiques
    plt.style.use('seaborn-v0_8')
    sns.set_palette("husl")
    
    # ============================================================
    # 1. COMPARAISON GLOBALE DES PERFORMANCES
    # ============================================================
    
    fig, axes = plt.subplots(2, 3, figsize=(20, 12))
    fig.suptitle('📊 ANALYSE COMPLÈTE DES ATTAQUES FGSM', fontsize=16, fontweight='bold')
    
    # 1.1 Barplot comparatif UNSW vs CICIDS
    ax1 = axes[0, 0]
    datasets = ['UNSW-NB15', 'CICIDS2017']
    original_acc = [results_unsw['original_accuracy'], results_cicids['original_accuracy']]
    adversarial_acc = [results_unsw['adversarial_accuracy'], results_cicids['adversarial_accuracy']]
    
    x = np.arange(len(datasets))
    width = 0.35
    
    bars1 = ax1.bar(x - width/2, original_acc, width, label='Original', alpha=0.8, color='lightgreen')
    bars2 = ax1.bar(x + width/2, adversarial_acc, width, label='Adversarial', alpha=0.8, color='lightcoral')
    
    # Ajouter les valeurs sur les barres
    for bar, acc in zip(bars1, original_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    for bar, acc in zip(bars2, adversarial_acc):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{acc:.3f}', ha='center', va='bottom', fontweight='bold')
    
    ax1.set_xlabel('Datasets')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('Comparaison des Performances\nAvant/Après Attaque FGSM')
    ax1.set_xticks(x)
    ax1.set_xticklabels(datasets)
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # 1.2 Taux de succès des attaques
    ax2 = axes[0, 1]
    attack_success = [results_unsw['attack_success_rate'], results_cicids['attack_success_rate']]
    
    bars = ax2.bar(datasets, attack_success, color=['skyblue', 'salmon'], alpha=0.8)
    
    for bar, rate in zip(bars, attack_success):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold', fontsize=12)
    
    ax2.set_ylabel('Taux de Succès')
    ax2.set_title('Efficacité des Attaques FGSM')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.3)
    
    # 1.3 Heatmap de comparaison
    ax3 = axes[0, 2]
    comparison_data = pd.DataFrame({
        'UNSW-NB15': [results_unsw['original_accuracy'], results_unsw['adversarial_accuracy'], results_unsw['attack_success_rate']],
        'CICIDS2017': [results_cicids['original_accuracy'], results_cicids['adversarial_accuracy'], results_cicids['attack_success_rate']]
    }, index=['Accuracy Orig', 'Accuracy Adv', 'Succès Attaque'])
    
    sns.heatmap(comparison_data, annot=True, fmt='.3f', cmap='RdYlGn_r', 
                ax=ax3, cbar_kws={'label': 'Valeur'})
    ax3.set_title('Heatmap de Comparaison\nEntre Datasets')
    
    # ============================================================
    # 2. ANALYSE DES PERTURBATIONS
    # ============================================================
    
    # 2.1 Distribution des perturbations UNSW
    ax4 = axes[1, 0]
    perturbations_unsw = X_adv_unsw - X_test_sample_unsw.values
    
    # Prendre un échantillon de features pour la lisibilité
    sample_features_unsw = min(10, len(feature_names_unsw))
    feature_sample_indices = np.random.choice(len(feature_names_unsw), sample_features_unsw, replace=False)
    
    # Boxplot des perturbations par feature
    perturbation_data_unsw = []
    for idx in feature_sample_indices:
        for pert in perturbations_unsw[:, idx]:
            perturbation_data_unsw.append({
                'Feature': feature_names_unsw[idx],
                'Perturbation': pert
            })
    
    perturbation_df_unsw = pd.DataFrame(perturbation_data_unsw)
    
    sns.boxplot(data=perturbation_df_unsw, x='Feature', y='Perturbation', ax=ax4)
    ax4.set_title('Distribution des Perturbations FGSM\nUNSW-NB15')
    ax4.set_xticklabels(ax4.get_xticklabels(), rotation=45)
    ax4.grid(True, alpha=0.3)
    
    # 2.2 Distribution des perturbations CICIDS
    ax5 = axes[1, 1]
    perturbations_cicids = X_adv_cicids - X_test_sample_cicids.values
    
    sample_features_cicids = min(10, len(feature_names_cicids))
    feature_sample_indices_cicids = np.random.choice(len(feature_names_cicids), sample_features_cicids, replace=False)
    
    perturbation_data_cicids = []
    for idx in feature_sample_indices_cicids:
        for pert in perturbations_cicids[:, idx]:
            perturbation_data_cicids.append({
                'Feature': feature_names_cicids[idx],
                'Perturbation': pert
            })
    
    perturbation_df_cicids = pd.DataFrame(perturbation_data_cicids)
    
    sns.boxplot(data=perturbation_df_cicids, x='Feature', y='Perturbation', ax=ax5)
    ax5.set_title('Distribution des Perturbations FGSM\nCICIDS2017')
    ax5.set_xticklabels(ax5.get_xticklabels(), rotation=45)
    ax5.grid(True, alpha=0.3)
    
    # 2.3 Impact sur les prédictions
    ax6 = axes[1, 2]
    
    # Calculer les changements de prédiction
    with torch.no_grad():
        # UNSW
        X_test_unsw_tensor = torch.FloatTensor(X_test_sample_unsw.values)
        y_pred_orig_unsw = torch.argmax(model_unsw(X_test_unsw_tensor), dim=1).numpy()
        y_pred_adv_unsw = torch.argmax(model_unsw(torch.FloatTensor(X_adv_unsw)), dim=1).numpy()
        
        # CICIDS
        X_test_cicids_tensor = torch.FloatTensor(X_test_sample_cicids.values)
        y_pred_orig_cicids = torch.argmax(model_cicids(X_test_cicids_tensor), dim=1).numpy()
        y_pred_adv_cicids = torch.argmax(model_cicids(torch.FloatTensor(X_adv_cicids)), dim=1).numpy()
    
    changes_data = {
        'Dataset': ['UNSW-NB15', 'CICIDS2017'],
        'Prédictions Changées': [
            np.sum(y_pred_orig_unsw != y_pred_adv_unsw) / len(y_pred_orig_unsw),
            np.sum(y_pred_orig_cicids != y_pred_adv_cicids) / len(y_pred_orig_cicids)
        ]
    }
    
    changes_df = pd.DataFrame(changes_data)
    bars = ax6.bar(changes_df['Dataset'], changes_df['Prédictions Changées'], 
                   color=['lightblue', 'lightcoral'], alpha=0.8)
    
    for bar, rate in zip(bars, changes_df['Prédictions Changées']):
        ax6.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01, 
                f'{rate:.1%}', ha='center', va='bottom', fontweight='bold')
    
    ax6.set_ylabel('Pourcentage de Prédictions Changées')
    ax6.set_title('Impact des Perturbations\nsur les Prédictions')
    ax6.set_ylim(0, 1)
    ax6.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Sauvegarde du premier graphique
    plt.savefig(f"{save_path}/fgsm_comprehensive_analysis.png", dpi=300, bbox_inches='tight')
    print(f"💾 Graphique principal sauvegardé: {save_path}/fgsm_comprehensive_analysis.png")
    
    # ============================================================
    # 3. GRAPHIQUE DÉTAILLÉ PAR DATASET
    # ============================================================
    
    # Graphique détaillé pour UNSW
    fig_unsw, axes_unsw = plt.subplots(2, 2, figsize=(15, 10))
    fig_unsw.suptitle('📈 ANALYSE DÉTAILLÉE FGSM - UNSW-NB15', fontsize=14, fontweight='bold')
    
    # 3.1 Matrice de confusion UNSW
    ax1 = axes_unsw[0, 0]
    from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
    
    cm_orig_unsw = confusion_matrix(y_test_sample_unsw, y_pred_orig_unsw)
    cm_adv_unsw = confusion_matrix(y_test_sample_unsw, y_pred_adv_unsw)
    
    disp_orig = ConfusionMatrixDisplay(confusion_matrix=cm_orig_unsw)
    disp_orig.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Matrice de Confusion - Original\nUNSW-NB15')
    
    # 3.2 Matrice de confusion adversarial UNSW
    ax2 = axes_unsw[0, 1]
    disp_adv = ConfusionMatrixDisplay(confusion_matrix=cm_adv_unsw)
    disp_adv.plot(ax=ax2, cmap='Reds')
    ax2.set_title('Matrice de Confusion - Adversarial\nUNSW-NB15')
    
    # 3.3 Distribution des scores de confiance UNSW
    ax3 = axes_unsw[1, 0]
    with torch.no_grad():
        confidences_orig_unsw = torch.softmax(model_unsw(torch.FloatTensor(X_test_sample_unsw.values)), dim=1).numpy()
        confidences_adv_unsw = torch.softmax(model_unsw(torch.FloatTensor(X_adv_unsw)), dim=1).numpy()
    
    # Scores de confiance pour la classe correcte
    correct_confidences_orig = [confidences_orig_unsw[i, y_true] for i, y_true in enumerate(y_test_sample_unsw)]
    correct_confidences_adv = [confidences_adv_unsw[i, y_true] for i, y_true in enumerate(y_test_sample_unsw)]
    
    ax3.hist(correct_confidences_orig, bins=20, alpha=0.7, label='Original', color='green')
    ax3.hist(correct_confidences_adv, bins=20, alpha=0.7, label='Adversarial', color='red')
    ax3.set_xlabel('Confiance dans la Classe Correcte')
    ax3.set_ylabel('Fréquence')
    ax3.set_title('Distribution de la Confiance\nUNSW-NB15')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # 3.4 Features les plus perturbées UNSW
    ax4 = axes_unsw[1, 1]
    mean_perturbations_unsw = np.mean(np.abs(perturbations_unsw), axis=0)
    top_features_idx_unsw = np.argsort(mean_perturbations_unsw)[-10:]  # Top 10 features
    
    features_plot_unsw = [feature_names_unsw[i] for i in top_features_idx_unsw]
    perturbations_plot_unsw = mean_perturbations_unsw[top_features_idx_unsw]
    
    ax4.barh(features_plot_unsw, perturbations_plot_unsw, color='orange', alpha=0.7)
    ax4.set_xlabel('Perturbation Moyenne (abs)')
    ax4.set_title('Top 10 Features les Plus Perturbées\nUNSW-NB15')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/fgsm_detailed_unsw.png", dpi=300, bbox_inches='tight')
    print(f"💾 Graphique UNSW détaillé sauvegardé: {save_path}/fgsm_detailed_unsw.png")
    
    # Graphique détaillé pour CICIDS
    fig_cicids, axes_cicids = plt.subplots(2, 2, figsize=(15, 10))
    fig_cicids.suptitle('📈 ANALYSE DÉTAILLÉE FGSM - CICIDS2017', fontsize=14, fontweight='bold')
    
    # Matrice de confusion CICIDS
    ax1 = axes_cicids[0, 0]
    cm_orig_cicids = confusion_matrix(y_test_sample_cicids, y_pred_orig_cicids)
    cm_adv_cicids = confusion_matrix(y_test_sample_cicids, y_pred_adv_cicids)
    
    disp_orig_cicids = ConfusionMatrixDisplay(confusion_matrix=cm_orig_cicids)
    disp_orig_cicids.plot(ax=ax1, cmap='Blues')
    ax1.set_title('Matrice de Confusion - Original\nCICIDS2017')
    
    ax2 = axes_cicids[0, 1]
    disp_adv_cicids = ConfusionMatrixDisplay(confusion_matrix=cm_adv_cicids)
    disp_adv_cicids.plot(ax=ax2, cmap='Reds')
    ax2.set_title('Matrice de Confusion - Adversarial\nCICIDS2017')
    
    # Distribution des scores de confiance CICIDS
    ax3 = axes_cicids[1, 0]
    with torch.no_grad():
        confidences_orig_cicids = torch.softmax(model_cicids(torch.FloatTensor(X_test_sample_cicids.values)), dim=1).numpy()
        confidences_adv_cicids = torch.softmax(model_cicids(torch.FloatTensor(X_adv_cicids)), dim=1).numpy()
    
    correct_confidences_orig_cicids = [confidences_orig_cicids[i, y_true] for i, y_true in enumerate(y_test_sample_cicids)]
    correct_confidences_adv_cicids = [confidences_adv_cicids[i, y_true] for i, y_true in enumerate(y_test_sample_cicids)]
    
    ax3.hist(correct_confidences_orig_cicids, bins=20, alpha=0.7, label='Original', color='green')
    ax3.hist(correct_confidences_adv_cicids, bins=20, alpha=0.7, label='Adversarial', color='red')
    ax3.set_xlabel('Confiance dans la Classe Correcte')
    ax3.set_ylabel('Fréquence')
    ax3.set_title('Distribution de la Confiance\nCICIDS2017')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Features les plus perturbées CICIDS
    ax4 = axes_cicids[1, 1]
    mean_perturbations_cicids = np.mean(np.abs(perturbations_cicids), axis=0)
    top_features_idx_cicids = np.argsort(mean_perturbations_cicids)[-10:]
    
    features_plot_cicids = [feature_names_cicids[i] for i in top_features_idx_cicids]
    perturbations_plot_cicids = mean_perturbations_cicids[top_features_idx_cicids]
    
    ax4.barh(features_plot_cicids, perturbations_plot_cicids, color='purple', alpha=0.7)
    ax4.set_xlabel('Perturbation Moyenne (abs)')
    ax4.set_title('Top 10 Features les Plus Perturbées\nCICIDS2017')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f"{save_path}/fgsm_detailed_cicids.png", dpi=300, bbox_inches='tight')
    print(f"💾 Graphique CICIDS détaillé sauvegardé: {save_path}/fgsm_detailed_cicids.png")
    
    # ============================================================
    # 4. RAPPORT TEXTUEL RÉCAPITULATIF
    # ============================================================
    
    print("\n" + "="*60)
    print("📋 RAPPORT RÉCAPITULATIF DES ATTAQUES FGSM")
    print("="*60)
    
    print(f"\n🎯 PERFORMANCES GLOBALES:")
    print(f"   UNSW-NB15:")
    print(f"   • Accuracy originale: {results_unsw['original_accuracy']:.4f}")
    print(f"   • Accuracy adversarial: {results_unsw['adversarial_accuracy']:.4f}")
    print(f"   • Taux de succès attaque: {results_unsw['attack_success_rate']:.4f}")
    print(f"   • Prédictions changées: {np.sum(y_pred_orig_unsw != y_pred_adv_unsw)}/{len(y_pred_orig_unsw)}")
    
    print(f"\n   CICIDS2017:")
    print(f"   • Accuracy originale: {results_cicids['original_accuracy']:.4f}")
    print(f"   • Accuracy adversarial: {results_cicids['adversarial_accuracy']:.4f}")
    print(f"   • Taux de succès attaque: {results_cicids['attack_success_rate']:.4f}")
    print(f"   • Prédictions changées: {np.sum(y_pred_orig_cicids != y_pred_adv_cicids)}/{len(y_pred_orig_cicids)}")
    
    print(f"\n🔍 ANALYSE DE ROBUSTESSE:")
    robust_unsw = results_unsw['adversarial_accuracy'] / results_unsw['original_accuracy']
    robust_cicids = results_cicids['adversarial_accuracy'] / results_cicids['original_accuracy']
    
    print(f"   UNSW-NB15 - Robustesse: {robust_unsw:.2%}")
    print(f"   CICIDS2017 - Robustesse: {robust_cicids:.2%}")
    
    if robust_unsw > 0.8:
        print("   ✅ UNSW: Bonne robustesse adversarial")
    else:
        print("   ⚠️  UNSW: Robustesse adversarial à améliorer")
    
    if robust_cicids > 0.8:
        print("   ✅ CICIDS: Bonne robustesse adversarial")
    else:
        print("   ⚠️  CICIDS: Robustesse adversarial à améliorer")
    
    print(f"\n📊 FICHIERS GÉNÉRÉS:")
    print(f"   • {save_path}/fgsm_comprehensive_analysis.png")
    print(f"   • {save_path}/fgsm_detailed_unsw.png")
    print(f"   • {save_path}/fgsm_detailed_cicids.png")
    
    plt.show()
    
    return {
        'robustness_unsw': robust_unsw,
        'robustness_cicids': robust_cicids,
        'predictions_changed_unsw': np.sum(y_pred_orig_unsw != y_pred_adv_unsw),
        'predictions_changed_cicids': np.sum(y_pred_orig_cicids != y_pred_adv_cicids)
    }