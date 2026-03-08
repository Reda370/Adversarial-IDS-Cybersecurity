import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def analyser_dataset_complet(dataframe, nom_dataset, sauvegarder_csv=True):
    """
    Analyse complète d'un dataset de détection d'intrusion
    """
    
    def analyser_features_detaille(df, nom_ds):
        """
        Sous-fonction pour l'analyse détaillée des features
        """
        print(f"\n{'='*80}")
        print(f"ANALYSE DÉTAILLÉE DES FEATURES - {nom_ds}")
        print(f"{'='*80}")
        
        analyse_features = []
        
        for colonne in df.columns:
            dtype = df[colonne].dtype
            
            valeurs_uniques = df[colonne].nunique()
            valeurs_manquantes = df[colonne].isnull().sum()
            
            if colonne.lower() in ['label', 'attack_cat', ' Label']:
                categorie = 'TARGET'
            elif 'ip' in colonne.lower():
                categorie = 'ADRESSE_RÉSEAU'
            elif 'port' in colonne.lower():
                categorie = 'PORT'
            elif 'proto' in colonne.lower() or 'state' in colonne.lower():
                categorie = 'PROTOCOLE'
            elif 'time' in colonne.lower() or 'dur' in colonne.lower():
                categorie = 'TEMPOREL'
            elif 'byte' in colonne.lower():
                categorie = 'VOLUME_DONNÉES'
            elif 'pkt' in colonne.lower() or 'packet' in colonne.lower():
                categorie = 'VOLUME_PAQUETS'
            elif 'length' in colonne.lower() or 'size' in colonne.lower() or 'sz' in colonne.lower():
                categorie = 'TAILLE'
            elif 'load' in colonne.lower() or 'rate' in colonne.lower() or 'speed' in colonne.lower():
                categorie = 'DÉBIT'
            elif 'ttl' in colonne.lower():
                categorie = 'TTL'
            elif 'iat' in colonne.lower() or 'jit' in colonne.lower() or 'rtt' in colonne.lower():
                categorie = 'TIMING'
            elif 'flag' in colonne.lower() or 'fin' in colonne.lower() or 'syn' in colonne.lower() or 'ack' in colonne.lower():
                categorie = 'FLAGS_TCP'
            elif 'service' in colonne.lower():
                categorie = 'SERVICE'
            elif 'ct_' in colonne.lower() or 'is_' in colonne.lower() or 'flow' in colonne.lower():
                categorie = 'COMPORTEMENT_RÉSEAU'
            elif 'fwd' in colonne.lower():
                categorie = 'DIRECTION_AVANT'
            elif 'bwd' in colonne.lower():
                categorie = 'DIRECTION_ARRIÈRE'
            else:
                categorie = 'AUTRE'
            
            if pd.api.types.is_numeric_dtype(df[colonne]):
                min_val = df[colonne].min()
                max_val = df[colonne].max()
                median_val = df[colonne].median()
                mean_val = df[colonne].mean()
                std_val = df[colonne].std()
                stats_resume = f"Min:{min_val:.2f} Max:{max_val:.2f} Méd:{median_val:.2f}"
            else:
                min_val = max_val = median_val = mean_val = std_val = np.nan
                top_valeurs = df[colonne].value_counts().head(3)
                stats_resume = "Top: " + ", ".join([f"{k}({v})" for k, v in top_valeurs.items()])
            
            analyse_features.append({
                'Feature': colonne,
                'Type': dtype,
                'Catégorie': categorie,
                'Valeurs_Uniques': valeurs_uniques,
                'Valeurs_Manquantes': valeurs_manquantes,
                'Statistiques': stats_resume
            })
        
        df_analyse = pd.DataFrame(analyse_features)
        
        print(f"\nRÉSUMÉ GÉNÉRAL - {nom_ds}")
        print(f"Nombre total de features: {len(df.columns)}")
        print(f"Nombre d'échantillons: {len(df)}")
        
        print(f"\nDistribution des types de données:")
        print(df_analyse['Type'].value_counts())
        
        print(f"\nDistribution des catégories:")
        print(df_analyse['Catégorie'].value_counts())
        
        features_manquants = df_analyse[df_analyse['Valeurs_Manquantes'] > 0]
        if len(features_manquants) > 0:
            print(f"\nFeatures avec valeurs manquantes:")
            print(features_manquants[['Feature', 'Valeurs_Manquantes']].to_string(index=False))
        else:
            print(f"\n Aucune valeur manquante détectée")
        
        return df_analyse
    
    def identifier_features_perturbation(df_analyse, nom_ds):
        """
        Identifier les features prometteurs pour les attaques adversariales
        """
        print(f"\n FEATURES PROMETTEURS POUR PERTURBATIONS ADVERSARIALES - {nom_ds}")
        
        features_prometteurs = df_analyse[
            (df_analyse['Type'].isin([np.dtype('int64'), np.dtype('float64'), 
                                    np.dtype('int32'), np.dtype('float32'),
                                    np.dtype('int16'), np.dtype('int8')])) &
            (df_analyse['Catégorie'] != 'TARGET') &
            (~df_analyse['Catégorie'].isin(['ADRESSE_RÉSEAU', 'PORT'])) &
            (df_analyse['Catégorie'] != 'FLAGS_TCP') &
            (df_analyse['Catégorie'] != 'PROTOCOLE')
        ]
        
        features_continus = features_prometteurs[
            features_prometteurs['Catégorie'].isin(['DÉBIT', 'TEMPOREL', 'TIMING', 'TAILLE', 'VOLUME_DONNÉES'])
        ]
        
        features_discrets = features_prometteurs[
            features_prometteurs['Catégorie'].isin(['VOLUME_PAQUETS', 'TTL', 'COMPORTEMENT_RÉSEAU', 'AUTRE'])
        ]
        
        print(f" Features continus (perturbation subtile):")
        for _, feat in features_continus.iterrows():
            print(f"   - {feat['Feature']} ({feat['Catégorie']})")
        
        print(f"\n Features discrets (perturbation modérée):")
        for _, feat in features_discrets.iterrows():
            print(f"   - {feat['Feature']} ({feat['Catégorie']})")
        
        return {
            'continus': features_continus,
            'discrets': features_discrets,
            'tous': features_prometteurs
        }
    
    print(f" DÉBUT DE L'ANALYSE - {nom_dataset}")
    
    df_analyse = analyser_features_detaille(dataframe, nom_dataset)
    
    print(f"\n TABLEAU COMPLET DES FEATURES - {nom_dataset}")
    with pd.option_context('display.max_rows', None, 'display.width', None):
        print(df_analyse.to_string(index=False))
    
    features_perturbation = identifier_features_perturbation(df_analyse, nom_dataset)
    
    if sauvegarder_csv:
        results_dir = Path("..") / "results_analyse_feature"
        results_dir.mkdir(exist_ok=True)
        
        nom_fichier = f"analyse_features_{nom_dataset.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        chemin_complet = results_dir / nom_fichier
        
        df_analyse.to_csv(chemin_complet, index=False)
        print(f"\n Analyse sauvegardée dans '{chemin_complet}'")
    
    return {
        'analyse_dataframe': df_analyse,
        'features_perturbation': features_perturbation,
        'dataset_original': dataframe
    }


def analyser_cicids2017_features(df, nom_dataset="CICIDS2017", sauvegarder_csv=True):
    """
    Analyse approfondie des features CICIDS2017
    """
    print(f"\n{'='*80}")
    print(f" ANALYSE APPROFONDIE DES FEATURES - {nom_dataset}")
    print(f"{'='*80}")
    
    print("\n1. VUE D'ENSEMBLE DU DATASET")
    print("-" * 40)
    
    print(f" Shape du dataset: {df.shape}")
    print(f" Target variable: {df.columns[df.columns.str.contains('label', case=False)][0] if df.columns[df.columns.str.contains('label', case=False)].any() else 'Non trouvée'}")
    
    type_analysis = df.dtypes.value_counts()
    print(f"\n Répartition des types de données:")
    for dtype, count in type_analysis.items():
        print(f"   - {dtype}: {count} features")
    
    missing_values = df.isnull().sum()
    missing_percentage = (missing_values / len(df)) * 100
    high_missing = missing_percentage[missing_percentage > 0]
    
    if len(high_missing) > 0:
        print(f"\n  Valeurs manquantes détectées:")
        for col, pct in high_missing.items():
            print(f"   - {col}: {pct:.2f}%")
    else:
        print(f"\n Aucune valeur manquante détectée")
    
    print("\n2. CATÉGORISATION THÉMATIQUE DES FEATURES")
    print("-" * 50)
    
    feature_categories = {
        'TARGET': [],
        'ADRESSE_RÉSEAU': [],
        'PORTS': [],
        'TIMING': [],
        'VOLUME_PAQUETS': [],
        'VOLUME_DONNÉES': [],
        'DÉBIT': [],
        'FLAGS_TCP': [],
        'STATISTIQUES_FLUX': [],
        'COMPORTEMENT_RÉSEAU': [],
        'AUTRE': []
    }
    
    for col in df.columns:
        col_lower = col.lower()
        
        if 'label' in col_lower:
            feature_categories['TARGET'].append(col)
        elif 'ip' in col_lower:
            feature_categories['ADRESSE_RÉSEAU'].append(col)
        elif 'port' in col_lower:
            feature_categories['PORTS'].append(col)
        elif 'time' in col_lower or 'duration' in col_lower or 'iat' in col_lower:
            feature_categories['TIMING'].append(col)
        elif 'packet' in col_lower or 'pkt' in col_lower:
            feature_categories['VOLUME_PAQUETS'].append(col)
        elif 'byte' in col_lower:
            feature_categories['VOLUME_DONNÉES'].append(col)
        elif 'rate' in col_lower or 'load' in col_lower or 'speed' in col_lower:
            feature_categories['DÉBIT'].append(col)
        elif 'flag' in col_lower or 'fin' in col_lower or 'syn' in col_lower or 'ack' in col_lower:
            feature_categories['FLAGS_TCP'].append(col)
        elif 'flow' in col_lower or 'subflow' in col_lower:
            feature_categories['STATISTIQUES_FLUX'].append(col)
        elif 'protocol' in col_lower or 'service' in col_lower or 'state' in col_lower:
            feature_categories['COMPORTEMENT_RÉSEAU'].append(col)
        else:
            feature_categories['AUTRE'].append(col)
    
    for category, features in feature_categories.items():
        if features:
            print(f"\n🔹 {category} ({len(features)} features):")
            for feat in features[:5]:
                print(f"   - {feat}")
            if len(features) > 5:
                print(f"   ... et {len(features) - 5} autres")
    
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    print(f"\nNombre de features numériques: {len(numeric_cols)}")
    
    print("\n3. ANALYSE STATISTIQUE DES FEATURES NUMÉRIQUES")
    print("-" * 55)
    
    stats_df = df[numeric_cols].describe(percentiles=[.25, .5, .75]).T
    stats_df['variance'] = df[numeric_cols].var()
    stats_df['skewness'] = df[numeric_cols].skew()
    stats_df['kurtosis'] = df[numeric_cols].kurtosis()
    
    print("\n Statistiques des 10 premiers features numériques:")
    print(stats_df.head(10).round(3))
    
    print("\n4. ANALYSE DE CORRÉLATION")
    print("-" * 30)
    
    corr_matrix = df[numeric_cols].corr()
    
    high_corr_pairs = []
    for i in range(len(corr_matrix.columns)):
        for j in range(i+1, len(corr_matrix.columns)):
            corr_val = corr_matrix.iloc[i, j]
            if abs(corr_val) > 0.8:
                high_corr_pairs.append({
                    'Feature_1': corr_matrix.columns[i],
                    'Feature_2': corr_matrix.columns[j],
                    'Corrélation': corr_val
                })
    
    high_corr_df = pd.DataFrame(high_corr_pairs).sort_values('Corrélation', key=abs, ascending=False)
    
    print(f" {len(high_corr_pairs)} paires avec |corrélation| > 0.8")
    if len(high_corr_pairs) > 0:
        print("\n Top 10 des paires les plus corrélées:")
        print(high_corr_df.head(10).to_string(index=False))
    
    print("\n5. FEATURES IDÉAUX POUR PERTURBATIONS ADVERSARIALES")
    print("-" * 60)
    
    features_perturbables = []
    
    for col in numeric_cols:
        col_lower = col.lower()
        
        if any(exclude in col_lower for exclude in ['ip', 'port', 'protocol', 'id']):
            continue
            
        variance = df[col].var()
        unique_ratio = df[col].nunique() / len(df)
        
        if 'rate' in col_lower or 'load' in col_lower or 'speed' in col_lower:
            categorie = 'DÉBIT'
            priorite = 'HAUTE'
        elif 'byte' in col_lower:
            categorie = 'VOLUME_DONNÉES' 
            priorite = 'HAUTE'
        elif 'packet' in col_lower or 'pkt' in col_lower:
            categorie = 'VOLUME_PAQUETS'
            priorite = 'MOYENNE'
        elif 'time' in col_lower or 'duration' in col_lower or 'iat' in col_lower:
            categorie = 'TIMING'
            priorite = 'HAUTE'
        elif 'length' in col_lower or 'size' in col_lower:
            categorie = 'TAILLE'
            priorite = 'MOYENNE'
        else:
            categorie = 'AUTRE'
            priorite = 'FAIBLE'
        
        features_perturbables.append({
            'Feature': col,
            'Catégorie': categorie,
            'Priorité': priorite,
            'Variance': variance,
            'Unique_Ratio': unique_ratio,
            'Min': df[col].min(),
            'Max': df[col].max(),
            'Moyenne': df[col].mean()
        })
    
    perturb_df = pd.DataFrame(features_perturbables)
    
    print("\n FEATURES CONTINUS - Perturbation Subtile (Priorité HAUTE):")
    high_priority = perturb_df[perturb_df['Priorité'] == 'HAUTE'].sort_values('Variance', ascending=False)
    for _, row in high_priority.head(10).iterrows():
        print(f"   - {row['Feature']} ({row['Catégorie']})")
        print(f"     Variance: {row['Variance']:.2f}, Range: [{row['Min']:.2f}, {row['Max']:.2f}]")
    
    print("\n FEATURES DISCRETS - Perturbation Modérée (Priorité MOYENNE):")
    med_priority = perturb_df[perturb_df['Priorité'] == 'MOYENNE'].sort_values('Variance', ascending=False)
    for _, row in med_priority.head(5).iterrows():
        print(f"   - {row['Feature']} ({row['Catégorie']})")
    
    print("\n6. RAPPORT DE SYNTHÈSE")
    print("-" * 25)
    
    print(f" Points forts du dataset:")
    print(f"   - {len(numeric_cols)} features numériques pour l'analyse")
    print(f"   - Richesse des métriques de flux bidirectionnelles")
    print(f"   - Granularité temporelle avancée (IAT, duration)")
    
    print(f"\n Recommandations pour perturbations adversariales:")
    print(f"   - Cibler les métriques de DÉBIT en priorité")
    print(f"   - Utiliser des perturbations subtiles sur les TIMINGS") 
    print(f"   - Explorer l'asymétrie forward/backward")
    
    print(f"\n  Features à éviter pour perturbations:")
    print(f"   - Adresses IP et ports (trop structurés)")
    print(f"   - Protocoles (valeurs discrètes)")
    print(f"   - Identifiants de flux")
    
    if sauvegarder_csv:
        summary_data = []
        for col in df.columns:
            col_lower = col.lower()
            category = 'AUTRE'
            for cat, features in feature_categories.items():
                if col in features:
                    category = cat
                    break
            
            summary_data.append({
                'Feature': col,
                'Type': df[col].dtype,
                'Catégorie': category,
                'Valeurs_Uniques': df[col].nunique(),
                'Valeurs_Manquantes': df[col].isnull().sum(),
                'Valeurs_Manquantes_Pourcentage': (df[col].isnull().sum() / len(df)) * 100
            })
        
        summary_df = pd.DataFrame(summary_data)
        
        results_dir = Path("..") / "results_analyse_feature"
        results_dir.mkdir(exist_ok=True)
        
        filename = f"analyse_detaille_{nom_dataset.lower().replace(' ', '_').replace('(', '').replace(')', '')}.csv"
        chemin_complet = results_dir / filename
        summary_df.to_csv(chemin_complet, index=False)
        print(f"\n Analyse détaillée sauvegardée dans: {chemin_complet}")
    
    return {
        'feature_categories': feature_categories,
        'statistics': stats_df,
        'correlation_analysis': high_corr_df,
        'perturbation_features': perturb_df,
        'numeric_features': numeric_cols
    }