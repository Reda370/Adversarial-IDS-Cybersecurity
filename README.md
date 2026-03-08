# 🛡️ Adversarial-IDS: Évaluation et Défense de NIDS face aux Attaques Adverses

Ce projet traite d'un enjeu majeur à l'intersection de l'Intelligence Artificielle et de la Cybersécurité : la robustesse des Systèmes de Détection d'Intrusion (NIDS) face aux manipulations intentionnelles de données (**Adversarial Machine Learning**).

## 🚀 Problématique de Cybersécurité

Les modèles de Machine Learning (comme le Random Forest ou le MLP) sont performants pour détecter des attaques connues (DoS, Scan) sur des datasets comme **UNSW-NB15** ou **CIC-IDS2017**. Cependant, ces modèles sont fragiles. Un attaquant peut subtilement modifier ses paquets réseau pour qu'ils soient classés comme "Bénins" par l'IA tout en restant malveillants.

Ce projet démontre :
1. La vulnérabilité des NIDS standard.
2. Une méthode d'attaque par perturbation de caractéristiques (Feature Perturbation).
3. Un mécanisme de défense efficace par **Entraînement Adverse**.

---

## 🛠️ Stack Technique & Compétences Valorisées

* **Langages :** Python
* **Data Science / ML :** Pandas, Scikit-learn (Random Forest, Isolation Forest), SMOTE (pour l'équilibrage des classes).
* **Cybersécurité :** Analyse de trafic réseau, Détection d'intrusion, Threat Modeling (Adversarial ML).

---

## 📊 Pipeline et Résultats

Le pipeline complet est documenté dans le notebook `notebooks/main.ipynb`.

### 1. Modèle Baseline et Importance des Features

Nous entraînons un modèle Random Forest sur le dataset nettoyé et équilibré via SMOTE. Nous identifions les caractéristiques réseau les plus critiques pour la détection (ex: `dur`, `sbytes`, `dbytes`).

*(Image de l'importance des caractéristiques ici)*

### 2. L'Attaque : Feature Perturbation

Nous simulons un attaquant qui connaît les features importantes. Le pirate modifie ses valeurs (ex: augmente légèrement la durée de la connexion) pour tromper le modèle.

> **Résultat :** Sous attaque, le Recall (capacité à détecter les vraies attaques) chute drastiquement, créant de nombreux faux négatifs.

### 3. La Défense : Adversarial Training

Nous mettons en place un mécanisme de défense en réentraînant le modèle sur un jeu de données "mixte" (données saines + exemples adverses).

*(Placeholder pour matrice de confusion avant/après défense)*

> **Résultat de la défense :** Le modèle défendu retrouve une précision proche du modèle original tout en devenant robuste contre l'attaque spécifique, au prix d'une légère augmentation de la latence de détection.

---

## 📁 Structure du Projet

```text
.
├── fonctions/          # Scripts Python modulaires
│   ├── clean_data.py   # Prétraitement et nettoyage des datasets
│   ├── Smote.py        # Équilibrage des classes (SMOTE)
│   ├── feature_attack.py # Logique de l'attaque adverse
│   └── adversarial_training.py # Scripts de défense
├── notebooks/          # Démonstration et Analyse
│   └── main.ipynb      # Pipeline complet détaillé
├── results/            # Dossier pour les graphiques (Matrices, Heatmaps)
└── requirements.txt    # Dépendances du projet
```
