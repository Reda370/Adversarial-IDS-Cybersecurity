# 🛡️ Adversarial-IDS: Évaluation et Défense de NIDS face aux Attaques Adverses

Ce projet traite d'un enjeu majeur à l'intersection de l'Intelligence Artificielle et de la Cybersécurité : la robustesse des Systèmes de Détection d'Intrusion (NIDS) face aux manipulations intentionnelles de données (**Adversarial Machine Learning**).

---

## 🚀 Objectif du projet

L'objectif est de reproduire une chaîne complète de recherche en adversarial ML appliqué à la détection d'intrusion réseau :

1. Charger et prétraiter des datasets NIDS (CICIDS2017 + UNSW-NB15).
2. Entraîner des modèles de référence (Random Forest, MLP).
3. Générer des attaques adversariales réalistes (Feature Perturbation, FGSM).
4. Évaluer l'impact de ces attaques (taux d'évasion, chute du recall, etc.).
5. Proposer des défenses (adversarial training, re-training sur données adversariales, vérifications de plausibilité).

---

## 📦 Prérequis & Installation

### 1) Créer un environnement Python

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

### 2) Installer les dépendances

```powershell
python -m pip install -r requirements.txt
python -m pip install torch torchvision torchaudio  # (optionnel mais recommandé pour MLP et FGSM)
```

> ⚠️ **Remarque** : Le projet utilise à la fois `scikit-learn` (Random Forest) et `PyTorch` (MLP + FGSM). Assure-toi d’avoir la bonne version de `torch` selon ton GPU/CPU.

---

## 📂 Données (Datasets)

Les deux datasets doivent être placés au niveau racine du projet (à côté de `notebooks/`):

- `CICIDS2017/` (csv) → [téléchargement officiel](https://www.unb.ca/cic/datasets/ids-2017.html)
- `UNSW8B15/` (parquet) → [téléchargement officiel](https://research.unsw.edu.au/projects/unsw-nb15-dataset)

> ✅ Le notebook `notebooks/main.ipynb` utilise ces chemins :
> - `../CICIDS2017`
> - `../UNSW8B15`

---

## 🧠 Architecture & Modules Clés

### 1) Prétraitement & Split

- `fonctions/load_data.py` : chargement de CICIDS et UNSW.
- `fonctions/clean_data.py` : suppression des NaN/inf, outliers, et détection automatique de dataset.
- `fonctions/split_data.py` : split stratifié train/val/test (70/15/15) et export CSV.

### 2) Modèles de référence

- `fonctions/model_rf.py` : entraînement et sauvegarde d’un Random Forest (baseline).
- `fonctions/MLP_baseline.py` : MLP PyTorch simple + utilitaires (pré-traitement, normalisation, sauvegarde, chargement).

### 3) Attaques adversariales

- `fonctions/feature_attack.py` : attaque **Feature Perturbation** (universal / réaliste) pour Random Forest.
- `fonctions/fgsm_realiste.py` / `fonctions/fgsm_non_realiste.py` / `fonctions/fgsm_ablation.py` : variantes FGSM (Fast Gradient Sign Method) sur MLP.
- `fonctions/gradient_free_attack.py` : attaque sans gradients.

### 4) Défense & Entraînement adverse

- `fonctions/adversarial_training.py` : entraînement adverse (FGSM + mix clean/adversarial) pour MLP.
- `fonctions/defense_feature_perturbation_rf.py` : génération d’attaques FP, vérification de plausibilité, et ré-entraînement RF.

### 5) Évaluation & Visualisation

- `fonctions/evaluate.py` : métriques et matrices de confusion (sklearn + PyTorch).
- `fonctions/evaluate_defense_fp.py` : pipeline d’évaluation de l’attaque + défense (taux d’évasion, delta de performances).
- `fonctions/plausibility.py` : vérification de la plausibilité des exemples adversariaux (distance L2/L∞, valeurs négatives / au-dessus du max).
- `fonctions/visualisation_attaque.py` + `fonctions/visualisation_attaque_plot.py` : graphiques d’analyse (perturbations, comparaisons, matrices, etc.).

---

## ▶️ Utilisation (Quick Start)

### 1) Ouvrir et exécuter le notebook principal

Le point d’entrée recommandé est le notebook suivant :

- `notebooks/main.ipynb` : pipeline complet (chargement, nettoyage, split, entraînement, attaque, défense, évaluation).

Exécute chaque cellule dans l’ordre (ou redémarre le kernel si nécessaire).

### 2) Exemples de commandes (optionnel)

Si tu veux exécuter des scripts en dehors du notebook, tu peux appeler directement les fonctions depuis Python :

```python
from fonctions.load_data import load_cicids, load_unsw
from fonctions.clean_data import clean_dataset
from fonctions.split_data import split_dataset
from fonctions.model_rf import train_rf, save_model
from fonctions.feature_attack import feature_perturbation_rf_universal
from fonctions.defense_feature_perturbation_rf import defense_feature_perturbation_rf

# Exemple de workflow rapide (simplifié)
df = load_cicids('CICIDS2017')
df_clean = clean_dataset(df)
X = df_clean.drop(columns=['Label'])
y = df_clean['Label']
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(X, y, name='cicids')

rf = train_rf(X_train, y_train)
save_model(rf, 'results/rf_baseline_cicids.pkl')
```

---

## ✅ Points clés du pipeline

### Attaque (Feature Perturbation)
- L’attaque modifie un petit nombre de features importantes (ex: durée, octets envoyés).
- Les modifications sont **limitée(s)** par des bornes (min/max observées) et respectent un masque de features immuables.
- Un module **plausibility** vérifie que les exemples adversaires restent réalistes (pas de valeurs négatives, pas de dépassement du maximum appris, etc.).

### Défense (Adversarial Training)
- Réentraînement du modèle sur un mélange de données propres + adversariales.
- Pour RF, on génère les adversaires sur le training set puis on réentraîne.
- Pour MLP, un module dédié ajoute du bruit FGSM en ligne pendant l'entraînement.

---

## 📂 Structure détaillée du projet

```
.
├── fonctions/                          # Modules Python réutilisables
│   ├── adversarial_training.py         # Entraînement adverse MLP (FGSM)
│   ├── clean_data.py                   # Nettoyage / outliers / NaN
│   ├── defense_feature_perturbation_rf.py # Défense RF par ré-entraînement
│   ├── evaluate.py                     # Metrics + matrices pour sklearn / pytorch
│   ├── evaluate_defense_fp.py          # Comparaison (baseline vs défense)
│   ├── feature_attack.py               # Attaque Feature Perturbation (RF)
│   ├── fgsm_*                          # Variantes d’attaques FGSM
│   ├── gradient_free_attack.py         # Attaque sans gradients
│   ├── load_data.py                    # Chargement CICIDS + UNSW
│   ├── MLP_baseline.py                 # Modèle MLP + utils de préparation
│   ├── model_rf.py                     # Entraînement / sauvegarde RF
│   ├── plausibility.py                 # Vérification des exemples adverses
│   ├── preprocess_data.py              # (autres utilitaires de prétraitement)
│   ├── split_data.py                   # Split train/val/test
│   ├── Smote.py                        # Oversampling / SMOTE
│   ├── visualisation_attaque*.py       # Graphiques d’analyse
│   └── utils_adversarial.py            # Fonctions utilitaires
├── notebooks/                          # Notebooks d’analyse et démonstration
│   ├── main.ipynb                      # Pipeline complet
│   └── main_demo.ipynb                 # Démo / expé avancées
├── results/                            # Sorties générées (figures, CSV, modèles)
├── requirements.txt                    # Dépendances Python
└── README.md                           # Documentation du projet
```

---

## 🧩 Astuces & bonnes pratiques

- **Travailler sur un subset** : les datasets sont volumineux. Pour itérer rapidement, extrais un sous-ensemble (ex: 10‑20 %) et vérifie que tout fonctionne.
- **Reproductibilité** : les pipelines utilisent des seeds (`seed=42`) pour garder les résultats stables.
- **Visualisation** : les notebooks sont pré‑configurés pour générer des graphiques dans `results/` (matrices, perturbations, etc.).

---

## 🚩 Prochaines évolutions possibles

- Ajouter une CLI pour exécuter les étapes (prétraitement, entraînement, attaque, défense, évaluation) sans notebook.
- Implémenter d’autres attaques (PGD, CW, AutoAttack) et défenses (certified robustness, feature squeezing).
- Comparer plusieurs architectures (XGBoost, LightGBM, CNNs, Transformers).

---

© Projet Adversarial-IDS
