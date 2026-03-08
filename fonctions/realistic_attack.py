import numpy as np
import pandas as pd

def is_realistic(stats, 
                 max_neg=0, 
                 max_nan=0, 
                 max_above_max=0, 
                 l2_threshold=1.0, 
                 linf_threshold=10.0):
    """
    Détermine si une attaque adversariale est réaliste
    à partir des métriques retournées par check_plausibility().

    Paramètres
    ----------
    stats : dict
        Dictionnaire retourné par check_plausibility().
    max_neg : int
        Nombre maximal de valeurs négatives autorisées.
    max_nan : int
        Nombre maximal de NaN autorisés.
    max_above_max : int
        Nombre maximal de valeurs dépassant max_dataset.
    l2_threshold : float
        Limite supérieure pour la distance L2 moyenne.
    linf_threshold : float
        Limite supérieure pour la distance L∞ maximale.

    Retour
    ------
    bool : True si l’attaque est réaliste, False sinon.
    """

    # Récupération des valeurs
    nb_neg = stats["nb_negative"]
    nb_nan = stats["nb_nan"]
    nb_above = stats["nb_above_max"]
    l2 = stats["l2_distance_mean"]
    linf = stats["l_inf_max"]

    # Conditions de plausibilité
    if nb_neg > max_neg:
        return False
    if nb_nan > max_nan:
        return False
    if nb_above > max_above_max:
        return False
    if l2 > l2_threshold:
        return False
    if linf > linf_threshold:
        return False

    # Si toutes les conditions sont respectées
    return True


def hard_clip_to_range(X_adv, X_clean):
    """
    Clipping robuste :
    - X_adv -> DataFrame aligné à X_clean
    - min = 0
    - max = max par feature dans X_clean
    """
    if not isinstance(X_clean, pd.DataFrame):
        X_clean = pd.DataFrame(X_clean)

    cols = X_clean.columns

    X_adv = pd.DataFrame(np.asarray(X_adv, dtype=float), columns=cols)

    X_min = 0.0
    X_max = X_clean.max()

    X_adv_clipped = X_adv.clip(lower=X_min, upper=X_max, axis=1)
    return X_adv_clipped

def hard_clip_unsw(X_adv, X_clean):
    """
    Clipping réaliste UNSW :
    - On identifie les colonnes qui dépassent le max (vraies features continues)
    - On clippe X_adv uniquement sur ces colonnes
    """

    # Convertir proprement
    if not isinstance(X_clean, pd.DataFrame):
        X_clean = pd.DataFrame(X_clean)

    X_adv = pd.DataFrame(X_adv, columns=X_clean.columns).astype(float)

    # 1) Colonnes qui dépassent le max_clean
    max_clean = X_clean.max()
    above_cols = [col for col in X_clean.columns 
                  if X_adv[col].max() > max_clean[col]]

    # 2) Clip sur min_clean / max_clean pour CES colonnes-là seulement
    min_clean = X_clean.min()

    for col in above_cols:
        X_adv[col] = X_adv[col].clip(lower=min_clean[col], upper=max_clean[col])

    return X_adv