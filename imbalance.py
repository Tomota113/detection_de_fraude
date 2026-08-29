"""
=============================================================================
 Gestion du Déséquilibre des Classes
 Détection de Fraude Énergétique — STEG Tunisie
=============================================================================
 Dans le contexte de la fraude énergétique, les cas positifs (fraude)
 sont extrêmement rares (<10%). Ce module implémente deux stratégies
 complémentaires pour contrer ce déséquilibre :

   1. Pondération interne (scale_pos_weight) — pour XGBoost/LightGBM
   2. Rééchantillonnage SMOTE-Tomek — combinant suréchantillonnage
      synthétique + nettoyage des frontières de décision

 Référence : Chawla et al. (2002), "SMOTE: Synthetic Minority
             Over-sampling Technique"
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Tuple

from imblearn.combine import SMOTETomek
from imblearn.over_sampling import SMOTE

from config import RANDOM_SEED, setup_logging

logger = setup_logging()


class ImbalanceHandler:
    """Gère le déséquilibre des classes pour la détection de fraude.

    Fournit :
    - Le calcul automatique de scale_pos_weight (ratio neg/pos)
    - L'application de SMOTE-Tomek pour le rééchantillonnage
    - Un rapport comparatif avant/après traitement

    Parameters
    ----------
    random_state : int
        Seed de reproductibilité (par défaut RANDOM_SEED).
    """

    def __init__(self, random_state: int = RANDOM_SEED):
        self.random_state = random_state
        self.scale_pos_weight: float = 1.0
        self.original_distribution: dict = {}
        self.resampled_distribution: dict = {}

    # -----------------------------------------------------------------
    # Calcul du poids de déséquilibre
    # -----------------------------------------------------------------

    def compute_scale_pos_weight(self, y: pd.Series) -> float:
        """Calcule le ratio de déséquilibre neg/pos pour scale_pos_weight.

        Ce ratio est utilisé par XGBoost et LightGBM pour pondérer
        la classe minoritaire lors de l'optimisation.

        Parameters
        ----------
        y : pd.Series
            Variable cible binaire (0/1).

        Returns
        -------
        float
            Ratio négatifs / positifs.
        """
        n_neg = (y == 0).sum()
        n_pos = (y == 1).sum()

        if n_pos == 0:
            logger.warning("Aucun cas positif (fraude) dans les données !")
            self.scale_pos_weight = 1.0
        else:
            self.scale_pos_weight = n_neg / n_pos

        self.original_distribution = {
            "n_total": len(y),
            "n_negatifs": int(n_neg),
            "n_positifs": int(n_pos),
            "ratio_positifs": float(n_pos / len(y)),
            "scale_pos_weight": float(self.scale_pos_weight),
        }

        logger.info("=" * 60)
        logger.info("ANALYSE DU DÉSÉQUILIBRE DES CLASSES")
        logger.info("=" * 60)
        logger.info(f"  Classe 0 (légitime) : {n_neg:>8,} ({n_neg/len(y):.2%})")
        logger.info(f"  Classe 1 (fraude)   : {n_pos:>8,} ({n_pos/len(y):.2%})")
        logger.info(f"  scale_pos_weight    : {self.scale_pos_weight:.2f}")

        return self.scale_pos_weight

    # -----------------------------------------------------------------
    # SMOTE-Tomek
    # -----------------------------------------------------------------

    def apply_smote_tomek(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        sampling_strategy: float = 0.3,
    ) -> Tuple[pd.DataFrame, pd.Series]:
        """Applique SMOTE-Tomek pour rééquilibrer les classes.

        Combinaison de :
        - SMOTE : génère des échantillons synthétiques de la classe
          minoritaire par interpolation k-NN.
        - Tomek Links : supprime les paires ambiguës à la frontière
          de décision pour nettoyer les données.

        Parameters
        ----------
        X : pd.DataFrame
            Matrice de features.
        y : pd.Series
            Variable cible.
        sampling_strategy : float
            Ratio cible pos/neg après rééchantillonnage (0.3 = 30%
            de positifs par rapport aux négatifs). Default 0.3 pour
            éviter un rééquilibrage total qui dégraderait le signal.

        Returns
        -------
        X_resampled : pd.DataFrame
            Features rééchantillonnées.
        y_resampled : pd.Series
            Cible rééchantillonnée.
        """
        logger.info("-" * 60)
        logger.info("APPLICATION DE SMOTE-TOMEK")
        logger.info(f"  Stratégie : ratio cible pos/neg = {sampling_strategy}")
        logger.info(f"  Avant : {dict(pd.Series(y).value_counts())}")

        smote_tomek = SMOTETomek(
            smote=SMOTE(
                sampling_strategy=sampling_strategy,
                k_neighbors=5,
                random_state=self.random_state,
            ),
            random_state=self.random_state,
            n_jobs=-1,
        )

        X_resampled, y_resampled = smote_tomek.fit_resample(X, y)

        # Reconvertir en DataFrame/Series pour cohérence
        X_resampled = pd.DataFrame(X_resampled, columns=X.columns)
        y_resampled = pd.Series(y_resampled, name=y.name)

        self.resampled_distribution = {
            "n_total": len(y_resampled),
            "n_negatifs": int((y_resampled == 0).sum()),
            "n_positifs": int((y_resampled == 1).sum()),
            "ratio_positifs": float((y_resampled == 1).mean()),
        }

        logger.info(f"  Après  : {dict(y_resampled.value_counts())}")
        logger.info(f"  Échantillons ajoutés : {len(y_resampled) - len(y):,}")

        return X_resampled, y_resampled

    # -----------------------------------------------------------------
    # Rapport
    # -----------------------------------------------------------------

    def get_report(self) -> dict:
        """Retourne un rapport comparatif avant/après traitement.

        Returns
        -------
        dict
            Dictionnaire contenant les distributions et le scale_pos_weight.
        """
        return {
            "original": self.original_distribution,
            "resampled": self.resampled_distribution,
        }
