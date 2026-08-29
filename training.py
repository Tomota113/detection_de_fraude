"""
=============================================================================
 Entraînement & Optimisation Multi-Modèles
 Détection de Fraude Énergétique — STEG Tunisie
=============================================================================
 Ce module orchestre l'entraînement de 3 modèles de pointe pour données
 tabulaires, avec validation croisée stratifiée pour garantir la robustesse.

 Modèles implémentés :
   1. XGBoost  — Gradient boosting avec régularisation L1/L2
   2. LightGBM — Gradient boosting par histogramme (rapide et léger)
   3. CatBoost — Gradient boosting avec gestion native des catégories

 Stratégie d'évaluation :
   - Stratified K-Fold (5 folds) pour préserver la distribution des classes
   - Métriques : ROC-AUC (discrimination), PR-AUC (performance sur fraude)
   - Sélection du meilleur modèle sur PR-AUC (métrique la plus pertinente
     pour les données déséquilibrées)
=============================================================================
"""

import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
)

from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from config import (
    RANDOM_SEED,
    N_CV_FOLDS,
    XGBOOST_PARAMS,
    LIGHTGBM_PARAMS,
    CATBOOST_PARAMS,
    setup_logging,
)

logger = setup_logging()


class FraudDetectionTrainer:
    """Entraîne et compare plusieurs modèles pour la détection de fraude.

    Implémente un workflow complet :
    1. Initialisation des 3 modèles avec hyperparamètres optimisés
    2. Validation croisée stratifiée avec métriques par fold
    3. Sélection automatique du meilleur modèle
    4. Entraînement final sur l'ensemble des données d'entraînement

    Parameters
    ----------
    scale_pos_weight : float
        Poids de la classe positive (calculé par ImbalanceHandler).
    random_state : int
        Seed de reproductibilité.
    n_folds : int
        Nombre de folds pour la validation croisée.
    """

    def __init__(
        self,
        scale_pos_weight: float = 1.0,
        random_state: int = RANDOM_SEED,
        n_folds: int = N_CV_FOLDS,
    ):
        self.scale_pos_weight = scale_pos_weight
        self.random_state = random_state
        self.n_folds = n_folds

        # Résultats de la validation croisée
        self.cv_results: Dict[str, dict] = {}
        self.best_model_name: str = ""
        self.best_model = None
        self.trained_models: Dict[str, object] = {}

    # -----------------------------------------------------------------
    # Initialisation des modèles
    # -----------------------------------------------------------------

    def _init_models(self) -> Dict[str, object]:
        """Initialise les 3 modèles avec les hyperparamètres configurés.

        Le scale_pos_weight est injecté dans XGBoost et LightGBM.
        CatBoost utilise auto_class_weights='Balanced'.

        Returns
        -------
        dict
            Dictionnaire {nom: modèle} des 3 classificateurs.
        """
        # XGBoost
        xgb_params = XGBOOST_PARAMS.copy()
        xgb_params["scale_pos_weight"] = self.scale_pos_weight

        # LightGBM
        lgbm_params = LIGHTGBM_PARAMS.copy()
        lgbm_params["scale_pos_weight"] = self.scale_pos_weight

        # CatBoost
        catboost_params = CATBOOST_PARAMS.copy()
        catboost_params["auto_class_weights"] = "Balanced"

        models = {
            "XGBoost": XGBClassifier(**xgb_params),
            "LightGBM": LGBMClassifier(**lgbm_params),
            "CatBoost": CatBoostClassifier(**catboost_params),
        }

        return models

    # -----------------------------------------------------------------
    # Validation croisée stratifiée
    # -----------------------------------------------------------------

    def cross_validate(
        self,
        X: pd.DataFrame,
        y: pd.Series,
    ) -> Dict[str, dict]:
        """Exécute la validation croisée stratifiée sur les 3 modèles.

        Pour chaque fold, entraîne le modèle et calcule :
        - ROC-AUC (capacité de discrimination globale)
        - PR-AUC (performance sur la classe fraude — métrique clé)
        - F1-Score (équilibre precision/recall)

        Parameters
        ----------
        X : pd.DataFrame
            Matrice de features.
        y : pd.Series
            Variable cible binaire.

        Returns
        -------
        dict
            Résultats de CV par modèle : moyennes et écarts-types.
        """
        logger.info("=" * 70)
        logger.info("VALIDATION CROISÉE STRATIFIÉE")
        logger.info(f"  Folds    : {self.n_folds}")
        logger.info(f"  Métriques : ROC-AUC, PR-AUC, F1-Score")
        logger.info("=" * 70)

        skf = StratifiedKFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        models = self._init_models()

        for model_name, model in models.items():
            logger.info(f"\n{'─' * 50}")
            logger.info(f"  Modèle : {model_name}")
            logger.info(f"{'─' * 50}")

            roc_scores = []
            pr_scores = []
            f1_scores = []

            for fold_idx, (train_idx, val_idx) in enumerate(skf.split(X, y), 1):
                X_train_fold = X.iloc[train_idx]
                y_train_fold = y.iloc[train_idx]
                X_val_fold = X.iloc[val_idx]
                y_val_fold = y.iloc[val_idx]

                # Entraîner
                model.fit(X_train_fold, y_train_fold)

                # Prédire des probabilités
                y_proba = model.predict_proba(X_val_fold)[:, 1]
                y_pred = (y_proba >= 0.5).astype(int)

                # Calculer les métriques
                roc = roc_auc_score(y_val_fold, y_proba)
                pr = average_precision_score(y_val_fold, y_proba)
                f1 = f1_score(y_val_fold, y_pred, zero_division=0)

                roc_scores.append(roc)
                pr_scores.append(pr)
                f1_scores.append(f1)

                logger.info(f"    Fold {fold_idx}/{self.n_folds} — "
                             f"ROC-AUC: {roc:.4f} | PR-AUC: {pr:.4f} | F1: {f1:.4f}")

            # Résultats agrégés
            self.cv_results[model_name] = {
                "roc_auc_mean": np.mean(roc_scores),
                "roc_auc_std": np.std(roc_scores),
                "pr_auc_mean": np.mean(pr_scores),
                "pr_auc_std": np.std(pr_scores),
                "f1_mean": np.mean(f1_scores),
                "f1_std": np.std(f1_scores),
            }

            logger.info(f"  ► Moyenne — "
                         f"ROC-AUC: {np.mean(roc_scores):.4f} ± {np.std(roc_scores):.4f} | "
                         f"PR-AUC: {np.mean(pr_scores):.4f} ± {np.std(pr_scores):.4f} | "
                         f"F1: {np.mean(f1_scores):.4f} ± {np.std(f1_scores):.4f}")

        return self.cv_results

    # -----------------------------------------------------------------
    # Sélection et entraînement du meilleur modèle
    # -----------------------------------------------------------------

    def train_best_model(
        self,
        X_train: pd.DataFrame,
        y_train: pd.Series,
    ) -> Tuple[str, object]:
        """Identifie le meilleur modèle (PR-AUC) et l'entraîne.

        La sélection se fait sur la PR-AUC moyenne de la validation
        croisée, car c'est la métrique la plus pertinente pour la
        détection de fraude (données très déséquilibrées).

        Parameters
        ----------
        X_train : pd.DataFrame
            Données d'entraînement.
        y_train : pd.Series
            Variable cible d'entraînement.

        Returns
        -------
        tuple
            (nom_du_meilleur_modèle, modèle_entraîné)
        """
        if not self.cv_results:
            raise ValueError("Exécutez cross_validate() avant train_best_model()")

        # Sélection sur PR-AUC (plus pertinent que ROC-AUC pour la fraude)
        self.best_model_name = max(
            self.cv_results,
            key=lambda name: self.cv_results[name]["pr_auc_mean"],
        )

        logger.info(f"\n{'=' * 70}")
        logger.info(f"MEILLEUR MODÈLE : {self.best_model_name}")
        logger.info(f"  PR-AUC moyen : "
                     f"{self.cv_results[self.best_model_name]['pr_auc_mean']:.4f}")
        logger.info(f"{'=' * 70}")

        # Réinitialiser et entraîner le meilleur modèle sur tout le train
        models = self._init_models()
        self.best_model = models[self.best_model_name]
        self.best_model.fit(X_train, y_train)

        # Stocker tous les modèles entraînés pour comparaison
        for name, model in models.items():
            if name != self.best_model_name:
                model.fit(X_train, y_train)
            self.trained_models[name] = model

        logger.info("  Modèle final entraîné sur l'ensemble du jeu d'entraînement.")

        return self.best_model_name, self.best_model

    # -----------------------------------------------------------------
    # Résumé des résultats
    # -----------------------------------------------------------------

    def get_results_summary(self) -> pd.DataFrame:
        """Génère un tableau récapitulatif des résultats de CV.

        Returns
        -------
        pd.DataFrame
            Tableau comparatif des 3 modèles avec toutes les métriques.
        """
        if not self.cv_results:
            return pd.DataFrame()

        rows = []
        for name, metrics in self.cv_results.items():
            rows.append({
                "Modèle": name,
                "ROC-AUC": f"{metrics['roc_auc_mean']:.4f} ± {metrics['roc_auc_std']:.4f}",
                "PR-AUC": f"{metrics['pr_auc_mean']:.4f} ± {metrics['pr_auc_std']:.4f}",
                "F1-Score": f"{metrics['f1_mean']:.4f} ± {metrics['f1_std']:.4f}",
                "★": "★" if name == self.best_model_name else "",
            })

        return pd.DataFrame(rows)
