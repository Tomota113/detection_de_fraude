"""
=============================================================================
 Orchestrateur Principal — Pipeline de Détection de Fraude STEG
=============================================================================
 Point d'entrée unique du pipeline complet :

   1. Chargement & nettoyage des données
   2. Feature Engineering avancé (30+ features)
   3. Analyse du déséquilibre des classes
   4. Split stratifié train/test
   5. Validation croisée (XGBoost, LightGBM, CatBoost)
   6. Entraînement du meilleur modèle
   7. Évaluation complète & explicabilité
   8. Sauvegarde des résultats

 Exécution :
   $ python main.py

 Sortie :
   - outputs/plots/   → Graphiques (confusion, ROC, PR, features)
   - outputs/models/   → Modèle entraîné (joblib)
   - outputs/pipeline.log → Journal d'exécution complet
=============================================================================
"""

import sys
import time
import warnings
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split

warnings.filterwarnings("ignore")

# Modules du pipeline
from config import (
    RANDOM_SEED,
    TEST_SIZE,
    OUTPUT_DIR,
    MODELS_DIR,
    setup_logging,
)
from preprocessing import DataLoader, FeatureEngineer
from imbalance import ImbalanceHandler
from training import FraudDetectionTrainer
from evaluation import ModelEvaluator

logger = setup_logging()


def main():
    """Pipeline complet de détection de fraude STEG.

    Orchestre l'ensemble des étapes du pipeline, de l'ingestion
    des données brutes à la génération du rapport d'évaluation.
    """
    start_time = time.time()

    logger.info("╔" + "═" * 68 + "╗")
    logger.info("║  PIPELINE DE DÉTECTION DE FRAUDE ÉNERGÉTIQUE — STEG TUNISIE     ║")
    logger.info("║  Preuve de Concept (PoC) — Projet Académique                    ║")
    logger.info("╚" + "═" * 68 + "╝")

    # ==================================================================
    # ÉTAPE 1 : CHARGEMENT DES DONNÉES
    # ==================================================================
    logger.info("\n" + "▶" * 3 + " ÉTAPE 1/7 : Chargement des données")

    loader = DataLoader()
    client_train, client_test, invoice_train, invoice_test = loader.load_all()

    # ==================================================================
    # ÉTAPE 2 : FEATURE ENGINEERING
    # ==================================================================
    logger.info("\n" + "▶" * 3 + " ÉTAPE 2/7 : Feature Engineering")

    engineer = FeatureEngineer()
    df_features = engineer.build_features(client_train, invoice_train)

    # Séparer X et y
    y = df_features["target"].astype(int)
    X = df_features.drop(columns=["target", "client_id"], errors="ignore")

    # Convertir en numérique
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    feature_names = X.columns.tolist()
    logger.info(f"  Matrice finale : {X.shape[0]} clients × {X.shape[1]} features")

    # ==================================================================
    # ÉTAPE 3 : ANALYSE DU DÉSÉQUILIBRE
    # ==================================================================
    logger.info("\n" + "▶" * 3 + " ÉTAPE 3/7 : Analyse du déséquilibre")

    imbalance_handler = ImbalanceHandler()
    scale_pos_weight = imbalance_handler.compute_scale_pos_weight(y)

    # ==================================================================
    # ÉTAPE 4 : SPLIT STRATIFIÉ TRAIN / TEST
    # ==================================================================
    logger.info("\n" + "▶" * 3 + " ÉTAPE 4/7 : Split stratifié train/test")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=TEST_SIZE,
        random_state=RANDOM_SEED,
        stratify=y,
    )

    logger.info(f"  Train : {X_train.shape[0]:,} ({(1-TEST_SIZE)*100:.0f}%)")
    logger.info(f"  Test  : {X_test.shape[0]:,} ({TEST_SIZE*100:.0f}%)")
    logger.info(f"  Distribution train : {dict(y_train.value_counts())}")
    logger.info(f"  Distribution test  : {dict(y_test.value_counts())}")

    # ==================================================================
    # ÉTAPE 5 : VALIDATION CROISÉE
    # ==================================================================
    logger.info("\n" + "▶" * 3 + " ÉTAPE 5/7 : Validation croisée stratifiée")

    trainer = FraudDetectionTrainer(
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_SEED,
    )

    cv_results = trainer.cross_validate(X_train, y_train)

    # Afficher le tableau récapitulatif
    summary = trainer.get_results_summary()
    logger.info(f"\n  Résumé de la validation croisée :")
    logger.info(f"\n{summary.to_string(index=False)}")

    # ==================================================================
    # ÉTAPE 6 : ENTRAÎNEMENT DU MEILLEUR MODÈLE
    # ==================================================================
    logger.info("\n" + "▶" * 3 + " ÉTAPE 6/7 : Entraînement du meilleur modèle")

    best_name, best_model = trainer.train_best_model(X_train, y_train)

    # ==================================================================
    # ÉTAPE 7 : ÉVALUATION & EXPLICABILITÉ
    # ==================================================================
    logger.info("\n" + "▶" * 3 + " ÉTAPE 7/7 : Évaluation & Explicabilité")

    # Prédictions sur le jeu de test
    y_proba = best_model.predict_proba(X_test)[:, 1]

    evaluator = ModelEvaluator(
        model=best_model,
        model_name=best_name,
        feature_names=feature_names,
    )

    metrics = evaluator.generate_full_report(y_test, y_proba)

    # ==================================================================
    # SAUVEGARDE DU MODÈLE
    # ==================================================================
    try:
        import joblib
        model_path = MODELS_DIR / f"best_model_{best_name.lower()}.joblib"
        joblib.dump(best_model, model_path)
        logger.info(f"\n  💾 Modèle sauvegardé → {model_path}")
    except ImportError:
        logger.warning("  joblib non installé, modèle non sauvegardé.")

    # ==================================================================
    # RÉSUMÉ FINAL
    # ==================================================================
    elapsed = time.time() - start_time

    logger.info("\n" + "═" * 70)
    logger.info("RÉSUMÉ FINAL DU PIPELINE")
    logger.info("═" * 70)
    logger.info(f"  Meilleur modèle   : {best_name}")
    logger.info(f"  ROC-AUC (test)    : {metrics['roc_auc']:.4f}")
    logger.info(f"  PR-AUC  (test)    : {metrics['pr_auc']:.4f}")
    logger.info(f"  F1-Score (test)   : {metrics['f1_score']:.4f}")
    logger.info(f"  Accuracy (test)   : {metrics['accuracy']:.4f}")
    logger.info(f"  Temps total       : {elapsed:.1f}s ({elapsed/60:.1f} min)")
    logger.info(f"  Sorties           : {OUTPUT_DIR}")
    logger.info("═" * 70)
    logger.info("Pipeline terminé avec succès ✓")

    return metrics


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        logger.error(f"Erreur fatale dans le pipeline : {e}", exc_info=True)
        sys.exit(1)
