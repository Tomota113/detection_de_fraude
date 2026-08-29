"""
=============================================================================
 Configuration Centrale — Détection de Fraude Énergétique STEG
=============================================================================
 Centralise l'ensemble des paramètres du pipeline :
   - Chemins d'accès aux données et aux sorties
   - Seed de reproductibilité
   - Hyperparamètres des modèles (XGBoost, LightGBM, CatBoost)
   - Configuration du logging
=============================================================================
"""

import os
import logging
from pathlib import Path

# =============================================================================
# 1. CHEMINS DU PROJET
# =============================================================================

# Racine du projet — détectée automatiquement à partir de ce fichier
PROJECT_ROOT = Path(__file__).resolve().parent

# Données brutes — situées dans Documents/data/python/data
DATA_DIR = PROJECT_ROOT.parents[3] / "data" / "python" / "data"
if not DATA_DIR.exists():
    # Fallback vers le chemin absolu
    DATA_DIR = Path("C:/Users/itomo/OneDrive/Documents/data/python/data")

CLIENT_TRAIN_PATH = DATA_DIR / "client_train.csv"
CLIENT_TEST_PATH = DATA_DIR / "client_test.csv"
INVOICE_TRAIN_PATH = DATA_DIR / "invoice_train.csv"
INVOICE_TEST_PATH = DATA_DIR / "invoice_test.csv"

# Répertoire de sortie (graphiques, rapports, modèles)
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODELS_DIR = OUTPUT_DIR / "models"
PLOTS_DIR = OUTPUT_DIR / "plots"

# Création automatique des répertoires de sortie
for _dir in [OUTPUT_DIR, MODELS_DIR, PLOTS_DIR]:
    _dir.mkdir(parents=True, exist_ok=True)


# =============================================================================
# 2. PARAMÈTRES GLOBAUX
# =============================================================================

RANDOM_SEED = 42
TEST_SIZE = 0.20          # Proportion du jeu de test pour le split stratifié
N_CV_FOLDS = 5            # Nombre de folds pour la validation croisée stratifiée
N_TOP_FEATURES = 20       # Nombre de features à afficher dans les graphiques


# =============================================================================
# 3. HYPERPARAMÈTRES DES MODÈLES
# =============================================================================
# Nota : scale_pos_weight est calculé dynamiquement dans imbalance.py
# en fonction du ratio de déséquilibre réel des classes.

XGBOOST_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 5,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "eval_metric": "auc",
    "use_label_encoder": False,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbosity": 0,
}

LIGHTGBM_PARAMS = {
    "n_estimators": 500,
    "max_depth": 6,
    "learning_rate": 0.05,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_samples": 20,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "num_leaves": 31,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
    "verbose": -1,
}

CATBOOST_PARAMS = {
    "iterations": 500,
    "depth": 6,
    "learning_rate": 0.05,
    "l2_leaf_reg": 3.0,
    "border_count": 128,
    "random_seed": RANDOM_SEED,
    "verbose": 0,
    "thread_count": -1,
}


# =============================================================================
# 4. CONFIGURATION DU LOGGING
# =============================================================================

LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)-25s | %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
LOG_FILE = OUTPUT_DIR / "pipeline.log"


def setup_logging(level: int = logging.INFO) -> logging.Logger:
    """Configure le logging global du pipeline.

    Parameters
    ----------
    level : int
        Niveau de logging (par défaut INFO).

    Returns
    -------
    logging.Logger
        Logger racine configuré.
    """
    logger = logging.getLogger("FraudDetection")
    logger.setLevel(level)

    # Éviter la duplication de handlers si appelé plusieurs fois
    if logger.handlers:
        return logger

    # Handler console
    console_handler = logging.StreamHandler()
    console_handler.setLevel(level)
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(console_handler)

    # Handler fichier
    file_handler = logging.FileHandler(LOG_FILE, encoding="utf-8")
    file_handler.setLevel(level)
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
    logger.addHandler(file_handler)

    return logger
