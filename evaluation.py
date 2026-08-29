"""
=============================================================================
 Évaluation Critique & Explicabilité du Modèle
 Détection de Fraude Énergétique — STEG Tunisie
=============================================================================
 Ce module génère l'ensemble des visualisations et rapports nécessaires
 pour évaluer un modèle de détection de fraude en contexte production :

   1. Matrice de Confusion (heatmap annotée)
   2. Courbe ROC-AUC (discrimination globale)
   3. Courbe Precision-Recall (performance sur fraude — CRITIQUE)
   4. Feature Importance (Top 20 — explicabilité métier)
   5. Rapport de classification textuel (precision, recall, F1)

 Tous les graphiques sont sauvegardés dans outputs/plots/ au format PNG
 haute résolution (300 DPI) pour insertion dans un rapport académique.
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional

from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    f1_score,
    accuracy_score,
)

from config import PLOTS_DIR, N_TOP_FEATURES, setup_logging

logger = setup_logging()

# Style global pour les graphiques
plt.style.use("seaborn-v0_8-darkgrid")
plt.rcParams.update({
    "figure.figsize": (10, 7),
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 100,
    "savefig.dpi": 300,
})

# Palette de couleurs professionnelle
COLORS = {
    "primary": "#2563EB",     # Bleu professionnel
    "secondary": "#DC2626",   # Rouge alerte
    "accent": "#059669",      # Vert succès
    "neutral": "#6B7280",     # Gris neutre
    "gradient": ["#1E40AF", "#3B82F6", "#93C5FD"],  # Dégradé bleu
}


class ModelEvaluator:
    """Évalue et explique un modèle de détection de fraude.

    Génère un rapport complet incluant les métriques quantitatives
    et les visualisations pour un audit de niveau production.

    Parameters
    ----------
    model : object
        Modèle entraîné avec predict_proba().
    model_name : str
        Nom du modèle (pour les titres des graphiques).
    feature_names : list
        Liste des noms de features.
    """

    def __init__(
        self,
        model,
        model_name: str,
        feature_names: list,
    ):
        self.model = model
        self.model_name = model_name
        self.feature_names = feature_names
        self.metrics: dict = {}

    # -----------------------------------------------------------------
    # 1. Matrice de Confusion
    # -----------------------------------------------------------------

    def plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        save: bool = True,
    ) -> None:
        """Trace la matrice de confusion sous forme de heatmap annotée.

        La matrice montre les vrais positifs, faux positifs, vrais
        négatifs et faux négatifs avec les effectifs et pourcentages.

        Parameters
        ----------
        y_true : array-like
            Vraies étiquettes.
        y_pred : array-like
            Prédictions binaires.
        save : bool
            Si True, sauvegarde dans outputs/plots/.
        """
        cm = confusion_matrix(y_true, y_pred)
        cm_pct = cm.astype("float") / cm.sum() * 100

        fig, ax = plt.subplots(figsize=(8, 6))

        # Heatmap avec annotations détaillées
        sns.heatmap(
            cm,
            annot=False,
            fmt="d",
            cmap="Blues",
            xticklabels=["Légitime (0)", "Fraude (1)"],
            yticklabels=["Légitime (0)", "Fraude (1)"],
            ax=ax,
            linewidths=2,
            linecolor="white",
            cbar_kws={"label": "Nombre de clients"},
        )

        # Annotations personnalisées avec effectif + pourcentage
        for i in range(2):
            for j in range(2):
                text = f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"
                ax.text(
                    j + 0.5, i + 0.5, text,
                    ha="center", va="center",
                    fontsize=14, fontweight="bold",
                    color="white" if cm[i, j] > cm.max() / 2 else "black",
                )

        ax.set_xlabel("Prédiction", fontsize=13, fontweight="bold")
        ax.set_ylabel("Réalité", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Matrice de Confusion — {self.model_name}",
            fontsize=15, fontweight="bold", pad=15,
        )

        plt.tight_layout()
        if save:
            path = PLOTS_DIR / "confusion_matrix.png"
            fig.savefig(path)
            logger.info(f"  📊 Matrice de confusion → {path}")
        plt.close(fig)

    # -----------------------------------------------------------------
    # 2. Courbe ROC-AUC
    # -----------------------------------------------------------------

    def plot_roc_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save: bool = True,
    ) -> float:
        """Trace la courbe ROC et calcule l'AUC.

        La courbe ROC montre la capacité de discrimination du modèle
        à travers tous les seuils de décision possibles.

        Parameters
        ----------
        y_true : array-like
            Vraies étiquettes.
        y_proba : array-like
            Probabilités de la classe positive.
        save : bool
            Si True, sauvegarde dans outputs/plots/.

        Returns
        -------
        float
            Score ROC-AUC.
        """
        fpr, tpr, thresholds = roc_curve(y_true, y_proba)
        roc_auc = roc_auc_score(y_true, y_proba)

        fig, ax = plt.subplots(figsize=(9, 7))

        # Courbe ROC
        ax.plot(
            fpr, tpr,
            color=COLORS["primary"],
            linewidth=2.5,
            label=f"{self.model_name} (AUC = {roc_auc:.4f})",
        )

        # Zone sous la courbe
        ax.fill_between(fpr, tpr, alpha=0.15, color=COLORS["primary"])

        # Diagonale (classifieur aléatoire)
        ax.plot(
            [0, 1], [0, 1],
            color=COLORS["neutral"],
            linewidth=1.5,
            linestyle="--",
            label="Classifieur aléatoire (AUC = 0.5000)",
        )

        # Point optimal (Youden's J)
        j_scores = tpr - fpr
        best_idx = np.argmax(j_scores)
        ax.scatter(
            fpr[best_idx], tpr[best_idx],
            color=COLORS["secondary"],
            s=120, zorder=5, edgecolors="white", linewidth=2,
            label=f"Seuil optimal = {thresholds[best_idx]:.3f}",
        )

        ax.set_xlabel("Taux de Faux Positifs (FPR)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Taux de Vrais Positifs (TPR)", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Courbe ROC — {self.model_name}",
            fontsize=15, fontweight="bold", pad=15,
        )
        ax.legend(loc="lower right", fontsize=11, framealpha=0.9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            path = PLOTS_DIR / "roc_curve.png"
            fig.savefig(path)
            logger.info(f"  📊 Courbe ROC-AUC → {path}")
        plt.close(fig)

        return roc_auc

    # -----------------------------------------------------------------
    # 3. Courbe Precision-Recall
    # -----------------------------------------------------------------

    def plot_precision_recall_curve(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        save: bool = True,
    ) -> float:
        """Trace la courbe Precision-Recall et calcule l'AP.

        CRITIQUE pour la fraude : cette courbe est plus informative que
        la ROC quand les classes sont très déséquilibrées. Elle montre
        le compromis entre la précision (combien de prédictions "fraude"
        sont vraies) et le rappel (combien de fraudes sont détectées).

        Parameters
        ----------
        y_true : array-like
            Vraies étiquettes.
        y_proba : array-like
            Probabilités de la classe positive.
        save : bool
            Si True, sauvegarde dans outputs/plots/.

        Returns
        -------
        float
            Score Average Precision (AP / PR-AUC).
        """
        precision, recall, thresholds = precision_recall_curve(y_true, y_proba)
        ap_score = average_precision_score(y_true, y_proba)

        # Baseline = proportion de positifs
        baseline = y_true.mean()

        fig, ax = plt.subplots(figsize=(9, 7))

        # Courbe PR
        ax.plot(
            recall, precision,
            color=COLORS["accent"],
            linewidth=2.5,
            label=f"{self.model_name} (AP = {ap_score:.4f})",
        )

        # Zone sous la courbe
        ax.fill_between(recall, precision, alpha=0.15, color=COLORS["accent"])

        # Baseline (classifieur aléatoire)
        ax.axhline(
            y=baseline,
            color=COLORS["neutral"],
            linewidth=1.5,
            linestyle="--",
            label=f"Baseline aléatoire (AP = {baseline:.4f})",
        )

        # Point F1 optimal
        f1_scores_curve = 2 * (precision * recall) / (precision + recall + 1e-10)
        best_f1_idx = np.argmax(f1_scores_curve)
        ax.scatter(
            recall[best_f1_idx], precision[best_f1_idx],
            color=COLORS["secondary"],
            s=120, zorder=5, edgecolors="white", linewidth=2,
            label=f"F1 max = {f1_scores_curve[best_f1_idx]:.3f}",
        )

        ax.set_xlabel("Rappel (Recall)", fontsize=13, fontweight="bold")
        ax.set_ylabel("Précision (Precision)", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Courbe Precision-Recall — {self.model_name}",
            fontsize=15, fontweight="bold", pad=15,
        )
        ax.legend(loc="upper right", fontsize=11, framealpha=0.9)
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([0, 1.05])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        if save:
            path = PLOTS_DIR / "precision_recall_curve.png"
            fig.savefig(path)
            logger.info(f"  📊 Courbe PR → {path}")
        plt.close(fig)

        return ap_score

    # -----------------------------------------------------------------
    # 4. Feature Importance
    # -----------------------------------------------------------------

    def plot_feature_importance(
        self,
        n_top: int = N_TOP_FEATURES,
        save: bool = True,
    ) -> pd.DataFrame:
        """Trace l'importance des variables (Top N).

        Utilise l'importance native du modèle (gain pour les arbres
        de gradient boosting) pour identifier les facteurs discriminants.

        Parameters
        ----------
        n_top : int
            Nombre de features à afficher.
        save : bool
            Si True, sauvegarde dans outputs/plots/.

        Returns
        -------
        pd.DataFrame
            DataFrame des importances triées.
        """
        # Récupérer l'importance des features
        if hasattr(self.model, "feature_importances_"):
            importances = self.model.feature_importances_
        else:
            logger.warning("Le modèle n'a pas d'attribut feature_importances_")
            return pd.DataFrame()

        # Construire le DataFrame
        fi_df = pd.DataFrame({
            "feature": self.feature_names,
            "importance": importances,
        }).sort_values("importance", ascending=False).head(n_top)

        # Graphique
        fig, ax = plt.subplots(figsize=(10, 8))

        colors = plt.cm.Blues(np.linspace(0.4, 0.9, len(fi_df)))[::-1]

        bars = ax.barh(
            range(len(fi_df)),
            fi_df["importance"].values[::-1],
            color=colors,
            edgecolor="white",
            linewidth=0.5,
            height=0.7,
        )

        ax.set_yticks(range(len(fi_df)))
        ax.set_yticklabels(fi_df["feature"].values[::-1], fontsize=11)
        ax.set_xlabel("Importance (Gain)", fontsize=13, fontweight="bold")
        ax.set_title(
            f"Top {n_top} Variables Discriminantes — {self.model_name}",
            fontsize=15, fontweight="bold", pad=15,
        )

        # Ajouter les valeurs sur les barres
        for i, (bar, val) in enumerate(
            zip(bars, fi_df["importance"].values[::-1])
        ):
            ax.text(
                bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                ha="left", va="center", fontsize=10, color=COLORS["neutral"],
            )

        ax.grid(axis="x", alpha=0.3)
        plt.tight_layout()
        if save:
            path = PLOTS_DIR / "feature_importance.png"
            fig.savefig(path)
            logger.info(f"  📊 Feature importance → {path}")
        plt.close(fig)

        return fi_df

    # -----------------------------------------------------------------
    # 5. Rapport Complet
    # -----------------------------------------------------------------

    def generate_full_report(
        self,
        y_true: np.ndarray,
        y_proba: np.ndarray,
        threshold: float = 0.5,
    ) -> dict:
        """Génère le rapport d'évaluation complet.

        Orchestre l'ensemble des évaluations et produit un rapport
        unifié avec toutes les métriques et visualisations.

        Parameters
        ----------
        y_true : array-like
            Vraies étiquettes.
        y_proba : array-like
            Probabilités de la classe positive.
        threshold : float
            Seuil de décision binaire (par défaut 0.5).

        Returns
        -------
        dict
            Dictionnaire contenant toutes les métriques calculées.
        """
        logger.info("=" * 70)
        logger.info(f"RAPPORT D'ÉVALUATION — {self.model_name}")
        logger.info("=" * 70)

        y_pred = (y_proba >= threshold).astype(int)

        # --- Métriques numériques ---
        roc_auc = roc_auc_score(y_true, y_proba)
        pr_auc = average_precision_score(y_true, y_proba)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        accuracy = accuracy_score(y_true, y_pred)

        self.metrics = {
            "accuracy": accuracy,
            "roc_auc": roc_auc,
            "pr_auc": pr_auc,
            "f1_score": f1,
            "threshold": threshold,
        }

        logger.info(f"  Accuracy  : {accuracy:.4f}")
        logger.info(f"  ROC-AUC   : {roc_auc:.4f}")
        logger.info(f"  PR-AUC    : {pr_auc:.4f}")
        logger.info(f"  F1-Score  : {f1:.4f}")

        # --- Rapport de classification ---
        report = classification_report(
            y_true, y_pred,
            target_names=["Légitime", "Fraude"],
            zero_division=0,
        )
        logger.info(f"\n{report}")

        # --- Visualisations ---
        logger.info("-" * 40)
        logger.info("Génération des graphiques...")

        self.plot_confusion_matrix(y_true, y_pred)
        self.plot_roc_curve(y_true, y_proba)
        self.plot_precision_recall_curve(y_true, y_proba)
        fi_df = self.plot_feature_importance()

        logger.info("-" * 40)
        logger.info("Rapport d'évaluation terminé.")
        logger.info(f"Graphiques sauvegardés dans : {PLOTS_DIR}")

        self.metrics["classification_report"] = report
        self.metrics["feature_importance"] = fi_df

        return self.metrics
