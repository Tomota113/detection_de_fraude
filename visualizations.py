"""
=============================================================================
 Visualisations Premium — Détection de Fraude Énergétique STEG
=============================================================================
 Graphiques avancés et esthétiques pour l'analyse exploratoire,
 la comparaison des modèles et l'interprétation des résultats.

 Inclut :
   1. Dashboard récapitulatif du dataset
   2. Distribution des classes (pie + bar enrichis)
   3. Distribution de consommation (fraudeurs vs légitimes)
   4. Heatmap de corrélation avec la cible
   5. Comparaison radar des modèles
   6. Analyse du seuil de décision optimal
   7. Distribution des scores de probabilité
   8. Dashboard de performance finale
=============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from typing import Dict, Optional, List

from sklearn.metrics import (
    roc_curve,
    roc_auc_score,
    precision_recall_curve,
    average_precision_score,
    confusion_matrix,
    f1_score,
)

from config import PLOTS_DIR, setup_logging

logger = setup_logging()

# =============================================================================
# PALETTE & STYLE PREMIUM
# =============================================================================

# Couleurs professionnelles
PALETTE = {
    "blue_dark": "#1E3A5F",
    "blue_main": "#2563EB",
    "blue_light": "#93C5FD",
    "blue_pale": "#DBEAFE",
    "red_main": "#DC2626",
    "red_light": "#FCA5A5",
    "green_main": "#059669",
    "green_light": "#6EE7B7",
    "orange_main": "#D97706",
    "orange_light": "#FCD34D",
    "purple_main": "#7C3AED",
    "purple_light": "#C4B5FD",
    "gray_dark": "#374151",
    "gray_main": "#6B7280",
    "gray_light": "#D1D5DB",
    "bg_dark": "#0F172A",
    "bg_card": "#1E293B",
    "white": "#FFFFFF",
}

# Gradient custom
FRAUD_CMAP = LinearSegmentedColormap.from_list(
    "fraud", ["#DBEAFE", "#2563EB", "#1E3A5F"]
)


def _apply_premium_style(fig, ax_or_axes, dark_mode=True):
    """Applique le style premium à une figure."""
    if dark_mode:
        fig.patch.set_facecolor(PALETTE["bg_dark"])
        axes = ax_or_axes if hasattr(ax_or_axes, '__iter__') else [ax_or_axes]
        for ax in (a for a in np.array(axes).flat if a is not None):
            ax.set_facecolor(PALETTE["bg_card"])
            ax.tick_params(colors=PALETTE["gray_light"])
            ax.xaxis.label.set_color(PALETTE["white"])
            ax.yaxis.label.set_color(PALETTE["white"])
            ax.title.set_color(PALETTE["white"])
            for spine in ax.spines.values():
                spine.set_color(PALETTE["gray_main"])
                spine.set_linewidth(0.5)


# =============================================================================
# 1. DASHBOARD DU DATASET
# =============================================================================

def plot_dataset_overview(client_df: pd.DataFrame, invoice_df: pd.DataFrame,
                          save: bool = True) -> None:
    """Dashboard récapitulatif du dataset avec métriques clés."""
    logger.info("  Génération du dashboard dataset...")

    fig = plt.figure(figsize=(18, 10))
    fig.patch.set_facecolor(PALETTE["bg_dark"])

    gs = gridspec.GridSpec(2, 3, hspace=0.35, wspace=0.3,
                           left=0.06, right=0.94, top=0.90, bottom=0.08)

    # ---- Titre ----
    fig.suptitle("DASHBOARD — Dataset STEG Tunisie",
                 fontsize=22, fontweight="bold", color=PALETTE["white"],
                 y=0.97)

    # ---- 1. Distribution de la cible ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PALETTE["bg_card"])
    target_counts = client_df["target"].value_counts().sort_index()
    colors = [PALETTE["blue_main"], PALETTE["red_main"]]
    wedges, texts, autotexts = ax1.pie(
        target_counts.values,
        labels=None,
        colors=colors,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 14, "fontweight": "bold", "color": PALETTE["white"]},
        wedgeprops={"edgecolor": PALETTE["bg_dark"], "linewidth": 3},
        explode=(0, 0.08),
        pctdistance=0.55,
    )
    ax1.legend(["Légitime", "Fraude"], loc="lower center",
               fontsize=11, framealpha=0, labelcolor=PALETTE["gray_light"],
               ncol=2, bbox_to_anchor=(0.5, -0.05))
    ax1.set_title("Répartition des Classes", fontsize=14,
                   fontweight="bold", color=PALETTE["white"], pad=10)

    # ---- 2. Métriques clés (KPI cards) ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PALETTE["bg_card"])
    ax2.axis("off")

    n_clients = len(client_df)
    n_invoices = len(invoice_df)
    n_fraud = int(target_counts.get(1.0, 0))
    fraud_pct = n_fraud / n_clients * 100

    kpis = [
        ("Clients", f"{n_clients:,}", PALETTE["blue_main"]),
        ("Factures", f"{n_invoices:,}", PALETTE["purple_main"]),
        ("Fraudeurs", f"{n_fraud:,}", PALETTE["red_main"]),
        ("Taux Fraude", f"{fraud_pct:.1f}%", PALETTE["orange_main"]),
    ]

    for i, (label, value, color) in enumerate(kpis):
        row, col = divmod(i, 2)
        x = 0.05 + col * 0.5
        y = 0.75 - row * 0.45

        # Rectangle de fond
        rect = mpatches.FancyBboxPatch(
            (x, y - 0.12), 0.42, 0.35,
            boxstyle="round,pad=0.02",
            facecolor=color, alpha=0.15,
            edgecolor=color, linewidth=1.5,
            transform=ax2.transAxes,
        )
        ax2.add_patch(rect)

        ax2.text(x + 0.21, y + 0.12, value,
                 fontsize=20, fontweight="bold", color=color,
                 ha="center", va="center", transform=ax2.transAxes)
        ax2.text(x + 0.21, y - 0.02, label,
                 fontsize=11, color=PALETTE["gray_light"],
                 ha="center", va="center", transform=ax2.transAxes)

    ax2.set_title("Métriques Clés", fontsize=14,
                   fontweight="bold", color=PALETTE["white"], pad=10)

    # ---- 3. Distribution des catégories client ----
    ax3 = fig.add_subplot(gs[0, 2])
    ax3.set_facecolor(PALETTE["bg_card"])
    categ_counts = client_df["client_catg"].value_counts().head(8)
    bars = ax3.barh(
        range(len(categ_counts)),
        categ_counts.values,
        color=plt.cm.Blues(np.linspace(0.4, 0.9, len(categ_counts)))[::-1],
        edgecolor=PALETTE["bg_dark"], linewidth=1,
        height=0.6,
    )
    ax3.set_yticks(range(len(categ_counts)))
    ax3.set_yticklabels([f"Cat. {c}" for c in categ_counts.index],
                        fontsize=10, color=PALETTE["gray_light"])
    ax3.set_xlabel("Nombre de clients", fontsize=11, color=PALETTE["gray_light"])
    for bar, val in zip(bars, categ_counts.values):
        ax3.text(bar.get_width() + 100, bar.get_y() + bar.get_height() / 2,
                 f"{val:,}", va="center", fontsize=9, color=PALETTE["gray_light"])
    ax3.set_title("Catégories Client", fontsize=14,
                   fontweight="bold", color=PALETTE["white"], pad=10)
    ax3.tick_params(axis="x", colors=PALETTE["gray_light"])
    for spine in ax3.spines.values():
        spine.set_color(PALETTE["gray_main"])

    # ---- 4. Distribution temporelle des factures ----
    ax4 = fig.add_subplot(gs[1, 0:2])
    ax4.set_facecolor(PALETTE["bg_card"])
    invoice_dates = pd.to_datetime(invoice_df["invoice_date"], errors="coerce")
    monthly = invoice_dates.dt.to_period("M").value_counts().sort_index()
    # Limiter aux 60 derniers mois pour lisibilité
    monthly = monthly.tail(60)
    ax4.fill_between(range(len(monthly)), monthly.values,
                     alpha=0.3, color=PALETTE["blue_main"])
    ax4.plot(range(len(monthly)), monthly.values,
             color=PALETTE["blue_main"], linewidth=2)
    ax4.set_xlabel("Mois", fontsize=11, color=PALETTE["gray_light"])
    ax4.set_ylabel("Nb Factures", fontsize=11, color=PALETTE["gray_light"])
    ax4.set_title("Volume Mensuel de Facturation (60 derniers mois)",
                   fontsize=14, fontweight="bold", color=PALETTE["white"], pad=10)
    # Afficher quelques labels de mois
    step = max(1, len(monthly) // 8)
    ticks = list(range(0, len(monthly), step))
    ax4.set_xticks(ticks)
    ax4.set_xticklabels([str(monthly.index[i]) for i in ticks],
                        rotation=45, fontsize=8, color=PALETTE["gray_light"])
    ax4.tick_params(axis="y", colors=PALETTE["gray_light"])
    for spine in ax4.spines.values():
        spine.set_color(PALETTE["gray_main"])

    # ---- 5. Types de compteur ----
    ax5 = fig.add_subplot(gs[1, 2])
    ax5.set_facecolor(PALETTE["bg_card"])
    counter_counts = invoice_df["counter_type"].value_counts()
    colors_counter = [PALETTE["blue_main"], PALETTE["orange_main"],
                      PALETTE["green_main"], PALETTE["purple_main"]][:len(counter_counts)]
    ax5.pie(
        counter_counts.values,
        labels=counter_counts.index,
        colors=colors_counter,
        autopct="%1.1f%%",
        startangle=90,
        textprops={"fontsize": 12, "color": PALETTE["white"]},
        wedgeprops={"edgecolor": PALETTE["bg_dark"], "linewidth": 2},
    )
    ax5.set_title("Types de Compteur", fontsize=14,
                   fontweight="bold", color=PALETTE["white"], pad=10)

    if save:
        path = PLOTS_DIR / "dashboard_dataset.png"
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        logger.info(f"  Dashboard dataset -> {path}")
    plt.close(fig)


# =============================================================================
# 2. DISTRIBUTION CONSOMMATION : FRAUDEURS VS LÉGITIMES
# =============================================================================

def plot_fraud_vs_legitimate(X: pd.DataFrame, y: pd.Series,
                              features: List[str] = None,
                              save: bool = True) -> None:
    """Compare les distributions de features clés entre fraudeurs et légitimes."""
    logger.info("  Distributions fraudeurs vs légitimes...")

    if features is None:
        features = ["conso_mean", "conso_coeff_variation",
                     "max_drop_ratio", "ratio_derniere_moyenne",
                     "nb_baisses_suspectes", "conso_mensuelle_mean"]

    # Ne garder que les features disponibles
    features = [f for f in features if f in X.columns]
    n_feats = len(features)
    if n_feats == 0:
        return

    ncols = 3
    nrows = (n_feats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    fig.patch.set_facecolor(PALETTE["bg_dark"])
    fig.suptitle("Analyse Comparative — Fraudeurs vs Clients Légitimes",
                 fontsize=20, fontweight="bold", color=PALETTE["white"], y=1.02)

    axes_flat = np.array(axes).flat

    for idx, feat in enumerate(features):
        ax = axes_flat[idx]
        ax.set_facecolor(PALETTE["bg_card"])

        data_legit = X.loc[y == 0, feat].clip(
            upper=X.loc[y == 0, feat].quantile(0.99))
        data_fraud = X.loc[y == 1, feat].clip(
            upper=X.loc[y == 1, feat].quantile(0.99))

        ax.hist(data_legit, bins=40, alpha=0.6, color=PALETTE["blue_main"],
                label="Légitime", density=True, edgecolor="none")
        ax.hist(data_fraud, bins=40, alpha=0.6, color=PALETTE["red_main"],
                label="Fraude", density=True, edgecolor="none")

        # Lignes de médiane
        ax.axvline(data_legit.median(), color=PALETTE["blue_light"],
                   linestyle="--", linewidth=2, alpha=0.8)
        ax.axvline(data_fraud.median(), color=PALETTE["red_light"],
                   linestyle="--", linewidth=2, alpha=0.8)

        ax.set_title(feat.replace("_", " ").title(),
                     fontsize=13, fontweight="bold", color=PALETTE["white"])
        ax.legend(fontsize=9, framealpha=0.3, labelcolor=PALETTE["gray_light"])
        ax.tick_params(colors=PALETTE["gray_light"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["gray_main"])

    # Masquer les axes vides
    for idx in range(n_feats, nrows * ncols):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    if save:
        path = PLOTS_DIR / "fraud_vs_legitimate.png"
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        logger.info(f"  Fraud vs Legitimate -> {path}")
    plt.close(fig)


# =============================================================================
# 3. HEATMAP DE CORRÉLATION PREMIUM
# =============================================================================

def plot_correlation_heatmap(X: pd.DataFrame, y: pd.Series,
                              n_top: int = 15, save: bool = True) -> None:
    """Heatmap des corrélations entre features et cible, style dark premium."""
    logger.info("  Heatmap de corrélation premium...")

    # Top features les plus corrélées à la cible
    corr_target = X.corrwith(y).abs().sort_values(ascending=False)
    top_features = corr_target.head(n_top).index.tolist()

    # Matrice de corrélation
    corr = X[top_features].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))

    fig, ax = plt.subplots(figsize=(14, 11))
    fig.patch.set_facecolor(PALETTE["bg_dark"])
    ax.set_facecolor(PALETTE["bg_card"])

    # Custom colormap
    cmap = LinearSegmentedColormap.from_list(
        "custom", [PALETTE["blue_main"], PALETTE["bg_card"],
                   PALETTE["red_main"]]
    )

    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f",
        cmap=cmap, center=0, vmin=-1, vmax=1,
        square=True, linewidths=1, linecolor=PALETTE["bg_dark"],
        cbar_kws={"shrink": 0.8, "label": "Corrélation"},
        ax=ax,
        annot_kws={"fontsize": 9, "color": PALETTE["white"]},
    )

    ax.set_title("Matrice de Corrélation — Top 15 Features",
                 fontsize=16, fontweight="bold", color=PALETTE["white"], pad=20)
    ax.tick_params(colors=PALETTE["gray_light"])
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", fontsize=10,
             color=PALETTE["gray_light"])
    plt.setp(ax.get_yticklabels(), fontsize=10, color=PALETTE["gray_light"])

    # Colorbar styling
    cbar = ax.collections[0].colorbar
    cbar.ax.tick_params(colors=PALETTE["gray_light"])
    cbar.set_label("Corrélation", color=PALETTE["gray_light"], fontsize=12)

    plt.tight_layout()
    if save:
        path = PLOTS_DIR / "correlation_heatmap.png"
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        logger.info(f"  Correlation heatmap -> {path}")
    plt.close(fig)


# =============================================================================
# 4. COMPARAISON RADAR DES MODÈLES
# =============================================================================

def plot_model_radar(cv_results: Dict[str, dict], save: bool = True) -> None:
    """Graphique radar comparant les 3 modèles sur toutes les métriques."""
    logger.info("  Radar de comparaison des modèles...")

    metrics = ["roc_auc_mean", "pr_auc_mean", "f1_mean"]
    labels = ["ROC-AUC", "PR-AUC", "F1-Score"]
    model_names = list(cv_results.keys())
    colors = [PALETTE["blue_main"], PALETTE["green_main"], PALETTE["red_main"]]

    # Données
    angles = np.linspace(0, 2 * np.pi, len(labels), endpoint=False).tolist()
    angles += angles[:1]  # Fermer le polygone

    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
    fig.patch.set_facecolor(PALETTE["bg_dark"])
    ax.set_facecolor(PALETTE["bg_card"])

    for i, (name, results) in enumerate(cv_results.items()):
        values = [results[m] for m in metrics]
        values += values[:1]

        ax.plot(angles, values, "o-", linewidth=2.5, color=colors[i],
                label=name, markersize=8)
        ax.fill(angles, values, alpha=0.15, color=colors[i])

    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(labels, fontsize=14, fontweight="bold",
                       color=PALETTE["white"])
    ax.set_ylim(0, 1)
    ax.set_rticks([0.2, 0.4, 0.6, 0.8])
    ax.set_yticklabels(["0.2", "0.4", "0.6", "0.8"],
                       fontsize=9, color=PALETTE["gray_light"])
    ax.grid(True, color=PALETTE["gray_main"], alpha=0.3)

    # Styling des spines
    ax.spines["polar"].set_color(PALETTE["gray_main"])
    ax.tick_params(colors=PALETTE["gray_light"])

    ax.legend(loc="upper right", bbox_to_anchor=(1.3, 1.1),
              fontsize=13, framealpha=0.3, labelcolor=PALETTE["white"],
              facecolor=PALETTE["bg_card"], edgecolor=PALETTE["gray_main"])

    ax.set_title("Comparaison des Modèles",
                 fontsize=18, fontweight="bold", color=PALETTE["white"],
                 pad=30, y=1.08)

    plt.tight_layout()
    if save:
        path = PLOTS_DIR / "model_radar.png"
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        logger.info(f"  Model radar -> {path}")
    plt.close(fig)


# =============================================================================
# 5. COMPARAISON EN BARRES GROUPÉES
# =============================================================================

def plot_model_comparison_bars(cv_results: Dict[str, dict],
                                save: bool = True) -> None:
    """Barres groupées comparant les 3 modèles (style glassmorphism)."""
    logger.info("  Barres de comparaison des modèles...")

    model_names = list(cv_results.keys())
    metrics_keys = ["roc_auc_mean", "pr_auc_mean", "f1_mean"]
    metrics_labels = ["ROC-AUC", "PR-AUC", "F1-Score"]
    metrics_std = ["roc_auc_std", "pr_auc_std", "f1_std"]
    bar_colors = [PALETTE["blue_main"], PALETTE["green_main"],
                  PALETTE["orange_main"]]

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor(PALETTE["bg_dark"])
    ax.set_facecolor(PALETTE["bg_card"])

    x = np.arange(len(model_names))
    width = 0.22

    for i, (key, label, std_key, color) in enumerate(
        zip(metrics_keys, metrics_labels, metrics_std, bar_colors)
    ):
        means = [cv_results[m][key] for m in model_names]
        stds = [cv_results[m][std_key] for m in model_names]

        bars = ax.bar(
            x + i * width - width, means, width,
            yerr=stds, label=label,
            color=color, alpha=0.85,
            edgecolor="white", linewidth=0.5,
            capsize=4, error_kw={"linewidth": 1.5, "color": PALETTE["gray_light"]},
        )

        # Valeurs sur les barres
        for bar, mean in zip(bars, means):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.008,
                f"{mean:.3f}", ha="center", va="bottom",
                fontsize=10, fontweight="bold", color=PALETTE["white"],
            )

    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=14, fontweight="bold",
                       color=PALETTE["white"])
    ax.set_ylabel("Score", fontsize=13, fontweight="bold",
                  color=PALETTE["white"])
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=12, framealpha=0.3, labelcolor=PALETTE["white"],
              facecolor=PALETTE["bg_card"], edgecolor=PALETTE["gray_main"],
              loc="upper left")
    ax.grid(axis="y", alpha=0.15, color=PALETTE["gray_light"])
    ax.set_title("Validation Croisée 5-Fold — Comparaison des Modèles",
                 fontsize=17, fontweight="bold", color=PALETTE["white"], pad=15)

    for spine in ax.spines.values():
        spine.set_color(PALETTE["gray_main"])
        spine.set_linewidth(0.5)
    ax.tick_params(colors=PALETTE["gray_light"])

    plt.tight_layout()
    if save:
        path = PLOTS_DIR / "model_comparison_bars.png"
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        logger.info(f"  Model comparison bars -> {path}")
    plt.close(fig)


# =============================================================================
# 6. ANALYSE DU SEUIL DE DÉCISION
# =============================================================================

def plot_threshold_analysis(y_true: np.ndarray, y_proba: np.ndarray,
                             model_name: str = "Modèle",
                             save: bool = True) -> float:
    """Analyse détaillée du seuil de décision optimal."""
    logger.info("  Analyse du seuil de décision...")

    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_proba)
    f1_scores = 2 * (precision * recall) / (precision + recall + 1e-10)

    # Seuil optimal
    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds_pr[best_idx] if best_idx < len(thresholds_pr) else 0.5

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18, 7))
    fig.patch.set_facecolor(PALETTE["bg_dark"])

    # ---- Gauche : Precision, Recall, F1 vs Seuil ----
    ax1.set_facecolor(PALETTE["bg_card"])
    thresholds_plot = thresholds_pr[:len(precision) - 1]

    ax1.plot(thresholds_plot, precision[:-1], color=PALETTE["blue_main"],
             linewidth=2.5, label="Precision")
    ax1.plot(thresholds_plot, recall[:-1], color=PALETTE["red_main"],
             linewidth=2.5, label="Recall")
    ax1.plot(thresholds_plot, f1_scores[:-1], color=PALETTE["green_main"],
             linewidth=2.5, label="F1-Score", linestyle="--")

    ax1.axvline(best_threshold, color=PALETTE["orange_main"],
                linewidth=2, linestyle=":", alpha=0.8,
                label=f"Seuil optimal = {best_threshold:.3f}")

    ax1.scatter(best_threshold, f1_scores[best_idx],
                color=PALETTE["orange_main"], s=150, zorder=5,
                edgecolors="white", linewidth=2)

    ax1.set_xlabel("Seuil de Décision", fontsize=13, fontweight="bold",
                   color=PALETTE["white"])
    ax1.set_ylabel("Score", fontsize=13, fontweight="bold",
                   color=PALETTE["white"])
    ax1.set_title("Precision / Recall / F1 vs Seuil",
                  fontsize=15, fontweight="bold", color=PALETTE["white"])
    ax1.legend(fontsize=11, framealpha=0.3, labelcolor=PALETTE["white"],
               facecolor=PALETTE["bg_card"])
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1.05)
    ax1.grid(alpha=0.15, color=PALETTE["gray_light"])
    for spine in ax1.spines.values():
        spine.set_color(PALETTE["gray_main"])
    ax1.tick_params(colors=PALETTE["gray_light"])

    # ---- Droite : Distribution des scores ----
    ax2.set_facecolor(PALETTE["bg_card"])

    ax2.hist(y_proba[y_true == 0], bins=50, alpha=0.7,
             color=PALETTE["blue_main"], label="Légitime",
             density=True, edgecolor="none")
    ax2.hist(y_proba[y_true == 1], bins=50, alpha=0.7,
             color=PALETTE["red_main"], label="Fraude",
             density=True, edgecolor="none")

    ax2.axvline(0.5, color=PALETTE["gray_light"], linewidth=2,
                linestyle="--", label="Seuil par défaut (0.5)")
    ax2.axvline(best_threshold, color=PALETTE["orange_main"],
                linewidth=2, linestyle=":",
                label=f"Seuil optimal ({best_threshold:.3f})")

    ax2.set_xlabel("Probabilité de Fraude", fontsize=13, fontweight="bold",
                   color=PALETTE["white"])
    ax2.set_ylabel("Densité", fontsize=13, fontweight="bold",
                   color=PALETTE["white"])
    ax2.set_title("Distribution des Scores de Probabilité",
                  fontsize=15, fontweight="bold", color=PALETTE["white"])
    ax2.legend(fontsize=11, framealpha=0.3, labelcolor=PALETTE["white"],
               facecolor=PALETTE["bg_card"])
    ax2.grid(alpha=0.15, color=PALETTE["gray_light"])
    for spine in ax2.spines.values():
        spine.set_color(PALETTE["gray_main"])
    ax2.tick_params(colors=PALETTE["gray_light"])

    fig.suptitle(f"Analyse du Seuil de Décision — {model_name}",
                 fontsize=19, fontweight="bold", color=PALETTE["white"],
                 y=1.02)

    plt.tight_layout()
    if save:
        path = PLOTS_DIR / "threshold_analysis.png"
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        logger.info(f"  Threshold analysis -> {path}")
    plt.close(fig)

    return best_threshold


# =============================================================================
# 7. DASHBOARD DE PERFORMANCE FINALE
# =============================================================================

def plot_performance_dashboard(y_true: np.ndarray, y_proba: np.ndarray,
                                model_name: str, cv_results: Dict[str, dict],
                                feature_importance_df: pd.DataFrame = None,
                                save: bool = True) -> None:
    """Dashboard final récapitulatif : 4 panels en un seul graphique."""
    logger.info("  Dashboard de performance finale...")

    y_pred = (y_proba >= 0.5).astype(int)

    fig = plt.figure(figsize=(20, 14))
    fig.patch.set_facecolor(PALETTE["bg_dark"])
    gs = gridspec.GridSpec(2, 2, hspace=0.3, wspace=0.25,
                           left=0.06, right=0.94, top=0.92, bottom=0.06)

    fig.suptitle(f"DASHBOARD DE PERFORMANCE — {model_name}",
                 fontsize=22, fontweight="bold", color=PALETTE["white"],
                 y=0.97)

    # ---- 1. Matrice de Confusion (haut-gauche) ----
    ax1 = fig.add_subplot(gs[0, 0])
    ax1.set_facecolor(PALETTE["bg_card"])
    cm = confusion_matrix(y_true, y_pred)
    cm_pct = cm.astype("float") / cm.sum() * 100

    sns.heatmap(cm, annot=False, cmap="Blues", ax=ax1,
                xticklabels=["Légitime", "Fraude"],
                yticklabels=["Légitime", "Fraude"],
                linewidths=2, linecolor=PALETTE["bg_dark"],
                cbar=False)

    for i in range(2):
        for j in range(2):
            text = f"{cm[i, j]:,}\n({cm_pct[i, j]:.1f}%)"
            ax1.text(j + 0.5, i + 0.5, text,
                     ha="center", va="center", fontsize=13,
                     fontweight="bold",
                     color="white" if cm[i, j] > cm.max() / 2 else PALETTE["gray_dark"])

    ax1.set_xlabel("Prédiction", fontsize=12, color=PALETTE["white"])
    ax1.set_ylabel("Réalité", fontsize=12, color=PALETTE["white"])
    ax1.set_title("Matrice de Confusion", fontsize=15,
                  fontweight="bold", color=PALETTE["white"])
    ax1.tick_params(colors=PALETTE["gray_light"])

    # ---- 2. Courbes ROC + PR (haut-droit) ----
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.set_facecolor(PALETTE["bg_card"])

    fpr, tpr, _ = roc_curve(y_true, y_proba)
    roc_auc = roc_auc_score(y_true, y_proba)
    precision, recall, _ = precision_recall_curve(y_true, y_proba)
    pr_auc = average_precision_score(y_true, y_proba)

    ax2.plot(fpr, tpr, color=PALETTE["blue_main"], linewidth=2.5,
             label=f"ROC (AUC={roc_auc:.4f})")
    ax2.fill_between(fpr, tpr, alpha=0.1, color=PALETTE["blue_main"])

    ax2.plot(recall, precision, color=PALETTE["green_main"], linewidth=2.5,
             label=f"PR (AP={pr_auc:.4f})")
    ax2.fill_between(recall, precision, alpha=0.1, color=PALETTE["green_main"])

    ax2.plot([0, 1], [0, 1], "--", color=PALETTE["gray_main"], linewidth=1)

    ax2.set_xlabel("FPR / Recall", fontsize=12, color=PALETTE["white"])
    ax2.set_ylabel("TPR / Precision", fontsize=12, color=PALETTE["white"])
    ax2.set_title("Courbes ROC & Precision-Recall", fontsize=15,
                  fontweight="bold", color=PALETTE["white"])
    ax2.legend(fontsize=12, framealpha=0.3, labelcolor=PALETTE["white"],
               facecolor=PALETTE["bg_card"])
    ax2.grid(alpha=0.15, color=PALETTE["gray_light"])
    ax2.tick_params(colors=PALETTE["gray_light"])
    for spine in ax2.spines.values():
        spine.set_color(PALETTE["gray_main"])

    # ---- 3. Top 10 Feature Importance (bas-gauche) ----
    ax3 = fig.add_subplot(gs[1, 0])
    ax3.set_facecolor(PALETTE["bg_card"])

    if feature_importance_df is not None and len(feature_importance_df) > 0:
        fi = feature_importance_df.head(10)
        colors_fi = plt.cm.Blues(np.linspace(0.4, 0.95, len(fi)))[::-1]
        bars = ax3.barh(range(len(fi)), fi["importance"].values[::-1],
                        color=colors_fi, edgecolor="none", height=0.65)
        ax3.set_yticks(range(len(fi)))
        ax3.set_yticklabels(fi["feature"].values[::-1], fontsize=10,
                            color=PALETTE["gray_light"])
        for bar, val in zip(bars, fi["importance"].values[::-1]):
            ax3.text(bar.get_width() + 0.001, bar.get_y() + bar.get_height() / 2,
                     f"{val:.4f}", va="center", fontsize=9,
                     color=PALETTE["gray_light"])
    else:
        ax3.text(0.5, 0.5, "N/A", ha="center", va="center",
                 fontsize=20, color=PALETTE["gray_main"], transform=ax3.transAxes)

    ax3.set_title("Top 10 Variables Discriminantes", fontsize=15,
                  fontweight="bold", color=PALETTE["white"])
    ax3.grid(axis="x", alpha=0.15, color=PALETTE["gray_light"])
    ax3.tick_params(colors=PALETTE["gray_light"])
    for spine in ax3.spines.values():
        spine.set_color(PALETTE["gray_main"])

    # ---- 4. KPI Cards (bas-droit) ----
    ax4 = fig.add_subplot(gs[1, 1])
    ax4.set_facecolor(PALETTE["bg_card"])
    ax4.axis("off")

    f1 = f1_score(y_true, y_pred, zero_division=0)
    recall_fraud = cm[1, 1] / (cm[1, 0] + cm[1, 1]) if (cm[1, 0] + cm[1, 1]) > 0 else 0
    precision_fraud = cm[1, 1] / (cm[0, 1] + cm[1, 1]) if (cm[0, 1] + cm[1, 1]) > 0 else 0

    kpis = [
        ("ROC-AUC", f"{roc_auc:.4f}", PALETTE["blue_main"], "Discrimination globale"),
        ("PR-AUC", f"{pr_auc:.4f}", PALETTE["green_main"], "Performance sur fraude"),
        ("Recall Fraude", f"{recall_fraud:.1%}", PALETTE["red_main"], "Taux de détection"),
        ("Precision Fraude", f"{precision_fraud:.1%}", PALETTE["orange_main"], "Fiabilité alerte"),
        ("F1-Score", f"{f1:.4f}", PALETTE["purple_main"], "Équilibre P/R"),
        ("Fraudeurs Détectés", f"{cm[1,1]:,}/{cm[1,0]+cm[1,1]:,}", PALETTE["blue_light"], "Vrais positifs"),
    ]

    for i, (label, value, color, desc) in enumerate(kpis):
        row, col = divmod(i, 2)
        x = 0.03 + col * 0.5
        y_pos = 0.78 - row * 0.32

        rect = mpatches.FancyBboxPatch(
            (x, y_pos - 0.08), 0.44, 0.28,
            boxstyle="round,pad=0.02",
            facecolor=color, alpha=0.12,
            edgecolor=color, linewidth=1.5,
            transform=ax4.transAxes,
        )
        ax4.add_patch(rect)

        ax4.text(x + 0.22, y_pos + 0.1, value,
                 fontsize=22, fontweight="bold", color=color,
                 ha="center", va="center", transform=ax4.transAxes)
        ax4.text(x + 0.22, y_pos + 0.0, label,
                 fontsize=12, fontweight="bold", color=PALETTE["white"],
                 ha="center", va="center", transform=ax4.transAxes)
        ax4.text(x + 0.22, y_pos - 0.06, desc,
                 fontsize=9, color=PALETTE["gray_light"],
                 ha="center", va="center", transform=ax4.transAxes,
                 style="italic")

    ax4.set_title("Métriques de Performance", fontsize=15,
                  fontweight="bold", color=PALETTE["white"], pad=15)

    if save:
        path = PLOTS_DIR / "performance_dashboard.png"
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        logger.info(f"  Performance dashboard -> {path}")
    plt.close(fig)


# =============================================================================
# 8. BOXPLOTS COMPARATIFS
# =============================================================================

def plot_boxplots_comparison(X: pd.DataFrame, y: pd.Series,
                              features: List[str] = None,
                              save: bool = True) -> None:
    """Boxplots côte à côte pour comparer les features entre classes."""
    logger.info("  Boxplots comparatifs...")

    if features is None:
        features = ["conso_mean", "conso_std", "conso_coeff_variation",
                     "max_drop_ratio", "nb_baisses_suspectes",
                     "ratio_zero_conso"]

    features = [f for f in features if f in X.columns]
    n_feats = len(features)
    if n_feats == 0:
        return

    ncols = 3
    nrows = (n_feats + ncols - 1) // ncols

    fig, axes = plt.subplots(nrows, ncols, figsize=(18, 5 * nrows))
    fig.patch.set_facecolor(PALETTE["bg_dark"])
    fig.suptitle("Boxplots — Features Clés par Classe",
                 fontsize=20, fontweight="bold", color=PALETTE["white"], y=1.02)

    axes_flat = np.array(axes).flat

    for idx, feat in enumerate(features):
        ax = axes_flat[idx]
        ax.set_facecolor(PALETTE["bg_card"])

        data = pd.DataFrame({"value": X[feat], "class": y.map({0: "Légitime", 1: "Fraude"})})
        # Clip outliers pour lisibilité
        q99 = data["value"].quantile(0.99)
        data["value"] = data["value"].clip(upper=q99)

        bp = ax.boxplot(
            [data.loc[data["class"] == "Légitime", "value"],
             data.loc[data["class"] == "Fraude", "value"]],
            labels=["Légitime", "Fraude"],
            patch_artist=True,
            widths=0.5,
            showfliers=False,
        )

        colors_box = [PALETTE["blue_main"], PALETTE["red_main"]]
        for patch, color in zip(bp["boxes"], colors_box):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
            patch.set_edgecolor("white")
        for element in ["whiskers", "caps"]:
            for line in bp[element]:
                line.set_color(PALETTE["gray_light"])
        for median in bp["medians"]:
            median.set_color(PALETTE["white"])
            median.set_linewidth(2)

        ax.set_title(feat.replace("_", " ").title(),
                     fontsize=13, fontweight="bold", color=PALETTE["white"])
        ax.tick_params(colors=PALETTE["gray_light"])
        for spine in ax.spines.values():
            spine.set_color(PALETTE["gray_main"])

    for idx in range(n_feats, nrows * ncols):
        axes_flat[idx].set_visible(False)

    plt.tight_layout()
    if save:
        path = PLOTS_DIR / "boxplots_comparison.png"
        fig.savefig(path, dpi=300, facecolor=fig.get_facecolor(),
                    bbox_inches="tight")
        logger.info(f"  Boxplots comparison -> {path}")
    plt.close(fig)


# =============================================================================
# ORCHESTRATEUR
# =============================================================================

def generate_all_premium_visuals(
    client_df: pd.DataFrame,
    invoice_df: pd.DataFrame,
    X: pd.DataFrame,
    y: pd.Series,
    y_true: np.ndarray,
    y_proba: np.ndarray,
    model_name: str,
    cv_results: Dict[str, dict],
    feature_importance_df: pd.DataFrame = None,
) -> None:
    """Génère l'ensemble des visualisations premium."""
    logger.info("=" * 70)
    logger.info("GÉNÉRATION DES VISUALISATIONS PREMIUM")
    logger.info("=" * 70)

    plot_dataset_overview(client_df, invoice_df)
    plot_fraud_vs_legitimate(X, y)
    plot_correlation_heatmap(X, y)
    plot_model_radar(cv_results)
    plot_model_comparison_bars(cv_results)
    plot_threshold_analysis(y_true, y_proba, model_name)
    plot_performance_dashboard(y_true, y_proba, model_name, cv_results,
                                feature_importance_df)
    plot_boxplots_comparison(X, y)

    logger.info(f"  8 visualisations premium générées dans {PLOTS_DIR}")
    logger.info("=" * 70)
