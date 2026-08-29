"""
=============================================================================
 Pipeline de Prétraitement & Feature Engineering
 Détection de Fraude Énergétique — STEG Tunisie
=============================================================================
 Ce module implémente le cœur analytique du pipeline :
   1. Chargement et nettoyage des données brutes
   2. Feature Engineering avancé à partir de l'historique de facturation
   3. Construction de la matrice de features finale prête pour le ML

 Architecture des features extraites (30+ variables) :
   - Features Client  : ancienneté, catégorie, région, district
   - Features Factures : agrégats statistiques de consommation
   - Features Temporelles : saisonnalité, tendances, variations
   - Indicateurs de Fraude : baisses suspectes, anomalies compteur
=============================================================================
"""

import warnings
import numpy as np
import pandas as pd
from typing import Tuple, Optional

from config import (
    CLIENT_TRAIN_PATH,
    CLIENT_TEST_PATH,
    INVOICE_TRAIN_PATH,
    INVOICE_TEST_PATH,
    RANDOM_SEED,
    setup_logging,
)

warnings.filterwarnings("ignore")
logger = setup_logging()


# =============================================================================
# CLASSE 1 : CHARGEMENT DES DONNÉES
# =============================================================================

class DataLoader:
    """Charge et effectue le parsing initial des données STEG.

    Gère le chargement des deux tables (clients + factures),
    le parsing des dates dans leurs formats respectifs, et les
    premières vérifications de qualité.
    """

    def __init__(self):
        self.client_train: Optional[pd.DataFrame] = None
        self.client_test: Optional[pd.DataFrame] = None
        self.invoice_train: Optional[pd.DataFrame] = None
        self.invoice_test: Optional[pd.DataFrame] = None

    def load_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Charge l'ensemble des fichiers du dataset STEG.

        Returns
        -------
        tuple of pd.DataFrame
            (client_train, client_test, invoice_train, invoice_test)
        """
        logger.info("=" * 70)
        logger.info("CHARGEMENT DES DONNÉES STEG")
        logger.info("=" * 70)

        # --- Clients ---
        self.client_train = pd.read_csv(CLIENT_TRAIN_PATH)
        self.client_test = pd.read_csv(CLIENT_TEST_PATH)

        # --- Factures ---
        self.invoice_train = pd.read_csv(INVOICE_TRAIN_PATH, low_memory=False)
        self.invoice_test = pd.read_csv(INVOICE_TEST_PATH, low_memory=False)

        # --- Parsing des dates ---
        # creation_date au format DD/MM/YYYY dans la table clients
        self.client_train["creation_date"] = pd.to_datetime(
            self.client_train["creation_date"], format="%d/%m/%Y", errors="coerce"
        )
        self.client_test["creation_date"] = pd.to_datetime(
            self.client_test["creation_date"], format="%d/%m/%Y", errors="coerce"
        )

        # invoice_date au format YYYY-MM-DD dans la table factures
        self.invoice_train["invoice_date"] = pd.to_datetime(
            self.invoice_train["invoice_date"], errors="coerce"
        )
        self.invoice_test["invoice_date"] = pd.to_datetime(
            self.invoice_test["invoice_date"], errors="coerce"
        )

        # --- Rapport de chargement ---
        logger.info(f"  Clients   train : {self.client_train.shape}")
        logger.info(f"  Clients   test  : {self.client_test.shape}")
        logger.info(f"  Factures  train : {self.invoice_train.shape}")
        logger.info(f"  Factures  test  : {self.invoice_test.shape}")

        # Distribution de la cible
        if "target" in self.client_train.columns:
            target_dist = self.client_train["target"].value_counts(normalize=True)
            logger.info(f"  Distribution cible (train) :")
            logger.info(f"    Non-fraude (0) : {target_dist.get(0.0, 0):.2%}")
            logger.info(f"    Fraude     (1) : {target_dist.get(1.0, 0):.2%}")

        return (
            self.client_train,
            self.client_test,
            self.invoice_train,
            self.invoice_test,
        )


# =============================================================================
# CLASSE 2 : NETTOYAGE DES DONNÉES
# =============================================================================

class DataCleaner:
    """Nettoie et prépare les données brutes.

    Opérations :
    - Suppression des anomalies évidentes (consommation négative, index inversés)
    - Traitement des valeurs manquantes
    - Nettoyage des types de données
    """

    @staticmethod
    def clean_invoices(invoices: pd.DataFrame) -> pd.DataFrame:
        """Nettoie la table des factures.

        Parameters
        ----------
        invoices : pd.DataFrame
            Table des factures brute.

        Returns
        -------
        pd.DataFrame
            Factures nettoyées.
        """
        logger.info("Nettoyage des factures...")
        n_initial = len(invoices)

        df = invoices.copy()

        # 1) Consommation totale = somme des 4 niveaux
        conso_cols = [
            "consommation_level_1",
            "consommation_level_2",
            "consommation_level_3",
            "consommation_level_4",
        ]

        # Remplir les NaN des niveaux de consommation par 0
        for col in conso_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0)

        df["total_consommation"] = df[conso_cols].sum(axis=1)

        # 2) Nettoyage des colonnes numériques
        numeric_cols = ["old_index", "new_index", "months_number", "counter_coefficient"]
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # 3) Supprimer les anomalies évidentes
        # - Consommation négative
        mask_neg = df["total_consommation"] < 0
        # - Nombre de mois aberrant (<=0 ou >60)
        mask_months = (df["months_number"] <= 0) | (df["months_number"] > 60)
        # - old_index > new_index (compteur qui recule → anomalie matérielle)
        mask_index = df["old_index"] > df["new_index"]

        mask_anomaly = mask_neg | mask_months | mask_index
        df = df[~mask_anomaly].copy()

        n_removed = n_initial - len(df)
        logger.info(f"  Lignes supprimées (anomalies) : {n_removed:,} / {n_initial:,} "
                     f"({n_removed / n_initial:.2%})")

        # 4) Remplir les NaN restants
        df["months_number"] = df["months_number"].fillna(df["months_number"].median())
        df["counter_coefficient"] = df["counter_coefficient"].fillna(1)

        return df

    @staticmethod
    def clean_clients(clients: pd.DataFrame) -> pd.DataFrame:
        """Nettoie la table des clients.

        Parameters
        ----------
        clients : pd.DataFrame
            Table clients brute.

        Returns
        -------
        pd.DataFrame
            Clients nettoyés.
        """
        logger.info("Nettoyage des clients...")
        df = clients.copy()

        # Convertir les colonnes catégorielles en int si possible
        for col in ["disrict", "client_catg", "region"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")

        # Remplir les NaN catégoriels par le mode
        for col in ["disrict", "client_catg", "region"]:
            if df[col].isna().any():
                df[col] = df[col].fillna(df[col].mode()[0])

        return df


# =============================================================================
# CLASSE 3 : FEATURE ENGINEERING AVANCÉ
# =============================================================================

class FeatureEngineer:
    """Construit la matrice de features pour la détection de fraude.

    Extrait 30+ features organisées en 5 familles :
      1. Features Client (démographiques)
      2. Agrégations de consommation (statistiques descriptives)
      3. Features temporelles (saisonnalité, tendances)
      4. Indicateurs de fraude (baisses suspectes, anomalies)
      5. Features compteur (statuts, types, remarques)
    """

    # Mapping des mois vers les saisons tunisiennes
    SEASON_MAP = {12: "hiver", 1: "hiver", 2: "hiver",
                  3: "printemps", 4: "printemps", 5: "printemps",
                  6: "ete", 7: "ete", 8: "ete",
                  9: "automne", 10: "automne", 11: "automne"}

    def __init__(self, reference_date: str = "2019-01-01"):
        """
        Parameters
        ----------
        reference_date : str
            Date de référence pour calculer l'ancienneté des clients.
        """
        self.reference_date = pd.Timestamp(reference_date)

    # -----------------------------------------------------------------
    # 3.1 Features Client
    # -----------------------------------------------------------------

    def _build_client_features(self, clients: pd.DataFrame) -> pd.DataFrame:
        """Extrait les features à partir de la table clients.

        Features construites :
        - client_anciennete_jours : nombre de jours depuis la création du compte
        - creation_mois, creation_annee : composantes de la date de création
        - creation_saison_* : one-hot encoding de la saison de création
        """
        logger.info("  → Features client (ancienneté, catégorie, région)...")
        df = clients.copy()

        # Ancienneté en jours
        df["client_anciennete_jours"] = (
            (self.reference_date - df["creation_date"]).dt.days
        ).clip(lower=0)

        # Composantes de date de création
        df["creation_mois"] = df["creation_date"].dt.month.fillna(0).astype(int)
        df["creation_annee"] = df["creation_date"].dt.year.fillna(0).astype(int)

        # Saison de création (one-hot)
        df["creation_saison"] = df["creation_mois"].map(self.SEASON_MAP).fillna("inconnu")
        saison_dummies = pd.get_dummies(df["creation_saison"], prefix="saison_creation")
        df = pd.concat([df, saison_dummies], axis=1)
        df = df.drop(columns=["creation_saison", "creation_date"], errors="ignore")

        return df

    # -----------------------------------------------------------------
    # 3.2 Agrégations de consommation
    # -----------------------------------------------------------------

    def _build_consumption_aggregates(self, invoices: pd.DataFrame) -> pd.DataFrame:
        """Calcule les agrégats statistiques de consommation par client.

        Features construites :
        - conso_mean, conso_std, conso_min, conso_max, conso_sum
        - conso_median, conso_q25, conso_q75, conso_iqr
        - conso_coeff_variation : std / mean (indicateur de fraude clé)
        - conso_mensuelle_mean : consommation normalisée par mois
        - nb_factures : nombre total de factures
        """
        logger.info("  → Agrégats de consommation (mean, std, min, max, CV)...")

        agg = invoices.groupby("client_id")["total_consommation"].agg(
            conso_mean="mean",
            conso_std="std",
            conso_min="min",
            conso_max="max",
            conso_sum="sum",
            conso_median="median",
            nb_factures="count",
        ).reset_index()

        # Quantiles et IQR
        q25 = invoices.groupby("client_id")["total_consommation"].quantile(0.25).rename("conso_q25")
        q75 = invoices.groupby("client_id")["total_consommation"].quantile(0.75).rename("conso_q75")
        quantiles = pd.concat([q25, q75], axis=1).reset_index()

        agg = agg.merge(quantiles, on="client_id", how="left")
        agg["conso_iqr"] = agg["conso_q75"] - agg["conso_q25"]

        # Coefficient de variation (CV) — indicateur majeur de fraude
        # Un CV élevé signale des fluctuations anormales de consommation
        agg["conso_coeff_variation"] = np.where(
            agg["conso_mean"] > 0,
            agg["conso_std"] / agg["conso_mean"],
            0,
        )

        # Consommation mensuelle moyenne normalisée
        months_sum = invoices.groupby("client_id")["months_number"].sum().rename("total_months")
        agg = agg.merge(months_sum.reset_index(), on="client_id", how="left")
        agg["conso_mensuelle_mean"] = np.where(
            agg["total_months"] > 0,
            agg["conso_sum"] / agg["total_months"],
            0,
        )

        # Nettoyer NaN (std=NaN si une seule facture)
        agg["conso_std"] = agg["conso_std"].fillna(0)
        agg["conso_coeff_variation"] = agg["conso_coeff_variation"].fillna(0)

        return agg

    # -----------------------------------------------------------------
    # 3.3 Features temporelles et de tendance
    # -----------------------------------------------------------------

    def _build_temporal_features(self, invoices: pd.DataFrame) -> pd.DataFrame:
        """Extrait les features temporelles et de tendance par client.

        Features construites :
        - duree_historique_jours : étendue temporelle des factures
        - nb_mois_moyen_entre_factures : fréquence de facturation
        - conso_derniere, conso_premiere : première et dernière consommation
        - ratio_derniere_moyenne : dernière conso / moyenne (baisse suspecte ?)
        - tendance_conso : pente linéaire de la consommation (régression)
        - conso_saison_* : consommation moyenne par saison
        - saison_variation : max saisonnier / min saisonnier
        """
        logger.info("  → Features temporelles (tendances, saisonnalité)...")

        df = invoices.sort_values(["client_id", "invoice_date"]).copy()

        # ---- Durée de l'historique ----
        date_agg = df.groupby("client_id")["invoice_date"].agg(
            date_premiere_facture="min",
            date_derniere_facture="max",
        ).reset_index()
        date_agg["duree_historique_jours"] = (
            (date_agg["date_derniere_facture"] - date_agg["date_premiere_facture"]).dt.days
        ).clip(lower=0)
        date_agg = date_agg.drop(columns=["date_premiere_facture", "date_derniere_facture"])

        # ---- Fréquence de facturation ----
        freq_agg = df.groupby("client_id")["months_number"].mean().rename(
            "nb_mois_moyen_entre_factures"
        ).reset_index()

        # ---- Première et dernière consommation ----
        first_last = df.groupby("client_id")["total_consommation"].agg(
            conso_premiere="first",
            conso_derniere="last",
        ).reset_index()

        # ---- Ratio dernière / moyenne (indicateur de baisse soudaine) ----
        mean_conso = df.groupby("client_id")["total_consommation"].mean().rename("_mean_tmp")
        first_last = first_last.merge(mean_conso.reset_index(), on="client_id", how="left")
        first_last["ratio_derniere_moyenne"] = np.where(
            first_last["_mean_tmp"] > 0,
            first_last["conso_derniere"] / first_last["_mean_tmp"],
            1.0,
        )
        first_last = first_last.drop(columns=["_mean_tmp"])

        # ---- Tendance linéaire de consommation ----
        # Régression linéaire simple (rang temporel vs consommation)
        def compute_trend(group):
            if len(group) < 3:
                return 0.0
            y = group["total_consommation"].values
            x = np.arange(len(y), dtype=float)
            # Normalisation pour stabilité numérique
            x_centered = x - x.mean()
            y_centered = y - y.mean()
            denom = np.sum(x_centered ** 2)
            if denom == 0:
                return 0.0
            slope = np.sum(x_centered * y_centered) / denom
            return slope

        trend = df.groupby("client_id").apply(compute_trend, include_groups=False).rename(
            "tendance_conso"
        ).reset_index()

        # ---- Consommation par saison ----
        df["saison"] = df["invoice_date"].dt.month.map(self.SEASON_MAP).fillna("inconnu")
        saison_pivot = df.groupby(["client_id", "saison"])["total_consommation"].mean().unstack(
            fill_value=0
        )
        saison_pivot.columns = [f"conso_saison_{col}" for col in saison_pivot.columns]
        saison_pivot = saison_pivot.reset_index()

        # Variation saisonnière : max / min (hors zéro)
        saison_cols = [c for c in saison_pivot.columns if c.startswith("conso_saison_")]
        saison_vals = saison_pivot[saison_cols].values
        saison_min = np.where(saison_vals > 0, saison_vals, np.inf).min(axis=1)
        saison_max = saison_vals.max(axis=1)
        saison_pivot["saison_variation"] = np.where(
            (saison_min > 0) & (saison_min != np.inf),
            saison_max / saison_min,
            1.0,
        )

        # ---- Fusion ----
        result = date_agg
        for sub_df in [freq_agg, first_last, trend, saison_pivot]:
            result = result.merge(sub_df, on="client_id", how="left")

        return result

    # -----------------------------------------------------------------
    # 3.4 Indicateurs de fraude
    # -----------------------------------------------------------------

    def _build_fraud_indicators(self, invoices: pd.DataFrame) -> pd.DataFrame:
        """Construit des indicateurs spécifiques à la détection de fraude.

        Features construites :
        - max_drop_ratio : plus forte baisse relative entre deux factures consécutives
        - nb_baisses_suspectes : nombre de baisses > 50% entre factures
        - ratio_zero_conso : proportion de factures à consommation nulle
        - conso_level_dominant : niveau de consommation le plus utilisé
        - ratio_level1_total : part du level_1 dans la consommation totale
        """
        logger.info("  → Indicateurs de fraude (baisses suspectes, zéros)...")

        df = invoices.sort_values(["client_id", "invoice_date"]).copy()

        # ---- Baisses de consommation entre factures consécutives ----
        df["prev_conso"] = df.groupby("client_id")["total_consommation"].shift(1)
        df["drop_ratio"] = np.where(
            df["prev_conso"] > 0,
            (df["prev_conso"] - df["total_consommation"]) / df["prev_conso"],
            0,
        )

        drop_agg = df.groupby("client_id")["drop_ratio"].agg(
            max_drop_ratio="max",
        ).reset_index()

        # Nombre de baisses > 50% (suspectes)
        df["is_baisse_suspecte"] = (df["drop_ratio"] > 0.5).astype(int)
        baisses = df.groupby("client_id")["is_baisse_suspecte"].sum().rename(
            "nb_baisses_suspectes"
        ).reset_index()

        # ---- Proportion de factures à consommation nulle ----
        df["is_zero_conso"] = (df["total_consommation"] == 0).astype(int)
        zero_ratio = df.groupby("client_id").agg(
            ratio_zero_conso=("is_zero_conso", "mean"),
        ).reset_index()

        # ---- Répartition des niveaux de consommation ----
        level_cols = [
            "consommation_level_1",
            "consommation_level_2",
            "consommation_level_3",
            "consommation_level_4",
        ]

        level_means = df.groupby("client_id")[level_cols].mean()
        level_means.columns = [f"mean_{c}" for c in level_cols]

        # Niveau dominant (lequel contribue le plus)
        level_means["conso_level_dominant"] = level_means.values.argmax(axis=1)

        # Ratio level_1 / total (les fraudeurs restent souvent en level_1)
        total = level_means.sum(axis=1)
        level_means["ratio_level1_total"] = np.where(
            total > 0,
            level_means["mean_consommation_level_1"] / total,
            0,
        )
        level_features = level_means[["conso_level_dominant", "ratio_level1_total"]].reset_index()

        # ---- Fusion ----
        result = drop_agg
        for sub_df in [baisses, zero_ratio, level_features]:
            result = result.merge(sub_df, on="client_id", how="left")

        return result

    # -----------------------------------------------------------------
    # 3.5 Features compteur
    # -----------------------------------------------------------------

    def _build_counter_features(self, invoices: pd.DataFrame) -> pd.DataFrame:
        """Extrait les features liées aux compteurs.

        Features construites :
        - nb_compteurs_uniques : nombre de compteurs distincts utilisés
        - nb_types_compteur : nombre de types (ELEC, GAZ) distincts
        - has_counter_anomaly : si le statut compteur > 0 (anormal)
        - nb_remarques_lecture : diversité des remarques de lecture
        - counter_code_mode : code compteur le plus fréquent
        - mean_counter_coefficient : coefficient compteur moyen
        """
        logger.info("  → Features compteur (statuts, anomalies, types)...")

        agg = invoices.groupby("client_id").agg(
            nb_compteurs_uniques=("counter_number", "nunique"),
            nb_types_compteur=("counter_type", "nunique"),
            nb_remarques_lecture=("reading_remarque", "nunique"),
            mean_counter_coefficient=("counter_coefficient", "mean"),
        ).reset_index()

        # Présence d'un statut compteur anormal (> 0)
        # Convertir en numérique car le champ peut contenir des strings
        counter_statue_numeric = pd.to_numeric(invoices["counter_statue"], errors="coerce").fillna(0)
        invoices_tmp = invoices.assign(counter_statue_num=counter_statue_numeric)
        anomaly = invoices_tmp.groupby("client_id")["counter_statue_num"].apply(
            lambda x: int((x > 0).any())
        ).rename("has_counter_anomaly").reset_index()

        # Code compteur le plus fréquent
        counter_code_numeric = pd.to_numeric(invoices["counter_code"], errors="coerce").fillna(0)
        invoices_tmp2 = invoices.assign(counter_code_num=counter_code_numeric)
        counter_code_mode = invoices_tmp2.groupby("client_id")["counter_code_num"].agg(
            lambda x: x.mode().iloc[0] if len(x.mode()) > 0 else 0
        ).rename("counter_code_mode").reset_index()

        result = agg.merge(anomaly, on="client_id", how="left")
        result = result.merge(counter_code_mode, on="client_id", how="left")

        return result

    # -----------------------------------------------------------------
    # 3.6 Fusion complète
    # -----------------------------------------------------------------

    def build_features(
        self,
        clients: pd.DataFrame,
        invoices: pd.DataFrame,
    ) -> pd.DataFrame:
        """Construit la matrice complète de features pour un jeu de données.

        Orchestre l'ensemble des étapes de feature engineering et fusionne
        les résultats en une seule matrice client × features.

        Parameters
        ----------
        clients : pd.DataFrame
            Table clients (avec ou sans colonne 'target').
        invoices : pd.DataFrame
            Table factures nettoyée.

        Returns
        -------
        pd.DataFrame
            Matrice de features fusionnée, indexée par client_id.
        """
        logger.info("-" * 60)
        logger.info("CONSTRUCTION DES FEATURES")
        logger.info("-" * 60)

        # Nettoyer les données
        cleaner = DataCleaner()
        invoices_clean = cleaner.clean_invoices(invoices)
        clients_clean = cleaner.clean_clients(clients)

        # Construire chaque famille de features
        client_feats = self._build_client_features(clients_clean)
        conso_agg = self._build_consumption_aggregates(invoices_clean)
        temporal_feats = self._build_temporal_features(invoices_clean)
        fraud_indicators = self._build_fraud_indicators(invoices_clean)
        counter_feats = self._build_counter_features(invoices_clean)

        # Fusion progressive sur client_id
        result = client_feats
        for feats in [conso_agg, temporal_feats, fraud_indicators, counter_feats]:
            result = result.merge(feats, on="client_id", how="left")

        # Remplir les NaN restants par 0 (clients sans factures par ex.)
        feature_cols = [c for c in result.columns if c not in ["client_id", "target"]]
        result[feature_cols] = result[feature_cols].fillna(0)

        # Remplacer les infinis
        result = result.replace([np.inf, -np.inf], 0)

        logger.info(f"  Matrice finale : {result.shape[0]} clients × {len(feature_cols)} features")
        logger.info(f"  Features : {feature_cols[:10]}... (+ {max(0, len(feature_cols)-10)} autres)")

        return result


# =============================================================================
# FONCTION UTILITAIRE D'ACCÈS RAPIDE
# =============================================================================

def prepare_training_data() -> Tuple[pd.DataFrame, pd.Series, list]:
    """Fonction de haut niveau : charge, nettoie, et construit les features.

    Returns
    -------
    X : pd.DataFrame
        Matrice de features (sans client_id ni target).
    y : pd.Series
        Variable cible binaire.
    feature_names : list
        Liste des noms de features.
    """
    # Charger les données
    loader = DataLoader()
    client_train, _, invoice_train, _ = loader.load_all()

    # Construire les features
    engineer = FeatureEngineer()
    df = engineer.build_features(client_train, invoice_train)

    # Séparer X et y
    target_col = "target"
    id_col = "client_id"

    y = df[target_col].astype(int)
    X = df.drop(columns=[target_col, id_col], errors="ignore")

    # S'assurer que toutes les colonnes sont numériques
    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce").fillna(0)

    feature_names = X.columns.tolist()

    logger.info(f"  X shape : {X.shape}")
    logger.info(f"  y distribution : {dict(y.value_counts())}")

    return X, y, feature_names
