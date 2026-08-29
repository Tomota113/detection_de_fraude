"""
=============================================================================
 Routes API — App Web de Détection de Fraude STEG
=============================================================================
 Endpoints REST pour le frontend :
   - /api/stats          → KPI globaux
   - /api/fraud-by-region → Taux de fraude par région
   - /api/score/{id}     → Scoring individuel
   - /api/distribution   → Distributions par feature
   - /api/correlation    → Corrélations avec la cible
   - /api/feature-importance → Importance des variables
   - /api/model/*        → Courbes ROC, PR, confusion, CV
   - /api/suspects       → Liste des suspects
=============================================================================
"""

from fastapi import APIRouter, Query, Response
from fastapi.responses import StreamingResponse
from typing import Optional
import csv
import io

from webapp.engine import engine

router = APIRouter(prefix="/api")


# =============================================================================
# DONNÉES GLOBALES
# =============================================================================

@router.get("/stats")
def get_stats():
    """KPI globaux du système."""
    return engine.get_stats()


@router.get("/fraud-by-region")
def get_fraud_by_region():
    """Taux de fraude par region."""
    return engine.get_fraud_by_region()


@router.get("/timeline")
def get_timeline():
    """Evolution temporelle de la fraude par annee."""
    return engine.get_timeline()


@router.get("/regions")
def get_regions():
    """Liste des régions disponibles."""
    return engine.get_regions()


@router.get("/features")
def get_features():
    """Liste des features disponibles pour l'exploration."""
    return engine.get_available_features()


# =============================================================================
# SCORING CLIENT
# =============================================================================

@router.get("/score/{client_id}")
def score_client(client_id: str):
    """Score un client individuel."""
    result = engine.score_client(client_id)
    if result is None:
        return {"error": f"Client '{client_id}' non trouvé"}
    return result


@router.get("/search-clients")
def search_clients(q: str = Query("", min_length=1)):
    """Recherche de clients par préfixe (autocomplétion)."""
    return engine.search_clients(q)


# =============================================================================
# EXPLORATION
# =============================================================================

@router.get("/distribution/{feature}")
def get_distribution(feature: str):
    """Distribution d'une feature par classe."""
    result = engine.get_distribution(feature)
    if result is None:
        return {"error": f"Feature '{feature}' non trouvée"}
    return result


@router.get("/correlation")
def get_correlations():
    """Top 15 corrélations avec la cible."""
    return engine.get_correlations()


@router.get("/feature-importance")
def get_feature_importance(n_top: int = Query(20, ge=5, le=40)):
    """Top N features par importance."""
    return engine.get_feature_importance(n_top)


# =============================================================================
# PERFORMANCE DU MODÈLE
# =============================================================================

@router.get("/model/roc")
def get_roc():
    """Données de la courbe ROC."""
    return engine.get_roc_data()


@router.get("/model/pr")
def get_pr():
    """Données de la courbe Precision-Recall."""
    return engine.get_pr_data()


@router.get("/model/confusion")
def get_confusion(threshold: float = Query(0.5, ge=0.0, le=1.0)):
    """Matrice de confusion pour un seuil donné."""
    return engine.get_confusion(threshold)


@router.get("/model/cv-results")
def get_cv_results():
    """Résultats de la validation croisée."""
    return engine.get_cv_results()


# =============================================================================
# CLIENTS SUSPECTS
# =============================================================================

@router.get("/suspects")
def get_suspects(
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    region: Optional[int] = Query(None),
    limit: int = Query(50, ge=1, le=500),
    offset: int = Query(0, ge=0),
):
    """Liste des clients suspects triés par probabilité."""
    return engine.get_suspects(
        threshold=threshold,
        region=region,
        limit=limit,
        offset=offset,
    )


@router.get("/suspects/export")
def export_suspects(
    threshold: float = Query(0.5, ge=0.0, le=1.0),
    region: Optional[int] = Query(None),
):
    """Exporte la liste des suspects au format CSV."""
    data = engine.get_suspects(
        threshold=threshold,
        region=region,
        limit=10000,
        offset=0,
    )

    output = io.StringIO()
    writer = csv.DictWriter(output, fieldnames=[
        "client_id", "proba", "region", "category", "top_factor", "actual",
    ])
    writer.writeheader()
    for suspect in data["suspects"]:
        writer.writerow(suspect)

    output.seek(0)
    return StreamingResponse(
        io.BytesIO(output.getvalue().encode("utf-8")),
        media_type="text/csv",
        headers={
            "Content-Disposition": f"attachment; filename=suspects_seuil_{threshold}.csv",
        },
    )
