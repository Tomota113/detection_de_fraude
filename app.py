"""
=============================================================================
 App Web — Détection de Fraude Énergétique STEG
=============================================================================
 Point d'entrée : lance le serveur FastAPI avec le moteur de scoring.
 
 Usage :
   python app.py
   → Ouvre http://localhost:8000 dans le navigateur
=============================================================================
"""

import sys
from pathlib import Path

# Ajouter la racine du projet au PYTHONPATH
sys.path.insert(0, str(Path(__file__).resolve().parent))

from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.requests import Request
import uvicorn

from webapp.api import router as api_router
from webapp.engine import engine

# =============================================================================
# Creation de l'app FastAPI
# =============================================================================

app = FastAPI(
    title="STEG Fraud Detection",
    description="Systeme de Detection de Fraude Energetique - STEG Tunisie",
    version="1.0.0",
)

# Fichiers statiques (CSS, JS, images)
static_dir = Path(__file__).parent / "webapp" / "static"
static_dir.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

# Routes API
app.include_router(api_router)

# Chemin vers le template HTML
HTML_FILE = Path(__file__).parent / "webapp" / "templates" / "index.html"


# =============================================================================
# Page principale (SPA)
# =============================================================================

@app.get("/")
async def index():
    """Page d'accueil - Single Page Application."""
    return FileResponse(str(HTML_FILE), media_type="text/html")


# =============================================================================
# Événement de démarrage
# =============================================================================

@app.on_event("startup")
async def startup_event():
    """Charge le modèle et les données au démarrage du serveur."""
    print("\n" + "=" * 60)
    print("  STEG Fraud Detection - Demarrage du serveur")
    print("=" * 60)
    engine.load()
    print("=" * 60)
    print("  [OK] Serveur pret -> http://localhost:8000")
    print("  [API] Documentation -> http://localhost:8000/docs")
    print("=" * 60 + "\n")


# =============================================================================
# Lancement
# =============================================================================

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="0.0.0.0",
        port=8000,
        reload=False,
    )
