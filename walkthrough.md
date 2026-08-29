# Walkthrough — Détection de Fraude Énergétique STEG

## Pipeline exécuté avec succès ✅

Le pipeline complet a été exécuté en **7.5 minutes** sur vos données STEG (135K clients, 4.47M factures).

---

## Résultats Clés

### Meilleur Modèle : LightGBM

| Métrique | Score |
|----------|-------|
| **ROC-AUC** | **0.8548** |
| **PR-AUC** | **0.3145** |
| **F1-Score** | **0.3060** |
| **Accuracy** | **0.8187** |
| **Recall Fraude** | **72%** |

> [!IMPORTANT]
> Le modèle détecte **72% des fraudes** avec une ROC-AUC de 0.85, ce qui est un résultat solide pour un PoC sur des données réelles fortement déséquilibrées (5.58% de fraudes).

### Comparaison des Modèles (CV 5-Fold)

| Modèle | ROC-AUC | PR-AUC | F1-Score |
|--------|---------|--------|----------|
| XGBoost | 0.8441 ± 0.0049 | 0.2971 ± 0.0091 | 0.3200 ± 0.0073 |
| **LightGBM** ★ | **0.8463 ± 0.0048** | **0.3001 ± 0.0091** | 0.3015 ± 0.0033 |
| CatBoost | 0.8487 ± 0.0048 | 0.2966 ± 0.0062 | 0.2867 ± 0.0034 |

### Rapport de Classification (Test Set)

```
              precision    recall  f1-score   support

    Légitime       0.98      0.82      0.90     25586
      Fraude       0.19      0.72      0.31      1513

    accuracy                           0.82     27099
   macro avg       0.59      0.77      0.60     27099
weighted avg       0.94      0.82      0.86     27099
```

---

## Graphiques Générés

### Matrice de Confusion
![Matrice de Confusion](C:/Users/itomo/.gemini/antigravity/brain/23b22d1b-ada0-4ba2-870d-5ffdb29ffb11/confusion_matrix.png)

### Courbe ROC-AUC
![Courbe ROC](C:/Users/itomo/.gemini/antigravity/brain/23b22d1b-ada0-4ba2-870d-5ffdb29ffb11/roc_curve.png)

### Courbe Precision-Recall
![Courbe PR](C:/Users/itomo/.gemini/antigravity/brain/23b22d1b-ada0-4ba2-870d-5ffdb29ffb11/precision_recall_curve.png)

### Feature Importance (Top 20)
![Feature Importance](C:/Users/itomo/.gemini/antigravity/brain/23b22d1b-ada0-4ba2-870d-5ffdb29ffb11/feature_importance.png)

---

## Structure du Projet

```
detection_de_fraude/
├── config.py                      # Configuration centralisée
├── preprocessing.py               # Prétraitement & Feature Engineering (30+ features)
├── imbalance.py                   # Gestion du déséquilibre (SMOTE-Tomek + scale_pos_weight)
├── training.py                    # XGBoost + LightGBM + CatBoost, CV 5-Fold
├── evaluation.py                  # Confusion, ROC, PR, Feature Importance
├── main.py                        # Orchestrateur principal (7 étapes)
├── fraud_detection_notebook.ipynb  # Notebook Jupyter interactif
├── requirements.txt               # Dépendances
├── client_train.csv               # Données clients (135K)
├── invoice_train.csv              # Factures (4.47M)
└── outputs/
    ├── plots/
    │   ├── confusion_matrix.png
    │   ├── roc_curve.png
    │   ├── precision_recall_curve.png
    │   └── feature_importance.png
    ├── models/
    │   └── best_model_lightgbm.joblib
    └── pipeline.log
```

## Données Traitées

- **135,493 clients** — 94.42% légitimes, 5.58% fraude
- **4,476,749 factures** — 3,709 anomalies supprimées (0.08%)
- **30+ features** extraites en 5 familles (client, consommation, temporel, fraude, compteur)
