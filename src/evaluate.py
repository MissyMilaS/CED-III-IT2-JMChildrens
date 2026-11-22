"""Métricas para evaluar y comparar modelos."""
import json
from pathlib import Path

from sklearn.metrics import (accuracy_score, precision_score, recall_score,f1_score, cohen_kappa_score, confusion_matrix)

METRICS_DIR = Path(__file__).resolve().parents[1] / "outputs" / "metrics"


def compute_metrics(y_true, y_pred):
    """Devuelve las 5 métricas del enunciado (clase positiva = 1)."""
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "kappa": cohen_kappa_score(y_true, y_pred),
    }


def save_metrics(name, metrics, y_true=None, y_pred=None):
    """Guarda outputs/metrics/<name>.json (mismo formato para todos los modelos)."""
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    out = {"model": name, **{k: round(float(v), 4) for k, v in metrics.items()}}
    if y_true is not None:
        out["confusion_matrix"] = confusion_matrix(y_true, y_pred).tolist()
    (METRICS_DIR / f"{name}.json").write_text(json.dumps(out, indent=2))
    return out