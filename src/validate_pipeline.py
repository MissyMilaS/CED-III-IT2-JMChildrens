"""
Script de validación integral del pipeline (CED III - Tarea Integradora 2).
Verifica la integridad de datos, métricas consolidadas, figuras, notebooks y módulos src.
"""

import sys
import json
from pathlib import Path
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "processed"
METRICS_DIR = REPO_ROOT / "outputs" / "metrics"
FIG_DIR = REPO_ROOT / "outputs" / "figures"
NOTEBOOKS_DIR = REPO_ROOT / "notebooks"
SRC_DIR = REPO_ROOT / "src"

sys.path.insert(0, str(REPO_ROOT))


def check_data():
    print("[1/5] Verificando datos procesados...")
    train_path = DATA_DIR / "train.csv"
    test_path = DATA_DIR / "test.csv"
    assert train_path.exists(), "Falta train.csv"
    assert test_path.exists(), "Falta test.csv"

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    assert len(train_df) == 800, f"train.csv esperado 800, obtenido {len(train_df)}"
    assert len(test_df) == 200, f"test.csv esperado 200, obtenido {len(test_df)}"
    assert (train_df["label"].value_counts().to_dict() == {0: 400, 1: 400}), "train.csv desbalanceado"
    assert (test_df["label"].value_counts().to_dict() == {0: 100, 1: 100}), "test.csv desbalanceado"
    print("   ✔ train.csv (800) y test.csv (200) correctamente balanceados (50/50).")


def check_metrics():
    print("[2/5] Verificando archivos de métricas y tabla consolidada...")
    models = ["baseline_dummy", "dense", "vanilla_rnn", "lstm", "bilstm", "distilbert"]
    for m in models:
        fp = METRICS_DIR / f"{m}.json"
        assert fp.exists(), f"Falta métrica {fp}"
        data = json.loads(fp.read_text(encoding="utf-8"))
        for k in ["accuracy", "precision", "recall", "f1", "kappa"]:
            assert k in data, f"Falta clave '{k}' en {m}.json"
        assert "confusion_matrix" in data, f"Falta 'confusion_matrix' en {m}.json"

    comp_path = METRICS_DIR / "comparison.csv"
    assert comp_path.exists(), "Falta comparison.csv"
    df_comp = pd.read_csv(comp_path)
    assert len(df_comp) == 6, f"comparison.csv debe tener 6 modelos, tiene {len(df_comp)}"
    assert set(df_comp["model"]) == set(models), "Discrepancia en modelos de comparison.csv"
    print(f"   ✔ Todos los 6 modelos ({', '.join(models)}) validados en métricas JSON y comparison.csv.")


def check_figures():
    print("[3/5] Verificando figuras generadas...")
    required_figures = [
        "all_models_metrics_comparison.png",
        "distilbert_confusion_matrix.png",
        "eda_class_balance.png",
        "eda_length.png",
        "eda_top_words.png",
        "nn_confusion_matrices.png",
        "nn_metrics_comparison.png",
        "nn_training_curves.png"
    ]
    for fig in required_figures:
        p = FIG_DIR / fig
        assert p.exists(), f"Falta figura {fig}"
        assert p.stat().st_size > 1000, f"Figura {fig} parece vacía o corrupta"
    print(f"   ✔ {len(required_figures)} figuras requeridas verificadas con éxito.")


def check_notebooks():
    print("[4/5] Verificando cuadernos Jupyter...")
    notebooks = [
        "01_eda.ipynb",
        "02_baseline_model.ipynb",
        "03_dense_rnn_lstm.ipynb",
        "04_transformer_extension.ipynb"
    ]
    for nb_name in notebooks:
        p = NOTEBOOKS_DIR / nb_name
        assert p.exists(), f"Falta cuaderno {nb_name}"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert "cells" in data and len(data["cells"]) > 0, f"Cuaderno {nb_name} vacío"
        with_outputs = sum(1 for c in data["cells"] if len(c.get("outputs", [])) > 0)
        assert with_outputs > 0, f"Cuaderno {nb_name} no tiene salidas guardadas"
    print(f"   ✔ Los 4 cuadernos verificados y con salidas de ejecución persistidas.")


def check_modules():
    print("[5/5] Verificando módulos de src/...")
    from src import preprocessing
    from src import models
    from src import evaluate
    from src import train
    from src import run_transformer

    assert hasattr(models, "load_transformer_model"), "models.py debe tener load_transformer_model"
    assert hasattr(models, "get_transformer_tokenizer"), "models.py debe tener get_transformer_tokenizer"
    assert hasattr(evaluate, "compute_transformer_metrics"), "evaluate.py debe tener compute_transformer_metrics"
    print("   ✔ Módulos de src/ cargados e interfaces validadas exitosamente.")


def main():
    print("==================================================================")
    print("  SUITE DE VERIFICACIÓN Y CONTROL DE CALIDAD - TAREA INTEGRADORA 2")
    print("==================================================================")
    try:
        check_data()
        check_metrics()
        check_figures()
        check_notebooks()
        check_modules()
        print("\n🎉 ¡TODOS LOS COMPONENTES TÉCNICOS PASARON LAS PRUEBAS CON ÉXITO! (5/5)")
        print("El proyecto está en estado óptimo, reproducible y sin inconsistencias.\n")
    except AssertionError as e:
        print(f"\n❌ ERROR DE VALIDACIÓN: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
