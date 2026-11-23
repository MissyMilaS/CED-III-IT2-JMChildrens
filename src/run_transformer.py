"""
Pipeline ejecutable autónomo y modular para DistilBERT sobre GoEmotions (CED III).
Permite entrenar, evaluar en test, predecir frases individuales y actualizar métricas oficiales.
"""

import os
import sys
import time
import json
import random
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

import torch
from datasets import Dataset
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    TrainingArguments,
    Trainer,
    DataCollatorWithPadding
)
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    cohen_kappa_score,
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

ROOT_DIR = Path(__file__).resolve().parents[1]
if not (ROOT_DIR / "data").exists():
    ROOT_DIR = Path.cwd().resolve() if (Path.cwd() / "data").exists() else Path.cwd() / "CED-III-IT2-JMChildrens"

DATA_DIR = ROOT_DIR / "data" / "processed"
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"
FIG_DIR = ROOT_DIR / "outputs" / "figures"
SAVED_MODELS_DIR = ROOT_DIR / "outputs" / "saved_models"

FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)
SAVED_MODELS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
ID2LABEL = {0: "negativo", 1: "positivo"}
LABEL2ID = {"negativo": 0, "positivo": 1}


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_processed_data(data_dir=DATA_DIR):
    train_path = data_dir / "train.csv"
    test_path = data_dir / "test.csv"
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"No se encontraron los datos procesados en {data_dir}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    return train_df, test_df


def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision_score(labels, preds, zero_division=0)),
        "recall": float(recall_score(labels, preds, zero_division=0)),
        "f1": float(f1_score(labels, preds, zero_division=0)),
        "kappa": float(cohen_kappa_score(labels, preds))
    }


def predict_text(model, tokenizer, texts, device, max_len=64):
    model.eval()
    if isinstance(texts, str):
        texts = [texts]
    encoded = tokenizer(texts, padding=True, truncation=True, max_length=max_len, return_tensors="pt").to(device)
    with torch.no_grad():
        logits = model(**encoded).logits
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
    
    results = []
    for t, p in zip(texts, probs):
        pred_idx = int(np.argmax(p))
        results.append({
            "text": t,
            "pred_label": ID2LABEL[pred_idx],
            "p_neg": round(float(p[0]), 4),
            "p_pos": round(float(p[1]), 4)
        })
    return pd.DataFrame(results)


def update_comparison_csv_and_plot(metrics_dir=METRICS_DIR, fig_dir=FIG_DIR):
    all_models = ["baseline_dummy", "dense", "vanilla_rnn", "lstm", "bilstm", "distilbert"]
    rows = []
    for m in all_models:
        fp = metrics_dir / f"{m}.json"
        if fp.exists():
            d = json.loads(fp.read_text(encoding="utf-8"))
            std = d.get("std", {})
            r = {"model": d["model"]}
            for k in ["accuracy", "precision", "recall", "f1", "kappa"]:
                val = d.get(k, 0.0)
                s = std.get(k)
                r[k] = f"{val:.3f}" + (f" ± {s:.3f}" if s is not None else "")
            r["params"] = f"{d.get('n_params', '-'):,}" if isinstance(d.get('n_params'), (int, float)) else d.get('n_params', '-')
            r["sec/run"] = d.get("train_seconds_per_run", "-")
            rows.append(r)

    df_comp = pd.DataFrame(rows)
    df_comp.to_csv(metrics_dir / "comparison.csv", index=False)
    print(f"[*] Actualizado: {metrics_dir / 'comparison.csv'}")

    # Plot
    mets = ["accuracy", "precision", "recall", "f1", "kappa"]
    models_avail = [m for m in all_models if (metrics_dir / f"{m}.json").exists()]
    if not models_avail:
        return df_comp
    data_models = {n: json.loads((metrics_dir / f"{n}.json").read_text(encoding="utf-8")) for n in models_avail}
    model_labels = {
        "baseline_dummy": "Baseline Dummy",
        "dense": "Dense NN",
        "vanilla_rnn": "Vanilla RNN",
        "lstm": "LSTM",
        "bilstm": "BiLSTM",
        "distilbert": "DistilBERT (Task 4)"
    }

    x = np.arange(len(mets))
    w = 0.82 / len(models_avail)
    colores = ["#95a5a6", "#3498db", "#e67e22", "#9b59b6", "#1abc9c", "#e74c3c"]

    fig, ax = plt.subplots(figsize=(11, 4.8), dpi=200)
    for i, n in enumerate(models_avail):
        vals = [data_models[n].get(m, 0.0) for m in mets]
        errs = [data_models[n].get("std", {}).get(m, 0.0) for m in mets]
        offset = (i - (len(models_avail) - 1) / 2) * w
        ax.bar(x + offset, vals, w, label=model_labels.get(n, n),
               color=colores[i % len(colores)], yerr=errs, capsize=2.5,
               edgecolor="black", linewidth=0.6, alpha=0.92)

    ax.set_xticks(x)
    ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score", "Cohen's Kappa (κ)"], fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score", fontsize=11)
    ax.set_title("Model Comparison across Evaluation Metrics (GoEmotions Test Set, n=200)", fontsize=12, fontweight="bold", pad=12)
    ax.grid(axis="y", linestyle="--", alpha=0.5)
    ax.legend(bbox_to_anchor=(1.01, 1), loc="upper left", frameon=True, fontsize=10)
    plt.tight_layout()
    plt.savefig(fig_dir / "all_models_metrics_comparison.png", dpi=200)
    plt.close()
    print(f"[*] Guardada: {fig_dir / 'all_models_metrics_comparison.png'}")
    return df_comp


def run_pipeline(epochs=3, lr=3e-5, batch_size=16, max_len=64, model_name="distilbert-base-uncased", save_weights=True):
    set_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[*] Dispositivo de cómputo: {device}")

    train_df, test_df = load_processed_data()
    print(f"[*] Datos cargados: {len(train_df)} train, {len(test_df)} test.")

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    ds_train = Dataset.from_pandas(train_df[["text", "label"]])
    ds_test = Dataset.from_pandas(test_df[["text", "label"]])

    def tok_fn(batch):
        return tokenizer(batch["text"], truncation=True, max_length=max_len)

    ds_train = ds_train.map(tok_fn, batched=True)
    ds_test = ds_test.map(tok_fn, batched=True)

    model = AutoModelForSequenceClassification.from_pretrained(
        model_name,
        num_labels=2,
        id2label=ID2LABEL,
        label2id=LABEL2ID
    ).to(device)

    checkpoints_dir = ROOT_DIR / "outputs" / "distilbert_checkpoints"
    training_args = TrainingArguments(
        output_dir=str(checkpoints_dir),
        learning_rate=lr,
        per_device_train_batch_size=batch_size,
        per_device_eval_batch_size=32,
        num_train_epochs=epochs,
        weight_decay=0.01,
        eval_strategy="epoch",
        save_strategy="no",
        seed=SEED,
        logging_steps=25,
        report_to="none"
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=ds_train,
        eval_dataset=ds_test,
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer),
        compute_metrics=compute_metrics
    )

    print("[*] Iniciando entrenamiento de DistilBERT...")
    start_time = time.time()
    trainer.train()
    total_time = round(time.time() - start_time, 1)
    print(f"[*] Entrenamiento completado en {total_time}s.")

    if save_weights:
        weights_path = SAVED_MODELS_DIR / "distilbert.pt"
        torch.save(model.state_dict(), weights_path)
        print(f"[*] Pesos guardados en: {weights_path}")

    # Eval
    preds = trainer.predict(ds_test)
    y_true = preds.label_ids
    y_pred = np.argmax(preds.predictions, axis=-1)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    kappa = cohen_kappa_score(y_true, y_pred)
    cm = confusion_matrix(y_true, y_pred)

    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    metrics_export = {
        "model": "distilbert",
        "accuracy": round(float(acc), 4),
        "precision": round(float(prec), 4),
        "recall": round(float(rec), 4),
        "f1": round(float(f1), 4),
        "kappa": round(float(kappa), 4),
        "n_params": int(n_params),
        "train_seconds_per_run": total_time,
        "confusion_matrix": cm.tolist()
    }
    (METRICS_DIR / "distilbert.json").write_text(json.dumps(metrics_export, indent=2), encoding="utf-8")
    print(f"[*] Guardado: {METRICS_DIR / 'distilbert.json'}")

    # Plot CM
    fig, ax = plt.subplots(figsize=(4.5, 3.8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negativo", "Positivo"])
    disp.plot(cmap="Blues", values_format="d", ax=ax)
    ax.set_title(f"Matriz de Confusión - DistilBERT (Acc: {acc:.3f})")
    plt.tight_layout()
    plt.savefig(FIG_DIR / "distilbert_confusion_matrix.png", dpi=200)
    plt.close()
    print(f"[*] Guardada: {FIG_DIR / 'distilbert_confusion_matrix.png'}")

    update_comparison_csv_and_plot()
    return metrics_export


def main():
    parser = argparse.ArgumentParser(description="Pipeline modular DistilBERT para GoEmotions (CED III)")
    parser.add_argument("--train", action="store_true", help="Ejecutar entrenamiento completo de DistilBERT")
    parser.add_argument("--epochs", type=int, default=3, help="Número de épocas (def: 3)")
    parser.add_argument("--lr", type=float, default=3e-5, help="Tasa de aprendizaje (def: 3e-5)")
    parser.add_argument("--batch_size", type=int, default=16, help="Tamaño de lote (def: 16)")
    parser.add_argument("--max_len", type=int, default=64, help="Longitud máxima de tokens (def: 64)")
    parser.add_argument("--update_plots", action="store_true", help="Actualizar comparison.csv y figuras globales")
    parser.add_argument("--predict", type=str, default=None, help="Texto individual para predecir sentimiento")
    args = parser.parse_args()

    if args.predict:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        model = AutoModelForSequenceClassification.from_pretrained("distilbert-base-uncased", num_labels=2).to(device)
        df_res = predict_text(model, tokenizer, args.predict, device)
        print("\nResultado de predicción:")
        print(df_res.to_string(index=False))
    elif args.train:
        run_pipeline(epochs=args.epochs, lr=args.lr, batch_size=args.batch_size, max_len=args.max_len)
    elif args.update_plots:
        update_comparison_csv_and_plot()
    else:
        print("Use --train para entrenar, --update_plots para regenerar gráficos, o --predict 'texto' para inferencia.")


if __name__ == "__main__":
    main()
