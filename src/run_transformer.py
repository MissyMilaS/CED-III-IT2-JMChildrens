"""
Script ejecutable autónomo para entrenar y evaluar DistilBERT sobre GoEmotions,
generar métricas oficiales, actualizar comparison.csv y generar gráficos.
"""

import os
import sys
import time
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

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
DATA_DIR = ROOT_DIR / "data" / "processed"
METRICS_DIR = ROOT_DIR / "outputs" / "metrics"
FIG_DIR = ROOT_DIR / "outputs" / "figures"

FIG_DIR.mkdir(parents=True, exist_ok=True)
METRICS_DIR.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Dispositivo: {device}")

# 1. Carga de datos
train_df = pd.read_csv(DATA_DIR / "train.csv")
test_df = pd.read_csv(DATA_DIR / "test.csv")
print(f"[*] Datos cargados: {len(train_df)} train, {len(test_df)} test.")

# 2. Tokenizador
MODEL_NAME = "distilbert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
MAX_LEN = 64

def tokenize_batch(batch):
    return tokenizer(batch["text"], truncation=True, max_length=MAX_LEN)

ds_train = Dataset.from_pandas(train_df[["text", "label"]]).map(tokenize_batch, batched=True)
ds_test = Dataset.from_pandas(test_df[["text", "label"]]).map(tokenize_batch, batched=True)
data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

# 3. Modelo
ID2LABEL = {0: "negativo", 1: "positivo"}
LABEL2ID = {"negativo": 0, "positivo": 1}

model = AutoModelForSequenceClassification.from_pretrained(
    MODEL_NAME,
    num_labels=2,
    id2label=ID2LABEL,
    label2id=LABEL2ID
).to(device)

n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "precision": precision_score(labels, preds, zero_division=0),
        "recall": recall_score(labels, preds, zero_division=0),
        "f1": f1_score(labels, preds, zero_division=0),
        "kappa": cohen_kappa_score(labels, preds)
    }

# 4. Training
training_args = TrainingArguments(
    output_dir=str(ROOT_DIR / "outputs" / "distilbert_checkpoints"),
    num_train_epochs=3,
    learning_rate=3e-5,
    per_device_train_batch_size=16,
    per_device_eval_batch_size=32,
    weight_decay=0.01,
    eval_strategy="epoch",
    logging_strategy="epoch",
    save_strategy="no",
    seed=SEED,
    report_to="none"
)

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=ds_train,
    eval_dataset=ds_test,
    processing_class=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("[*] Iniciando entrenamiento de DistilBERT (3 épocas)...")
start_time = time.time()
trainer.train()
train_time = time.time() - start_time
print(f"[*] Entrenamiento completado en {train_time:.1f}s.")

# 5. Evaluación en test
preds = trainer.predict(ds_test)
y_true = preds.label_ids
y_pred = np.argmax(preds.predictions, axis=-1)

cm = confusion_matrix(y_true, y_pred)
acc = accuracy_score(y_true, y_pred)
prec = precision_score(y_true, y_pred, zero_division=0)
rec = recall_score(y_true, y_pred, zero_division=0)
f1 = f1_score(y_true, y_pred, zero_division=0)
kappa = cohen_kappa_score(y_true, y_pred)

print(f"[*] Test Accuracy: {acc:.4f} | F1: {f1:.4f} | Cohen's Kappa: {kappa:.4f}")

# 6. Guardar distilbert.json
metric_dict = {
    "model": "distilbert",
    "accuracy": round(float(acc), 4),
    "precision": round(float(prec), 4),
    "recall": round(float(rec), 4),
    "f1": round(float(f1), 4),
    "kappa": round(float(kappa), 4),
    "n_params": int(n_params),
    "train_seconds_per_run": round(float(train_time), 1),
    "confusion_matrix": cm.tolist()
}
(METRICS_DIR / "distilbert.json").write_text(json.dumps(metric_dict, indent=2), encoding="utf-8")
print(f"[*] Guardado: outputs/metrics/distilbert.json")

# 7. Matriz de confusión
fig, ax = plt.subplots(figsize=(4.5, 3.8))
disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["Negativo", "Positivo"])
disp.plot(cmap="Blues", values_format="d", ax=ax)
plt.title(f"Matriz de Confusión - DistilBERT (Acc: {acc:.3f})")
plt.tight_layout()
plt.savefig(FIG_DIR / "distilbert_confusion_matrix.png", dpi=200)
plt.close()
print(f"[*] Guardada: outputs/figures/distilbert_confusion_matrix.png")

# 8. Actualizar comparison.csv
ALL_MODELS = ["baseline_dummy", "dense", "vanilla_rnn", "lstm", "bilstm", "distilbert"]
rows = []
for m in ALL_MODELS:
    fp = METRICS_DIR / f"{m}.json"
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
df_comp.to_csv(METRICS_DIR / "comparison.csv", index=False)
print(f"[*] Actualizado: outputs/metrics/comparison.csv")
print(df_comp)

# 9. Gráfica comparativa global
mets = ["accuracy", "precision", "recall", "f1", "kappa"]
models_avail = [m for m in ALL_MODELS if (METRICS_DIR / f"{m}.json").exists()]
data_models = {n: json.loads((METRICS_DIR / f"{n}.json").read_text(encoding="utf-8")) for n in models_avail}

x = np.arange(len(mets))
w = 0.8 / len(models_avail)
colores = ["#7F8C8D", "#3498DB", "#E67E22", "#9B59B6", "#1ABC9C", "#27AE60"]

fig, ax = plt.subplots(figsize=(10, 4.2))
for i, n in enumerate(models_avail):
    vals = [data_models[n].get(m, 0.0) for m in mets]
    errs = [data_models[n].get("std", {}).get(m, 0.0) for m in mets]
    ax.bar(x + (i - (len(models_avail)-1)/2)*w, vals, w, label=n, color=colores[i % len(colores)],
           yerr=errs, capsize=2, edgecolor="black", linewidth=0.5)

ax.set_xticks(x)
ax.set_xticklabels(["Accuracy", "Precision", "Recall", "F1-Score", "Cohen's Kappa (κ)"], fontsize=10)
ax.set_ylim(0, 1.05)
ax.set_title("Comparación Integral de Modelos — Tarea Integradora 2", fontsize=12, fontweight="bold")
ax.grid(axis="y", linestyle=":", alpha=0.6)
ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
plt.tight_layout()
plt.savefig(FIG_DIR / "all_models_metrics_comparison.png", dpi=200)
plt.close()
print(f"[*] Guardada: outputs/figures/all_models_metrics_comparison.png")
print("[*] ¡Proceso completado exitosamente!")
