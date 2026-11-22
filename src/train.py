"""Rutinas de entrenamiento, búsqueda de hiperparámetros y evaluación final.

Flujo:
  1. load_tokens()      -> tokens de train.csv / test.csv (NO se vuelve a dividir el test)
  2. grid_search()      -> CV estratificada SOLO sobre train (el test no se toca)
  3. final_runs()       -> entrena con varias semillas, evalúa UNA vez en test y guarda JSON
"""
import json
import random
import time
from collections import Counter
from datetime import datetime
from itertools import product
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluate import METRICS_DIR, compute_metrics
from src.models import PAD, UNK, build_model, count_params

ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "outputs" / "saved_models"
LOG_FILE = ROOT / "logs" / "training_logs.txt"

SEED = 77767
MAX_LEN = 30            # los comentarios tienen como máximo 30 tokens
MAX_VOCAB = 5000
MODEL_NAMES = ["dense", "vanilla_rnn", "lstm"]

# Hiperparámetros explorados (epochs se decide con early stopping)
GRID = {"hidden": [32, 64, 128], "batch_size": [16, 32, 64], "lr": [1e-3, 3e-3]}
FIXED = {"emb_dim": 64, "dropout": 0.5}


# ----------------------------------------------------------------- utilidades
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def log(msg):
    """Imprime y agrega la línea a logs/training_logs.txt."""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line)
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


# ----------------------------------------------------------------- datos
def load_tokens(text_mode="clean"):
    """Lee train.csv / test.csv ya generados por el equipo (mismo split para todos).

    text_mode='clean': usa la columna `clean` (sin stopwords, como el pipeline del equipo).
    text_mode='full' : tokeniza `text` conservando stopwords (para el estudio de ablación).
    Devuelve (tok_train, y_train, tok_test, y_test) con tokens en listas de listas.
    """
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    if text_mode == "clean":
        tok = lambda s: [str(t).split() if isinstance(t, str) else [] for t in s]
        tok_tr, tok_te = tok(train["clean"]), tok(test["clean"])
    elif text_mode == "full":
        from src.preprocessing import tokenize
        tok_tr = [tokenize(t, remove_stopwords=False) for t in train["text"]]
        tok_te = [tokenize(t, remove_stopwords=False) for t in test["text"]]
    else:
        raise ValueError("text_mode debe ser 'clean' o 'full'")
    return (tok_tr, train["label"].astype(int).values,
            tok_te, test["label"].astype(int).values)


def build_vocab(token_lists, max_size=MAX_VOCAB, min_freq=1):
    """Vocabulario construido SOLO con datos de entrenamiento (evita fuga)."""
    counts = Counter(t for toks in token_lists for t in toks)
    words = [w for w, c in counts.most_common(max_size - 2) if c >= min_freq]
    vocab = {"<pad>": PAD, "<unk>": UNK}
    vocab.update({w: i + 2 for i, w in enumerate(words)})
    return vocab


def encode(token_lists, vocab, max_len=MAX_LEN):
    """Tokens -> tensor (N, max_len) con padding a la derecha + longitudes reales (>= 1)."""
    ids = np.zeros((len(token_lists), max_len), dtype=np.int64)
    lengths = np.zeros(len(token_lists), dtype=np.int64)
    for i, toks in enumerate(token_lists):
        seq = [vocab.get(t, UNK) for t in toks][:max_len] or [UNK]   # vacío -> <unk>
        ids[i, :len(seq)] = seq
        lengths[i] = len(seq)
    return torch.from_numpy(ids), torch.from_numpy(lengths)


def _subset(lst, idx):
    return [lst[i] for i in idx]


# ----------------------------------------------------------------- entrenamiento
@torch.no_grad()
def predict_proba(model, X, L):
    model.eval()
    return torch.sigmoid(model(X, L)).numpy()


def train_model(model, Xtr, Ltr, ytr, Xva, Lva, yva, lr, batch_size,
                max_epochs=30, patience=5, seed=SEED):
    """Adam + BCEWithLogits + clipping de gradiente + early stopping sobre val loss.

    Devuelve el modelo con los mejores pesos y el historial por época.
    """
    g = torch.Generator().manual_seed(seed)
    loader = DataLoader(TensorDataset(Xtr, Ltr, ytr), batch_size=batch_size,
                        shuffle=True, generator=g)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.BCEWithLogitsLoss()

    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0
    history = []
    for epoch in range(1, max_epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for xb, lb, yb in loader:
            opt.zero_grad()
            loss = loss_fn(model(xb, lb), yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item() * len(yb)
            n += len(yb)

        model.eval()
        with torch.no_grad():
            logits = model(Xva, Lva)
            val_loss = loss_fn(logits, yva).item()
            val_acc = ((logits > 0).float() == yva).float().mean().item()
        history.append({"epoch": epoch, "train_loss": tot / n,
                        "val_loss": val_loss, "val_acc": val_acc})

        if val_loss < best_loss - 1e-4:
            best_loss, best_epoch, wait = val_loss, epoch, 0
            best_state = {k: v.clone() for k, v in model.state_dict().items()}
        else:
            wait += 1
            if wait >= patience:
                break

    model.load_state_dict(best_state)
    return model, history, best_epoch


def fit_eval(model_name, tok_tr, y_tr, tok_va, y_va, params, seed=SEED,
             max_epochs=30, patience=5):
    """Construye vocab con tok_tr, entrena con early stopping en (tok_va, y_va)."""
    set_seed(seed)
    vocab = build_vocab(tok_tr)
    Xtr, Ltr = encode(tok_tr, vocab)
    Xva, Lva = encode(tok_va, vocab)
    ytr = torch.tensor(y_tr, dtype=torch.float32)
    yva = torch.tensor(y_va, dtype=torch.float32)

    model_kw = {k: params[k] for k in ("emb_dim", "hidden", "dropout") if k in params}
    model = build_model(model_name, len(vocab), **model_kw)
    model, history, best_epoch = train_model(
        model, Xtr, Ltr, ytr, Xva, Lva, yva, lr=params["lr"],
        batch_size=params["batch_size"], max_epochs=max_epochs,
        patience=patience, seed=seed)
    pred_va = (predict_proba(model, Xva, Lva) >= 0.5).astype(int)
    return {"model": model, "vocab": vocab, "history": history,
            "best_epoch": best_epoch, "val_metrics": compute_metrics(y_va, pred_va)}


# ----------------------------------------------------------------- búsqueda de hiperparámetros
def grid_search(model_name, tok_tr, y_tr, grid=GRID, fixed=FIXED, n_splits=3,
                seed=SEED, max_epochs=30, patience=5):
    """Grid manual con CV estratificada SOLO sobre train. Devuelve DataFrame ordenado por accuracy."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    keys = list(grid)
    rows = []
    combos = list(product(*grid.values()))
    log(f"[{model_name}] grid search: {len(combos)} configuraciones x {n_splits} folds")
    for i, combo in enumerate(combos, 1):
        params = {**fixed, **dict(zip(keys, combo))}
        accs, f1s, eps = [], [], []
        t0 = time.time()
        for tr_idx, va_idx in skf.split(np.zeros(len(y_tr)), y_tr):
            r = fit_eval(model_name, _subset(tok_tr, tr_idx), y_tr[tr_idx],
                         _subset(tok_tr, va_idx), y_tr[va_idx], params,
                         seed=seed, max_epochs=max_epochs, patience=patience)
            accs.append(r["val_metrics"]["accuracy"])
            f1s.append(r["val_metrics"]["f1"])
            eps.append(r["best_epoch"])
        row = {**dict(zip(keys, combo)), "val_acc": np.mean(accs),
               "val_acc_std": np.std(accs), "val_f1": np.mean(f1s),
               "best_epoch": float(np.mean(eps))}
        rows.append(row)
        log(f"[{model_name}] {i}/{len(combos)} {dict(zip(keys, combo))} "
            f"val_acc={row['val_acc']:.4f} best_epoch={row['best_epoch']:.1f} "
            f"({time.time() - t0:.1f}s)")
    return pd.DataFrame(rows).sort_values("val_acc", ascending=False).reset_index(drop=True)


# ----------------------------------------------------------------- evaluación final
def final_runs(model_name, tok_tr, y_tr, tok_te, y_te, params,
               seeds=(77767, 77768, 77769, 77770, 77771), val_size=0.15, save_name=None,
               max_epochs=30, patience=5):
    """Entrena con varias semillas, evalúa en test UNA vez por semilla y guarda el JSON.

    - Validación (para early stopping) = 15 % de train; el test nunca se usa para decidir nada.
    - outputs/metrics/<save_name>.json mantiene el formato del equipo (model, accuracy,
      precision, recall, f1, kappa, confusion_matrix) con la MEDIA de las semillas, y añade
      `std`, `n_seeds`, `params`, `n_params`, `train_seconds`.
    - Guarda los pesos de la primera semilla en outputs/saved_models/<save_name>.pt
    """
    save_name = save_name or model_name
    all_metrics, cms, secs, results = [], [], [], []
    for s in seeds:
        idx_tr, idx_va = train_test_split(np.arange(len(y_tr)), test_size=val_size,
                                          stratify=y_tr, random_state=s)
        t0 = time.time()
        r = fit_eval(model_name, _subset(tok_tr, idx_tr), y_tr[idx_tr],
                     _subset(tok_tr, idx_va), y_tr[idx_va], params, seed=s,
                     max_epochs=max_epochs, patience=patience)
        secs.append(time.time() - t0)

        Xte, Lte = encode(tok_te, r["vocab"])
        proba = predict_proba(r["model"], Xte, Lte)
        pred = (proba >= 0.5).astype(int)
        m = compute_metrics(y_te, pred)
        all_metrics.append(m)
        cm = np.zeros((2, 2), dtype=int)
        for t, p in zip(y_te, pred):
            cm[t, p] += 1
        cms.append(cm)
        r.update({"test_metrics": m, "proba": proba, "pred": pred, "seed": s})
        results.append(r)
        log(f"[{save_name}] seed={s} best_epoch={r['best_epoch']} "
            f"test_acc={m['accuracy']:.4f} f1={m['f1']:.4f} kappa={m['kappa']:.4f} "
            f"({secs[-1]:.1f}s)")

    keys = ["accuracy", "precision", "recall", "f1", "kappa"]
    mean = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}
    std = {k: float(np.std([m[k] for m in all_metrics])) for k in keys}

    out = {"model": save_name, **{k: round(v, 4) for k, v in mean.items()},
           "confusion_matrix": np.mean(cms, axis=0).round(1).tolist(),
           "std": {k: round(v, 4) for k, v in std.items()},
           "n_seeds": len(seeds), "params": params,
           "n_params": count_params(results[0]["model"]),
           "train_seconds_per_run": round(float(np.mean(secs)), 1)}
    METRICS_DIR.mkdir(parents=True, exist_ok=True)
    (METRICS_DIR / f"{save_name}.json").write_text(json.dumps(out, indent=2))

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": results[0]["model"].state_dict(),
                "vocab": results[0]["vocab"], "params": params},
               MODELS_DIR / f"{save_name}.pt")
    log(f"[{save_name}] MEDIA {len(seeds)} semillas: "
        + " ".join(f"{k}={mean[k]:.4f}±{std[k]:.4f}" for k in keys))
    return out, results