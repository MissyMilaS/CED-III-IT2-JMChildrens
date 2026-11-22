"""Rutinas de entrenamiento, búsqueda de hiperparámetros y evaluación final.

Flujo:
  1. load_tokens()      -> tokens de train.csv / test.csv (NO se vuelve a dividir el test)
  2. grid_search()      -> CV estratificada SOLO sobre train (el test no se toca)
  3. final_runs()       -> entrena con varias semillas, evalúa UNA vez en test y guarda JSON
"""
import json  # para guardar/leer métricas en formato JSON
import random  # generador aleatorio de Python (se fija para reproducibilidad)
import time  # para medir cuántos segundos tarda cada entrenamiento
from collections import Counter  # cuenta cuántas veces aparece cada palabra
from datetime import datetime  # fecha y hora para el log
from itertools import product  # producto cartesiano: todas las combinaciones del grid
from pathlib import Path  # manejo de rutas de archivos independiente del sistema operativo

import numpy as np  # arreglos numéricos
import pandas as pd  # tablas (DataFrame) y lectura de CSV
import torch  # deep learning
from sklearn.model_selection import StratifiedKFold, train_test_split  # CV estratificada y split estratificado
from torch import nn  # capas y funciones de pérdida
from torch.utils.data import DataLoader, TensorDataset  # para armar lotes (batches) de entrenamiento

from src.evaluate import METRICS_DIR, compute_metrics  # métricas del equipo (accuracy, precision, recall, f1, kappa)
from src.models import PAD, UNK, build_model, count_params  # arquitecturas y constantes de vocabulario

ROOT = Path(__file__).resolve().parents[1]  # carpeta raíz del repo (dos niveles arriba de este archivo)
DATA_DIR = ROOT / "data" / "processed"  # donde están train.csv y test.csv
MODELS_DIR = ROOT / "outputs" / "saved_models"  # donde se guardan los pesos entrenados
LOG_FILE = ROOT / "logs" / "training_logs.txt"  # archivo de log de entrenamiento

SEED = 42  # semilla base para reproducibilidad
MAX_LEN = 30  # los comentarios tienen como máximo 30 tokens
MAX_VOCAB = 5000  # tamaño máximo del vocabulario (incluye <pad> y <unk>)
MODEL_NAMES = ["dense", "vanilla_rnn", "lstm"]  # los 3 modelos que pide el enunciado
EXTRA_MODEL_NAMES = ["bilstm"]  # modelo extra del equipo: LSTM bidireccional
ALL_MODEL_NAMES = MODEL_NAMES + EXTRA_MODEL_NAMES  # todos los modelos neuronales de esta parte

# Hiperparámetros explorados (epochs se decide con early stopping)
GRID = {"hidden": [32, 64, 128], "batch_size": [16, 32, 64], "lr": [1e-3, 3e-3]}  # valores a probar en el grid
FIXED = {"emb_dim": 64, "dropout": 0.5}  # hiperparámetros que no se exploran


# ----------------------------------------------------------------- utilidades
def set_seed(seed):  # fija todas las semillas para que los resultados se puedan repetir
    random.seed(seed)  # semilla del módulo random
    np.random.seed(seed)  # semilla de numpy
    torch.manual_seed(seed)  # semilla de torch (inicialización de pesos, dropout)


def log(msg):  # imprime un mensaje y lo guarda en el archivo de log
    """Imprime y agrega la línea a logs/training_logs.txt."""
    line = f"[{datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"  # antepone fecha y hora al mensaje
    print(line)  # lo muestra en pantalla/notebook
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)  # crea la carpeta logs/ si no existe
    with open(LOG_FILE, "a", encoding="utf-8") as f:  # abre el log en modo "agregar"
        f.write(line + "\n")  # escribe la línea al final del archivo


# ----------------------------------------------------------------- datos
def load_tokens(text_mode="clean"):  # lee los CSV y devuelve los comentarios como listas de tokens
    """Lee train.csv / test.csv ya generados por el equipo (mismo split para todos).

    text_mode='clean': usa la columna `clean` (sin stopwords, como el pipeline del equipo).
    text_mode='full' : tokeniza `text` conservando stopwords (para el estudio de ablación).
    Devuelve (tok_train, y_train, tok_test, y_test) con tokens en listas de listas.
    """
    train = pd.read_csv(DATA_DIR / "train.csv")  # carga el conjunto de entrenamiento (800 filas)
    test = pd.read_csv(DATA_DIR / "test.csv")  # carga el conjunto de prueba (200 filas)
    if text_mode == "clean":  # usar el texto ya limpio por Sarah
        tok = lambda s: [str(t).split() if isinstance(t, str) else [] for t in s]  # separa por espacios; si la celda quedó vacía (NaN) devuelve lista vacía
        tok_tr, tok_te = tok(train["clean"]), tok(test["clean"])  # tokens de train y de test
    elif text_mode == "full":  # usar el texto original conservando stopwords
        from src.preprocessing import tokenize  # se importa aquí para no exigir nltk si no se usa
        tok_tr = [tokenize(t, remove_stopwords=False) for t in train["text"]]  # tokeniza cada comentario de train sin quitar stopwords
        tok_te = [tokenize(t, remove_stopwords=False) for t in test["text"]]  # igual para test
    else:  # cualquier otro valor es un error de uso
        raise ValueError("text_mode debe ser 'clean' o 'full'")  # mensaje claro
    return (tok_tr, train["label"].astype(int).values,  # tokens y etiquetas (0 neg / 1 pos) de train
            tok_te, test["label"].astype(int).values)  # tokens y etiquetas de test


def build_vocab(token_lists, max_size=MAX_VOCAB, min_freq=1):  # crea el diccionario palabra -> id
    """Vocabulario construido SOLO con datos de entrenamiento (evita fuga)."""
    counts = Counter(t for toks in token_lists for t in toks)  # frecuencia de cada palabra en los comentarios recibidos
    words = [w for w, c in counts.most_common(max_size - 2) if c >= min_freq]  # las palabras más frecuentes (deja 2 lugares para <pad> y <unk>)
    vocab = {"<pad>": PAD, "<unk>": UNK}  # ids reservados: 0 relleno, 1 desconocida
    vocab.update({w: i + 2 for i, w in enumerate(words)})  # el resto de palabras recibe ids desde 2
    return vocab  # diccionario final


def encode(token_lists, vocab, max_len=MAX_LEN):  # convierte tokens a tensores de enteros con padding
    """Tokens -> tensor (N, max_len) con padding a la derecha + longitudes reales (>= 1)."""
    ids = np.zeros((len(token_lists), max_len), dtype=np.int64)  # matriz N x max_len llena de 0 (= <pad>)
    lengths = np.zeros(len(token_lists), dtype=np.int64)  # largo real de cada comentario
    for i, toks in enumerate(token_lists):  # recorre cada comentario
        seq = [vocab.get(t, UNK) for t in toks][:max_len] or [UNK]  # palabra -> id (o <unk> si no está), recorta a max_len; si quedó vacío usa [<unk>]
        ids[i, :len(seq)] = seq  # escribe los ids al inicio de la fila (el resto sigue en 0)
        lengths[i] = len(seq)  # guarda el largo real (nunca 0)
    return torch.from_numpy(ids), torch.from_numpy(lengths)  # convierte a tensores de torch


def _subset(lst, idx):  # selecciona elementos de una lista según posiciones
    return [lst[i] for i in idx]  # lista nueva solo con las posiciones pedidas


# ----------------------------------------------------------------- entrenamiento
@torch.no_grad()  # en predicción no se calculan gradientes (más rápido, menos memoria)
def predict_proba(model, X, L):  # devuelve la probabilidad de clase positiva
    model.eval()  # modo evaluación: apaga el dropout
    return torch.sigmoid(model(X, L)).numpy()  # logit -> probabilidad (0 a 1) -> arreglo numpy


def train_model(model, Xtr, Ltr, ytr, Xva, Lva, yva, lr, batch_size,
                max_epochs=30, patience=5, seed=SEED):  # entrena un modelo ya construido
    """Adam + BCEWithLogits + clipping de gradiente + early stopping sobre val loss.

    Devuelve el modelo con los mejores pesos y el historial por época.
    """
    g = torch.Generator().manual_seed(seed)  # generador propio para barajar los lotes de forma reproducible
    loader = DataLoader(TensorDataset(Xtr, Ltr, ytr), batch_size=batch_size,
                        shuffle=True, generator=g)  # reparte train en lotes y los baraja cada época
    opt = torch.optim.Adam(model.parameters(), lr=lr)  # optimizador Adam con la tasa de aprendizaje dada
    loss_fn = nn.BCEWithLogitsLoss()  # pérdida binaria: aplica la sigmoide internamente sobre el logit

    best_loss, best_state, best_epoch, wait = float("inf"), None, 0, 0  # mejor pérdida de validación vista, sus pesos, su época y épocas sin mejorar
    history = []  # aquí se guarda el resumen de cada época
    for epoch in range(1, max_epochs + 1):  # repite hasta max_epochs épocas
        model.train()  # modo entrenamiento: activa el dropout
        tot, n = 0.0, 0  # acumuladores de pérdida total y de nº de ejemplos
        for xb, lb, yb in loader:  # recorre un lote a la vez (ids, largos, etiquetas)
            opt.zero_grad()  # borra los gradientes del lote anterior
            loss = loss_fn(model(xb, lb), yb)  # predice y calcula la pérdida contra las etiquetas reales
            loss.backward()  # backpropagation: calcula los gradientes
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)  # limita la norma del gradiente a 1 (evita gradientes explosivos en RNN)
            opt.step()  # actualiza los pesos
            tot += loss.item() * len(yb)  # suma la pérdida ponderada por el tamaño del lote
            n += len(yb)  # cuenta los ejemplos vistos

        model.eval()  # modo evaluación para medir en validación
        with torch.no_grad():  # sin gradientes
            logits = model(Xva, Lva)  # logits sobre todo el conjunto de validación
            val_loss = loss_fn(logits, yva).item()  # pérdida de validación
            val_acc = ((logits > 0).float() == yva).float().mean().item()  # accuracy: logit > 0 equivale a probabilidad > 0.5
        history.append({"epoch": epoch, "train_loss": tot / n,
                        "val_loss": val_loss, "val_acc": val_acc})  # guarda el resumen de la época

        if val_loss < best_loss - 1e-4:  # ¿mejoró la validación de forma apreciable?
            best_loss, best_epoch, wait = val_loss, epoch, 0  # sí: actualiza el mejor valor y reinicia el contador de espera
            best_state = {k: v.clone() for k, v in model.state_dict().items()}  # copia los pesos actuales como los mejores
        else:  # no mejoró
            wait += 1  # una época más sin mejorar
            if wait >= patience:  # si se agotó la paciencia
                break  # early stopping: se detiene el entrenamiento

    model.load_state_dict(best_state)  # restaura los mejores pesos (no los de la última época)
    return model, history, best_epoch  # modelo, historial y época óptima


def fit_eval(model_name, tok_tr, y_tr, tok_va, y_va, params, seed=SEED,
             max_epochs=30, patience=5):  # prepara datos, construye y entrena un modelo
    """Construye vocab con tok_tr, entrena con early stopping en (tok_va, y_va)."""
    set_seed(seed)  # fija semillas antes de construir el modelo
    vocab = build_vocab(tok_tr)  # vocabulario solo con los datos de entrenamiento
    Xtr, Ltr = encode(tok_tr, vocab)  # entrenamiento -> ids y largos
    Xva, Lva = encode(tok_va, vocab)  # validación -> ids y largos (palabras no vistas pasan a <unk>)
    ytr = torch.tensor(y_tr, dtype=torch.float32)  # etiquetas de train como float (lo que pide BCEWithLogitsLoss)
    yva = torch.tensor(y_va, dtype=torch.float32)  # etiquetas de validación como float

    model_kw = {k: params[k] for k in ("emb_dim", "hidden", "dropout") if k in params}  # separa los parámetros que son de la arquitectura
    model = build_model(model_name, len(vocab), **model_kw)  # crea el modelo ('dense', 'vanilla_rnn', 'lstm' o 'bilstm')
    model, history, best_epoch = train_model(
        model, Xtr, Ltr, ytr, Xva, Lva, yva, lr=params["lr"],
        batch_size=params["batch_size"], max_epochs=max_epochs,
        patience=patience, seed=seed)  # entrena con lr y batch size del grid
    pred_va = (predict_proba(model, Xva, Lva) >= 0.5).astype(int)  # predicciones de validación: probabilidad >= 0.5 -> clase 1
    return {"model": model, "vocab": vocab, "history": history,
            "best_epoch": best_epoch, "val_metrics": compute_metrics(y_va, pred_va)}  # todo lo necesario para analizar la corrida


# ----------------------------------------------------------------- búsqueda de hiperparámetros
def grid_search(model_name, tok_tr, y_tr, grid=GRID, fixed=FIXED, n_splits=3,
                seed=SEED, max_epochs=30, patience=5):  # prueba todas las combinaciones del grid
    """Grid manual con CV estratificada SOLO sobre train. Devuelve DataFrame ordenado por accuracy."""
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)  # divide train en 3 folds manteniendo 50/50 de clases
    keys = list(grid)  # nombres de los hiperparámetros (hidden, batch_size, lr)
    rows = []  # una fila de resultados por combinación
    combos = list(product(*grid.values()))  # todas las combinaciones posibles de valores
    log(f"[{model_name}] grid search: {len(combos)} configuraciones x {n_splits} folds")  # registra el tamaño de la búsqueda
    for i, combo in enumerate(combos, 1):  # recorre cada combinación
        params = {**fixed, **dict(zip(keys, combo))}  # une los parámetros fijos con los de esta combinación
        accs, f1s, eps = [], [], []  # métricas por fold
        t0 = time.time()  # inicio del cronómetro
        for tr_idx, va_idx in skf.split(np.zeros(len(y_tr)), y_tr):  # recorre los 3 folds (índices de train y de validación)
            r = fit_eval(model_name, _subset(tok_tr, tr_idx), y_tr[tr_idx],
                         _subset(tok_tr, va_idx), y_tr[va_idx], params,
                         seed=seed, max_epochs=max_epochs, patience=patience)  # entrena en 2 folds y valida en el tercero
            accs.append(r["val_metrics"]["accuracy"])  # accuracy en el fold de validación
            f1s.append(r["val_metrics"]["f1"])  # F1 en el fold de validación
            eps.append(r["best_epoch"])  # época óptima de este fold
        row = {**dict(zip(keys, combo)), "val_acc": np.mean(accs),
               "val_acc_std": np.std(accs), "val_f1": np.mean(f1s),
               "best_epoch": float(np.mean(eps))}  # promedios sobre los 3 folds
        rows.append(row)  # guarda la fila
        log(f"[{model_name}] {i}/{len(combos)} {dict(zip(keys, combo))} "
            f"val_acc={row['val_acc']:.4f} best_epoch={row['best_epoch']:.1f} "
            f"({time.time() - t0:.1f}s)")  # registra el avance y el tiempo
    return pd.DataFrame(rows).sort_values("val_acc", ascending=False).reset_index(drop=True)  # tabla ordenada: la mejor configuración queda en la fila 0


# ----------------------------------------------------------------- evaluación final
def final_runs(model_name, tok_tr, y_tr, tok_te, y_te, params,
               seeds=(42, 43, 44, 45, 46), val_size=0.15, save_name=None,
               max_epochs=30, patience=5):  # entrenamiento final con varias semillas y evaluación en test
    """Entrena con varias semillas, evalúa en test UNA vez por semilla y guarda el JSON.

    - Validación (para early stopping) = 15 % de train; el test nunca se usa para decidir nada.
    - outputs/metrics/<save_name>.json mantiene el formato del equipo (model, accuracy,
      precision, recall, f1, kappa, confusion_matrix) con la MEDIA de las semillas, y añade
      `std`, `n_seeds`, `params`, `n_params`, `train_seconds`.
    - Guarda los pesos de la primera semilla en outputs/saved_models/<save_name>.pt
    """
    save_name = save_name or model_name  # nombre del archivo de salida (por defecto, el del modelo)
    all_metrics, cms, secs, results = [], [], [], []  # acumuladores: métricas, matrices de confusión, tiempos y resultados completos
    for s in seeds:  # una corrida por semilla
        idx_tr, idx_va = train_test_split(np.arange(len(y_tr)), test_size=val_size,
                                          stratify=y_tr, random_state=s)  # separa 15 % de train como validación para el early stopping
        t0 = time.time()  # inicio del cronómetro
        r = fit_eval(model_name, _subset(tok_tr, idx_tr), y_tr[idx_tr],
                     _subset(tok_tr, idx_va), y_tr[idx_va], params, seed=s,
                     max_epochs=max_epochs, patience=patience)  # entrena con el 85 % restante
        secs.append(time.time() - t0)  # guarda los segundos que tardó

        Xte, Lte = encode(tok_te, r["vocab"])  # codifica el test con el vocabulario de train
        proba = predict_proba(r["model"], Xte, Lte)  # probabilidades sobre test
        pred = (proba >= 0.5).astype(int)  # clase predicha: 1 si probabilidad >= 0.5
        m = compute_metrics(y_te, pred)  # accuracy, precision, recall, f1, kappa en test
        all_metrics.append(m)  # guarda las métricas de esta semilla
        cm = np.zeros((2, 2), dtype=int)  # matriz de confusión 2x2 vacía
        for t, p in zip(y_te, pred):  # recorre real y predicho
            cm[t, p] += 1  # fila = real, columna = predicho
        cms.append(cm)  # guarda la matriz de esta semilla
        r.update({"test_metrics": m, "proba": proba, "pred": pred, "seed": s})  # agrega datos de test al resultado
        results.append(r)  # guarda el resultado completo (sirve para graficar curvas)
        log(f"[{save_name}] seed={s} best_epoch={r['best_epoch']} "
            f"test_acc={m['accuracy']:.4f} f1={m['f1']:.4f} kappa={m['kappa']:.4f} "
            f"({secs[-1]:.1f}s)")  # registra el resultado de la semilla

    keys = ["accuracy", "precision", "recall", "f1", "kappa"]  # las 5 métricas del enunciado
    mean = {k: float(np.mean([m[k] for m in all_metrics])) for k in keys}  # media de cada métrica entre semillas
    std = {k: float(np.std([m[k] for m in all_metrics])) for k in keys}  # desviación estándar entre semillas

    out = {"model": save_name, **{k: round(v, 4) for k, v in mean.items()},  # nombre y métricas medias a 4 decimales (formato del equipo)
           "confusion_matrix": np.mean(cms, axis=0).round(1).tolist(),  # matriz de confusión promedio
           "std": {k: round(v, 4) for k, v in std.items()},  # desviaciones
           "n_seeds": len(seeds), "params": params,  # nº de semillas e hiperparámetros usados
           "n_params": count_params(results[0]["model"]),  # nº de parámetros entrenables (complejidad)
           "train_seconds_per_run": round(float(np.mean(secs)), 1)}  # segundos promedio por corrida
    METRICS_DIR.mkdir(parents=True, exist_ok=True)  # crea outputs/metrics si no existe
    (METRICS_DIR / f"{save_name}.json").write_text(json.dumps(out, indent=2))  # escribe el JSON de métricas

    MODELS_DIR.mkdir(parents=True, exist_ok=True)  # crea outputs/saved_models si no existe
    torch.save({"state_dict": results[0]["model"].state_dict(),
                "vocab": results[0]["vocab"], "params": params},
               MODELS_DIR / f"{save_name}.pt")  # guarda pesos, vocabulario e hiperparámetros de la primera semilla
    log(f"[{save_name}] MEDIA {len(seeds)} semillas: "
        + " ".join(f"{k}={mean[k]:.4f}±{std[k]:.4f}" for k in keys))  # registra el resumen final
    return out, results  # resumen (para tablas) y resultados completos (para figuras)