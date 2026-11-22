"""
Preparación de datos para GoEmotions (sentimiento positivo vs. negativo).

Pasos: carga de los CSV crudos -> agregación por comentario (>= 2 evaluadores)
-> asignación de sentimiento -> dataset balanceado -> limpieza de texto -> split.
"""

import re                                   # expresiones regulares (para tokenizar y quitar URLs)
from pathlib import Path                    # manejo de rutas que funciona igual en Windows, Mac y Linux

import pandas as pd                         # tablas (DataFrames)
from sklearn.model_selection import train_test_split   # divide los datos en train y test

# Constantes
SEED = 42 # semilla fija: cualquiera obtiene el mismo muestreo y el mismo split
# __file__ = esta archivo (src/preprocessing.py); .resolve() = ruta absoluta;
# .parents[1] = dos niveles arriba = raíz del repo; de ahí se baja a data/raw
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

# Mapeo oficial de GoEmotions: cada una de las 27 emociones pertenece a un grupo de sentimiento
POSITIVE = ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"]
NEGATIVE = ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"]
AMBIGUOUS = ["realization", "surprise", "curiosity", "confusion"]
# Lista de todas las columnas de emociones del CSV (27 + neutral); sirve para seleccionarlas de golpe
EMOTIONS = POSITIVE + NEGATIVE + AMBIGUOUS + ["neutral"]

# Stopwords y tokenizador
try:
    import nltk                                          # librería de PLN; aquí solo aporta la lista de stopwords
    try:
        nltk.data.find("corpora/stopwords")              # ¿ya está descargada la lista de stopwords?
    except LookupError:                                  # si no está...
        nltk.download("stopwords", quiet=True)           # ...se descarga (solo la primera vez)
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))          # set = búsqueda rápida de palabras
except Exception:                                        # nltk no instalado o sin internet para descargar
    print("Aviso: usando stopwords de scikit-learn (nltk no disponible)")
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS   # lista de respaldo
    STOPWORDS = set(ENGLISH_STOP_WORDS)

# Las negaciones cambian el sentimiento ("not good"), así que no se eliminan
NEGATIONS = {"no", "not", "nor", "never", "nothing", "none", "neither", "cannot"}
# |= agrega al conjunto todas las stopwords con "n't" (didn't, isn't, wasn't, ...)
NEGATIONS |= {w for w in STOPWORDS if "n't" in w}
# Stopwords a eliminar = todas menos las negaciones (resta de conjuntos)
STOPWORDS_SENT = STOPWORDS - NEGATIONS

# Patrón que define qué es un "token". Se prueba en este orden, separado por "|":
#   \[name\] y \[religion\]  -> los marcadores del dataset (los corchetes llevan \ porque en regex son especiales)
#   [a-z]+(?:'[a-z]+)?       -> una palabra, con parte opcional tras apóstrofe (don't, it's, i'm)
#   \d+                      -> números
# Todo lo que no encaje (puntuación, emojis, símbolos) queda fuera.
TOKEN_RE = re.compile(r"\[name\]|\[religion\]|[a-z]+(?:'[a-z]+)?|\d+")


# Carga
def load_raw():
    """Carga y une los tres CSV crudos (una fila por evaluador y comentario)."""
    files = [RAW_DIR / f"goemotions_{i}.csv" for i in (1, 2, 3)]          # rutas de los 3 archivos
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)  # lee cada uno y los apila; ignore_index renumera las filas 0..N


# Agregación y sentimiento
def aggregate_filtered(raw):
    """Una emoción es válida si al menos 2 evaluadores la marcaron.
    Se descartan los comentarios que quedan sin ninguna emoción."""
    raw = raw.drop_duplicates(subset=["id", "rater_id"])        # protección: un mismo evaluador no debe contar dos veces
    votes = raw.groupby("id")[EMOTIONS].sum()                   # por comentario, suma los votos de cada emoción (0 a 5)
    text = raw.drop_duplicates("id").set_index("id")["text"]    # una sola copia del texto por comentario, indexada por id
    agreed = (votes >= 2).astype(int)                           # 1 si al menos 2 evaluadores marcaron la emoción, 0 si no
    agreed = agreed[agreed.sum(axis=1) > 0]                     # conserva solo comentarios con al menos una emoción válida
    return agreed.join(text)                                    # pega el texto (une por el índice, que es el id)


def assign_sentiment(df):
    """Clasifica cada comentario como positive, negative, neutral, ambiguous o mixed."""
    pos = df[POSITIVE].sum(axis=1) > 0      # True si el comentario tiene alguna emoción positiva
    neg = df[NEGATIVE].sum(axis=1) > 0      # ... alguna negativa
    amb = df[AMBIGUOUS].sum(axis=1) > 0     # ... alguna ambigua
    neu = df["neutral"] > 0                 # ... neutral
    # Cuenta en cuántos grupos cae cada comentario (True = 1, False = 0)
    n_groups = pos.astype(int) + neg.astype(int) + amb.astype(int) + neu.astype(int)

    out = df.copy()                         # copia para no modificar el DataFrame original
    out["group"] = "mixed"                  # por defecto: si cae en 2 o más grupos, es "mixed"
    # Si cae en un único grupo (n_groups == 1) y ese grupo es X, recibe la etiqueta X
    out.loc[(n_groups == 1) & pos, "group"] = "positive"
    out.loc[(n_groups == 1) & neg, "group"] = "negative"
    out.loc[(n_groups == 1) & amb, "group"] = "ambiguous"
    out.loc[(n_groups == 1) & neu, "group"] = "neutral"
    return out


# Dataset balanceado
def _pos_neg(df):
    """Se queda solo con positivos y negativos y agrega text_key (texto normalizado para comparar)."""
    d = df[df["group"].isin(["positive", "negative"])].copy()    # solo positivos y negativos
    d["text_key"] = d["text"].str.strip().str.lower()            # texto normalizado para detectar repetidos ("Hi " = "hi")
    return d


def _drop_conflicts(d):
    """Quita los textos repetidos que aparecen con etiquetas distintas (contradictorios)."""
    conflict = d.groupby("text_key")["group"].nunique() > 1      # True si el mismo texto aparece con 2 etiquetas
    return d[~d["text_key"].isin(conflict[conflict].index)]      # ~ = negación: quita esos textos


def build_balanced(df, n=500, seed=SEED):
    """Solo positivos y negativos, sin duplicados, n comentarios por clase."""
    d = _drop_conflicts(_pos_neg(df))                            # positivos/negativos sin textos contradictorios
    d = d.drop_duplicates("text_key")                            # deja una sola copia de cada texto

    pos = d[d["group"] == "positive"].sample(n, random_state=seed)   # muestra aleatoria reproducible de n positivos
    neg = d[d["group"] == "negative"].sample(n, random_state=seed)   # y de n negativos
    out = pd.concat([pos, neg]).sample(frac=1, random_state=seed)    # los une y los baraja (frac=1 = todas las filas en otro orden)
    out["label"] = (out["group"] == "positive").astype(int)   # etiqueta numérica para los modelos: 1 = positivo, 0 = negativo
    return out[["text", "group", "label"]].reset_index()      # deja 3 columnas; reset_index vuelve el id una columna normal


def filter_funnel(raw, filtered, n=500):
    """Cuántos comentarios quedan tras cada paso del filtrado (evidencia de calidad de datos)."""
    d = _pos_neg(filtered)                                       # positivos o negativos de un solo grupo
    n_pos_neg = len(d)
    d = _drop_conflicts(d)                                       # sin etiquetas contradictorias
    n_no_conflict = len(d)
    d = d.drop_duplicates("text_key")                            # sin textos duplicados
    steps = [
        ("Raw rows (one per rater)", len(raw)),
        ("Unique comments", raw["id"].nunique()),
        ("With >= 1 emotion agreed by >= 2 raters", len(filtered)),
        ("Single group: positive or negative only", n_pos_neg),
        ("After removing conflicting duplicate texts", n_no_conflict),
        ("After removing duplicate texts", len(d)),
        (f"Balanced sample ({n} + {n})", 2 * n),
    ]
    out = pd.DataFrame(steps, columns=["step", "comments"])
    out["removed"] = (-out["comments"].diff()).astype("Int64")   # cuántos se pierden respecto al paso anterior
    out.loc[:1, "removed"] = pd.NA                               # filas -> comentarios únicos no es una pérdida
    return out


# Limpieza de texto
def tokenize(text, remove_stopwords=True):
    """Minúsculas, sin URLs, puntuación ni emojis; opcionalmente sin stopwords."""
    # minúsculas y apóstrofes curvos (’ ‘) pasados a rectos ('), para que "didn’t" no se parta en dos
    t = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"https?://\S+", " ", t)                       # reemplaza cualquier URL por un espacio
    tokens = TOKEN_RE.findall(t)                              # extrae todos los tokens que encajan con el patrón
    if remove_stopwords:                                      # opcional: las RNN/LSTM podrían rendir mejor sin quitarlas
        tokens = [w for w in tokens if w not in STOPWORDS_SENT]   # conserva solo las palabras que NO son stopwords
    return tokens                                             # lista de palabras


def clean_text(text, remove_stopwords=True):
    return " ".join(tokenize(text, remove_stopwords))         # junta los tokens en un texto separado por espacios


# Split
def split_data(df, test_size=0.2, seed=SEED):
    """Train/test estratificado por clase."""
    # test_size=0.2 -> 20 % para test (200) y 80 % para train (800)
    # stratify=label -> mantiene la proporción 50/50 en ambos conjuntos
    # random_state -> el mismo split para todos los integrantes
    return train_test_split(df, test_size=test_size, stratify=df["label"], random_state=seed)


# Calidad de datos y resumen del preprocesamiento
def data_quality(df, text_col="text", clean_col="clean"):
    """Chequeos básicos de calidad: nulos, duplicados y textos vacíos."""
    key = df[text_col].fillna("").str.strip().str.lower()
    report = {
        "rows": len(df),
        "null_cells": int(df.isna().sum().sum()),
        "duplicated_texts": int(key.duplicated().sum()),
        "empty_texts": int((key == "").sum()),
    }
    if clean_col in df.columns:
        report["empty_after_cleaning"] = int((df[clean_col].fillna("").str.strip() == "").sum())
    return pd.Series(report, name="value")


def preprocessing_summary(texts):
    """Compara texto original, tokenizado con stopwords y tokenizado sin stopwords."""
    texts = list(texts)
    variants = {
        "Original (split on whitespace)": [t.split() for t in texts],
        "Tokenized, with stopwords": [tokenize(t, remove_stopwords=False) for t in texts],
        "Tokenized, without stopwords": [tokenize(t, remove_stopwords=True) for t in texts],
    }
    rows = []
    for name, docs in variants.items():
        flat = [w for doc in docs for w in doc]
        rows.append({"variant": name,
                     "avg_length": round(len(flat) / len(docs), 2),
                     "total_tokens": len(flat),
                     "vocab_size": len(set(flat))})
    return pd.DataFrame(rows).set_index("variant")


def length_percentiles(texts, remove_stopwords=False, qs=(50, 90, 95, 99, 100)):
    """Percentiles de longitud en tokens; sirve para elegir max_len en RNN/LSTM."""
    lengths = pd.Series([len(tokenize(t, remove_stopwords)) for t in texts])
    return pd.Series({f"p{q}": int(lengths.quantile(q / 100)) for q in qs})