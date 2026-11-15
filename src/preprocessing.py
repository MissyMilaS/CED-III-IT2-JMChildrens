"""
Preparación de datos para GoEmotions (sentimiento positivo vs. negativo).

Pasos: carga de los CSV crudos -> agregación por comentario (>= 2 evaluadores)
-> asignación de sentimiento -> dataset balanceado -> limpieza de texto -> split.
"""

import re
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

# ---------------------------------------------------------------- Constantes
SEED = 42
RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

POSITIVE = ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"]
NEGATIVE = ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"]
AMBIGUOUS = ["realization", "surprise", "curiosity", "confusion"]
EMOTIONS = POSITIVE + NEGATIVE + AMBIGUOUS + ["neutral"]

# Stopwords y tokenizador
try:
    import nltk
    try:
        nltk.data.find("corpora/stopwords")
    except LookupError:
        nltk.download("stopwords", quiet=True)   # solo la primera vez
    from nltk.corpus import stopwords
    STOPWORDS = set(stopwords.words("english"))
except Exception:
    print("Aviso: usando stopwords de scikit-learn (nltk no disponible)")
    from sklearn.feature_extraction.text import ENGLISH_STOP_WORDS
    STOPWORDS = set(ENGLISH_STOP_WORDS)

# Las negaciones cambian el sentimiento ("not good"), así que no se eliminan
NEGATIONS = {"no", "not", "nor", "never", "nothing", "none", "neither", "cannot"}
NEGATIONS |= {w for w in STOPWORDS if "n't" in w}   # didn't, isn't, wasn't, ...
STOPWORDS_SENT = STOPWORDS - NEGATIONS

# Conserva [name] y [religion] (marcadores del dataset), palabras con apóstrofe y números
TOKEN_RE = re.compile(r"\[name\]|\[religion\]|[a-z]+(?:'[a-z]+)?|\d+")


# Carga
def load_raw():
    """Carga y une los tres CSV crudos (una fila por evaluador y comentario)."""
    files = [RAW_DIR / f"goemotions_{i}.csv" for i in (1, 2, 3)]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)


# Agregación y sentimiento
def aggregate_filtered(raw):
    """Una emoción es válida si al menos 2 evaluadores la marcaron.
    Se descartan los comentarios que quedan sin ninguna emoción."""
    raw = raw.drop_duplicates(subset=["id", "rater_id"])
    votes = raw.groupby("id")[EMOTIONS].sum()
    text = raw.drop_duplicates("id").set_index("id")["text"]
    agreed = (votes >= 2).astype(int)
    agreed = agreed[agreed.sum(axis=1) > 0]
    return agreed.join(text)


def assign_sentiment(df):
    """Clasifica cada comentario como positive, negative, neutral, ambiguous o mixed."""
    pos = df[POSITIVE].sum(axis=1) > 0
    neg = df[NEGATIVE].sum(axis=1) > 0
    amb = df[AMBIGUOUS].sum(axis=1) > 0
    neu = df["neutral"] > 0
    n_groups = pos.astype(int) + neg.astype(int) + amb.astype(int) + neu.astype(int)

    out = df.copy()
    out["group"] = "mixed"
    out.loc[(n_groups == 1) & pos, "group"] = "positive"
    out.loc[(n_groups == 1) & neg, "group"] = "negative"
    out.loc[(n_groups == 1) & amb, "group"] = "ambiguous"
    out.loc[(n_groups == 1) & neu, "group"] = "neutral"
    return out


# Dataset balanceado
def build_balanced(df, n=500, seed=SEED):
    """Solo positivos y negativos, sin duplicados, n comentarios por clase."""
    d = df[df["group"].isin(["positive", "negative"])].copy()
    d["text_key"] = d["text"].str.strip().str.lower()
    # textos repetidos con etiquetas distintas: se descartan por contradictorios
    conflict = d.groupby("text_key")["group"].nunique() > 1
    d = d[~d["text_key"].isin(conflict[conflict].index)]
    d = d.drop_duplicates("text_key")

    pos = d[d["group"] == "positive"].sample(n, random_state=seed)
    neg = d[d["group"] == "negative"].sample(n, random_state=seed)
    out = pd.concat([pos, neg]).sample(frac=1, random_state=seed)
    out["label"] = (out["group"] == "positive").astype(int)   # 1 = positivo
    return out[["text", "group", "label"]].reset_index()


# Limpieza de texto
def tokenize(text, remove_stopwords=True):
    """Minúsculas, sin URLs, puntuación ni emojis; opcionalmente sin stopwords."""
    t = text.lower().replace("\u2019", "'").replace("\u2018", "'")
    t = re.sub(r"https?://\S+", " ", t)
    tokens = TOKEN_RE.findall(t)
    if remove_stopwords:
        tokens = [w for w in tokens if w not in STOPWORDS_SENT]
    return tokens


def clean_text(text, remove_stopwords=True):
    return " ".join(tokenize(text, remove_stopwords))


# Split
def split_data(df, test_size=0.2, seed=SEED):
    """Train/test estratificado por clase."""
    return train_test_split(df, test_size=test_size,stratify=df["label"], random_state=seed)