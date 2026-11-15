#Limpieza de texto, tokenización y preparación de representaciones

from pathlib import Path
import pandas as pd

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"

def load_raw():
    files = [RAW_DIR / f"goemotions_{i}.csv" for i in (1, 2, 3)]
    return pd.concat([pd.read_csv(f) for f in files], ignore_index=True)

POSITIVE = ["amusement", "excitement", "joy", "love", "desire", "optimism", "caring", "pride", "admiration", "gratitude", "relief", "approval"]
NEGATIVE = ["fear", "nervousness", "remorse", "embarrassment", "disappointment", "sadness", "grief", "disgust", "anger", "annoyance", "disapproval"]
AMBIGUOUS = ["realization", "surprise", "curiosity", "confusion"]
EMOTIONS = POSITIVE + NEGATIVE + AMBIGUOUS + ["neutral"]

def aggregate_filtered(raw):
    #Una emoción es válida si al menos 2 evaluadores la marcaron. Se descartan los comentarios que quedan sin ninguna emoción
    raw = raw.drop_duplicates(subset=["id", "rater_id"])
    votes = raw.groupby("id")[EMOTIONS].sum()
    text = raw.drop_duplicates("id").set_index("id")["text"]
    agreed = (votes >= 2).astype(int)
    agreed = agreed[agreed.sum(axis=1) > 0]
    return agreed.join(text)

def assign_sentiment(df):
    #Clasifica cada comentario como positive, negative, neutral, ambiguous o mixed
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