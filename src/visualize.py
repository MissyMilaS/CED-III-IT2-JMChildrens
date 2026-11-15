"""Visualizaciones de datos y resultados experimentales."""
"""Gráficas del EDA y de resultados. Cada función guarda su figura en outputs/figures/."""
from collections import Counter
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
COLORS = {"negative": "#e76f51", "positive": "#2a9d8f"}
ORDER = ["negative", "positive"]

def _save(fig, name):
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")

def plot_class_balance(df, name="eda_class_balance.png"):
    fig, ax = plt.subplots(figsize=(4, 4))
    counts = df["group"].value_counts().reindex(ORDER)
    labels = [str(g) for g in counts.index]
    values = [int(v) for v in counts.values]
    ax.bar(labels, values, color=[COLORS[g] for g in labels])
    for i, v in enumerate(values):
        ax.text(i, v + 5, str(v), ha="center")
    ax.set_title("Balance de clases")
    ax.set_ylabel("Comentarios")
    _save(fig, name)
    return fig

def plot_length_distribution(df, col="n_words", name="eda_length.png"):
    """Histograma + boxplot de la longitud (en palabras) por clase."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    sns.histplot(data=df, x=col, hue="group", hue_order=ORDER, palette=COLORS,
                 bins=range(0, 32, 2), multiple="layer", alpha=0.6, ax=axes[0])
    axes[0].set_title("Distribución de la longitud")
    axes[0].set_xlabel("Número de palabras")
    sns.boxplot(data=df, x="group", y=col, order=ORDER, palette=COLORS,
                hue="group", legend=False, ax=axes[1])
    axes[1].set_title("Longitud por clase")
    axes[1].set_xlabel("Clase")
    axes[1].set_ylabel("Número de palabras")
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_top_words(df, text_col="clean", k=15, name="eda_top_words.png"):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, g in zip(axes, ORDER):
        words = Counter(w for t in df.loc[df["group"] == g, text_col] for w in t.split())
        top = pd.DataFrame(words.most_common(k), columns=["palabra", "frecuencia"])
        sns.barplot(data=top, x="frecuencia", y="palabra", color=COLORS[g], ax=ax)
        ax.set_title(f"Top {k} palabras: {g}")
    fig.tight_layout()
    _save(fig, name)
    return fig