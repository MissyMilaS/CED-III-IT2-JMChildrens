"""Plots for the EDA and for model results. Each function saves its figure in outputs/figures/."""

import math                                   # log() para el cociente de log-odds de las palabras distintivas
from collections import Counter               # Counter cuenta cuántas veces aparece cada elemento (palabras, bigramas)
from pathlib import Path                      

import matplotlib.pyplot as plt               # base para crear figuras y ejes
import pandas as pd                           # tablas (DataFrames)
import seaborn as sns                         # gráficos estadísticos sobre matplotlib
from sklearn.metrics import confusion_matrix  # calcula la matriz de confusión

# __file__ = este archivo (src/visualize.py); .parents[1] = raíz del repo; de ahí se baja a outputs/figures
FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"
COLORS = {"negative": "#e76f51", "positive": "#2a9d8f"}   # un color fijo por clase, igual en todas las gráficas
ORDER = ["negative", "positive"]                          # orden en que aparecen las clases en todas las gráficas
# Colores de los 5 grupos del dataset filtrado (solo positive/negative se usan después para los modelos)
GROUP_COLORS = {"positive": "#2a9d8f", "negative": "#e76f51",
                "neutral": "#adb5bd", "ambiguous": "#e9c46a", "mixed": "#8d99ae"}


def _save(fig, name):
    """Guarda la figura en outputs/figures/ (el guion bajo indica que es una función interna)."""
    FIG_DIR.mkdir(parents=True, exist_ok=True)                  # crea la carpeta si no existe
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")   # 150 DPI y sin márgenes sobrantes


def _word_counts(df, group, text_col):
    """Word frequencies of one class (texts must be already cleaned)."""
    # recorre los textos de la clase, los parte por espacios y cuenta cada palabra
    return Counter(w for t in df.loc[df["group"] == group, text_col] for w in t.split())


# Data quality and class balance
def plot_filter_funnel(funnel, name="eda_filter_funnel.png"):
    """Horizontal bars with the comments left after each filtering step (log scale)."""
    fig, ax = plt.subplots(figsize=(9, 4))                  # una sola gráfica de 9x4 pulgadas
    steps = funnel["step"].tolist()[::-1]                   # [::-1] invierte el orden: el primer paso queda arriba
    values = funnel["comments"].tolist()[::-1]              # mismos valores, mismo orden invertido
    ax.barh(steps, values, color="#457b9d")                 # barras horizontales, una por paso
    ax.set_xscale("log")                                    # escala log: 211,225 y 1,000 no caben en una escala lineal
    ax.set_xlim(right=max(values) * 3)                      # deja espacio a la derecha para escribir los números
    for i, v in enumerate(values):                          # recorre cada barra con su posición i y su valor v
        ax.text(v * 1.08, i, f"{v:,}", va="center", fontsize=9)   # número con separador de miles junto a la barra
    ax.set_xlabel("Comments (log scale)")
    ax.set_title("Comments left after each filtering step")
    fig.tight_layout()                                      # evita que textos y títulos se encimen
    _save(fig, name)
    return fig


def plot_group_distribution(df, name="eda_group_distribution.png"):
    """Distribution of the five sentiment groups in the full filtered dataset (before balancing)."""
    fig, ax = plt.subplots(figsize=(6, 4))
    counts = df["group"].value_counts()                     # cuántos comentarios hay en cada grupo, de mayor a menor
    ax.bar(counts.index, counts.values, color=[GROUP_COLORS[g] for g in counts.index])   # un color por grupo
    for i, v in enumerate(counts.values):                   # escribe el conteo encima de cada barra
        ax.text(i, v + counts.max() * 0.01, f"{v:,}", ha="center", fontsize=9)   # 1 % de la barra más alta como separación
    ax.set_title("Sentiment groups in the filtered GoEmotions dataset")
    ax.set_ylabel("Comments")
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_class_balance(df, name="eda_class_balance.png"):
    """Number of comments per class in the balanced dataset."""
    fig, ax = plt.subplots(figsize=(4, 4))
    counts = df["group"].value_counts().reindex(ORDER)      # cuenta por clase y las ordena según ORDER
    labels = [str(g) for g in counts.index]                 # nombres de las clases como texto
    values = [int(v) for v in counts.values]                # conteos como enteros normales de Python
    ax.bar(labels, values, color=[COLORS[g] for g in labels])
    for i, v in enumerate(values):
        ax.text(i, v + 5, str(v), ha="center")              # valor centrado encima de la barra
    ax.set_title("Class balance")
    ax.set_ylabel("Comments")
    _save(fig, name)
    return fig


# Text length
def plot_length_distribution(df, col="n_words", vline=None, name="eda_length.png"):
    """Histogram + boxplot of text length per class. `vline` marks a length (e.g. the p95 used as max_len)."""
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))         # dos gráficas lado a lado: axes[0] y axes[1]
    # histograma por clase: bins de 2 en 2 (0 a 30), capas superpuestas y semitransparentes
    sns.histplot(data=df, x=col, hue="group", hue_order=ORDER, palette=COLORS,
                 bins=range(0, 32, 2), multiple="layer", alpha=0.6, ax=axes[0])
    if vline is not None:                                   # línea vertical opcional (por ejemplo, el p95 para max_len)
        axes[0].axvline(vline, color="black", linestyle="--", linewidth=1, label=f"p95 = {vline}")
        axes[0].legend()
    axes[0].set_title("Length distribution")
    axes[0].set_xlabel("Number of words")
    axes[0].set_ylabel("Comments")
    # diagrama de caja por clase; hue + legend=False colorea cada caja sin repetir la leyenda
    sns.boxplot(data=df, x="group", y=col, order=ORDER, palette=COLORS,
                hue="group", legend=False, ax=axes[1])
    axes[1].set_title("Length by class")
    axes[1].set_xlabel("Class")
    axes[1].set_ylabel("Number of words")
    fig.tight_layout()
    _save(fig, name)
    return fig


# Vocabulary
def plot_top_words(df, text_col="clean", k=15, name="eda_top_words.png"):
    """Top-k most frequent words per class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))         # un panel por clase
    for ax, g in zip(axes, ORDER):                          # empareja cada panel con su clase
        # las k palabras más comunes de la clase g, en una tabla de dos columnas
        top = pd.DataFrame(_word_counts(df, g, text_col).most_common(k), columns=["word", "frequency"])
        sns.barplot(data=top, x="frequency", y="word", color=COLORS[g], ax=ax)   # barras horizontales
        ax.set_title(f"Top {k} words: {g}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("")                                   # sin etiqueta en Y: las palabras ya se explican solas
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_top_bigrams(df, text_col="clean", k=10, name="eda_top_bigrams.png"):
    """Top-k most frequent bigrams (pairs of consecutive words) per class."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, g in zip(axes, ORDER):
        counts = Counter()                                  # contador vacío de bigramas
        for t in df.loc[df["group"] == g, text_col]:        # recorre los textos de la clase
            w = t.split()                                   # lista de palabras del texto
            counts.update(zip(w, w[1:]))                    # zip(w, w[1:]) arma los pares consecutivos: (w0,w1), (w1,w2), ...
        # une cada par en un solo texto ("not good") para usarlo como etiqueta
        top = pd.DataFrame([(" ".join(b), c) for b, c in counts.most_common(k)], columns=["bigram", "frequency"])
        sns.barplot(data=top, x="frequency", y="bigram", color=COLORS[g], ax=ax)
        ax.set_title(f"Top {k} bigrams: {g}")
        ax.set_xlabel("Frequency")
        ax.set_ylabel("")
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_distinctive_words(df, text_col="clean", k=12, min_count=5, name="eda_distinctive_words.png"):
    """Words most associated with each class, by smoothed log-odds ratio (positive vs negative).

    Frequent words shared by both classes (e.g. 'like', 'people') score near 0, so this shows
    what actually separates the classes, which the raw top-words plot cannot.
    """
    pos, neg = _word_counts(df, "positive", text_col), _word_counts(df, "negative", text_col)
    n_pos, n_neg = sum(pos.values()), sum(neg.values())     # total de palabras de cada clase (para normalizar)
    rows = []
    for w in set(pos) | set(neg):                           # todas las palabras que aparecen en alguna clase
        if pos[w] + neg[w] < min_count:                     # se ignoran las muy raras: su puntaje sería ruido
            continue
        # log-odds = log(frecuencia relativa en positivos) - log(frecuencia relativa en negativos)
        # > 0: la palabra es más típica de positivos; < 0: de negativos; ~ 0: se usa igual en ambas
        # el +1 (suavizado) evita log(0) cuando la palabra no aparece en una de las clases
        score = math.log((pos[w] + 1) / (n_pos + 1)) - math.log((neg[w] + 1) / (n_neg + 1))
        rows.append((w, score))
    scores = pd.DataFrame(rows, columns=["word", "log_odds"]).sort_values("log_odds")   # de más negativa a más positiva
    top = pd.concat([scores.head(k), scores.tail(k)])       # las k más negativas y las k más positivas
    fig, ax = plt.subplots(figsize=(7, 6))
    ax.barh(top["word"], top["log_odds"],
            color=[COLORS["negative"] if s < 0 else COLORS["positive"] for s in top["log_odds"]])   # color según el signo
    ax.axvline(0, color="black", linewidth=0.8)             # línea en 0 = palabra neutra entre clases
    ax.set_xlabel("Log-odds ratio  (← negative | positive →)")
    ax.set_title(f"Most distinctive words per class (min count = {min_count})")
    fig.tight_layout()
    _save(fig, name)
    return fig


def plot_wordclouds(df, text_col="clean", name="eda_wordclouds.png"):
    """One word cloud per class. Requires the `wordcloud` package (see requirements.txt)."""
    try:
        from wordcloud import WordCloud  # se importa aquí para que el resto del módulo funcione sin esta librería
    except ImportError as e:
        raise ImportError("Install wordcloud first: pip install wordcloud") from e
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    for ax, g in zip(axes, ORDER):
        # el tamaño de cada palabra es proporcional a su frecuencia; random_state fija la disposición
        wc = WordCloud(width=800, height=500, background_color="white", colormap="Greens" if g == "positive" else "Reds",
                       random_state=42).generate_from_frequencies(_word_counts(df, g, text_col))
        ax.imshow(wc, interpolation="bilinear")             # dibuja la nube como imagen
        ax.set_title(g.capitalize())
        ax.axis("off")                                      # sin ejes: no aportan nada en una nube
    fig.tight_layout()
    _save(fig, name)
    return fig


# Model results
def plot_confusion_matrix(y_true, y_pred, title="Confusion matrix", name="confusion_matrix.png"):
    """Confusion matrix (rows = actual, columns = predicted)."""
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])    # 0 = negative, 1 = positive
    fig, ax = plt.subplots(figsize=(4, 4))
    # mapa de calor con los conteos escritos (annot, fmt="d" = enteros) y sin barra de color
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=ORDER, yticklabels=ORDER, ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Predicted")                              # columnas = lo que predijo el modelo
    ax.set_ylabel("Actual")                                 # filas = la etiqueta real
    fig.tight_layout()
    _save(fig, name)
    return fig