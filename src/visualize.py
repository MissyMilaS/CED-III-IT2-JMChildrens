"""Visualizaciones de datos y resultados experimentales."""  # Docstring del módulo: describe su propósito general.
"""Gráficas del EDA y de resultados. Cada función guarda su figura en outputs/figures/."""  # Segundo docstring (redundante, actúa como cadena literal).
from collections import Counter  # Importa Counter para contar la frecuencia de elementos (como palabras)[cite: 1].
from pathlib import Path  # Importa Path para manejar rutas de archivos y carpetas de forma multiplataforma[cite: 1].

import matplotlib.pyplot as plt  # Importa matplotlib para crear figuras, ejes y guardar gráficos[cite: 1].
import pandas as pd  # Importa pandas para manipular datos tabulares (DataFrames)[cite: 1].
import seaborn as sns  # Importa seaborn para generar gráficos estadísticos atractivos sobre matplotlib[cite: 1].
from sklearn.metrics import confusion_matrix  # Importa la función para calcular la matriz de confusión[cite: 1].

FIG_DIR = Path(__file__).resolve().parents[1] / "outputs" / "figures"  # Define la ruta ../outputs/figures/ relativa a este archivo[cite: 1].
COLORS = {"negative": "#e76f51", "positive": "#2a9d8f"}  # Mapea cada clase a un color hexadecimal específico[cite: 1].
ORDER = ["negative", "positive"]  # Define el orden estándar en que aparecerán las clases en las gráficas[cite: 1].

def _save(fig, name):  # Función auxiliar interna para guardar una figura en disco[cite: 1].
    FIG_DIR.mkdir(parents=True, exist_ok=True)  # Crea la carpeta destino (y carpetas padres) si aún no existen[cite: 1].
    fig.savefig(FIG_DIR / name, dpi=150, bbox_inches="tight")  # Guarda la figura a 150 DPI recortando márgenes extra[cite: 1].

def plot_class_balance(df, name="eda_class_balance.png"):  # Grafica cuántos registros hay por clase en el DataFrame[cite: 1].
    fig, ax = plt.subplots(figsize=(4, 4))  # Crea una figura y un eje de tamaño 4x4 pulgadas[cite: 1].
    counts = df["group"].value_counts().reindex(ORDER)  # Cuenta ocurrencias de "group" y las ordena según ORDER[cite: 1].
    labels = [str(g) for g in counts.index]  # Convierte las etiquetas de las clases a texto[cite: 1].
    values = [int(v) for v in counts.values]  # Convierte los conteos a enteros estándar de Python[cite: 1].
    ax.bar(labels, values, color=[COLORS[g] for g in labels])  # Dibuja el gráfico de barras con el color de cada clase[cite: 1].
    for i, v in enumerate(values):  # Recorre cada barra junto con su índice y valor[cite: 1].
        ax.text(i, v + 5, str(v), ha="center")  # Escribe el valor numérico centrado justo encima de cada barra[cite: 1].
    ax.set_title("Balance de clases")  # Asigna el título principal de la gráfica[cite: 1].
    ax.set_ylabel("Comentarios")  # Etiqueta el eje Y indicando que representa cantidad de comentarios[cite: 1].
    _save(fig, name)  # Guarda la figura en outputs/figures/ con el nombre indicado[cite: 1].
    return fig  # Devuelve el objeto figura de matplotlib[cite: 1].

def plot_length_distribution(df, col="n_words", name="eda_length.png"):  # Grafica longitud de textos por clase[cite: 1].
    """Histograma + boxplot de la longitud (en palabras) por clase."""  # Documentación de la función[cite: 1].
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))  # Crea 2 subgráficas lado a lado (1 fila, 2 columnas)[cite: 1].
    sns.histplot(data=df, x=col, hue="group", hue_order=ORDER, palette=COLORS,  # Dibuja histograma segmentado por clase[cite: 1].
                 bins=range(0, 32, 2), multiple="layer", alpha=0.6, ax=axes[0])  # Bins de 2 en 2 (0 a 30), superpuestos y semitransparentes[cite: 1].
    axes[0].set_title("Distribución de la longitud")  # Título del histograma (izquierda)[cite: 1].
    axes[0].set_xlabel("Número de palabras")  # Etiqueta del eje X del histograma[cite: 1].
    sns.boxplot(data=df, x="group", y=col, order=ORDER, palette=COLORS,  # Dibuja diagrama de caja por clase[cite: 1].
                hue="group", legend=False, ax=axes[1])  # Colorea por clase sin duplicar leyenda en el eje derecho[cite: 1].
    axes[1].set_title("Longitud por clase")  # Título del boxplot (derecha)[cite: 1].
    axes[1].set_xlabel("Clase")  # Etiqueta del eje X del boxplot[cite: 1].
    axes[1].set_ylabel("Número de palabras")  # Etiqueta del eje Y del boxplot[cite: 1].
    fig.tight_layout()  # Ajusta automáticamente los espacios para que textos y títulos no se encimen[cite: 1].
    _save(fig, name)  # Guarda la figura combinada en disco[cite: 1].
    return fig  # Retorna la figura creada[cite: 1].


def plot_top_words(df, text_col="clean", k=15, name="eda_top_words.png"):  # Grafica las k palabras más frecuentes por clase[cite: 1].
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))  # Crea 2 paneles horizontales (uno para cada clase)[cite: 1].
    for ax, g in zip(axes, ORDER):  # Empareja cada eje con su clase correspondiente en ORDER[cite: 1].
        words = Counter(w for t in df.loc[df["group"] == g, text_col] for w in t.split())  # Tokeniza por espacios y cuenta palabras de la clase g[cite: 1].
        top = pd.DataFrame(words.most_common(k), columns=["palabra", "frecuencia"])  # Crea un DataFrame con las k palabras más comunes[cite: 1].
        sns.barplot(data=top, x="frecuencia", y="palabra", color=COLORS[g], ax=ax)  # Dibuja barras horizontales de frecuencia[cite: 1].
        ax.set_title(f"Top {k} palabras: {g}")  # Pone el título indicando el límite k y la clase[cite: 1].
    fig.tight_layout()  # Optimiza el espaciado entre ambos paneles[cite: 1].
    _save(fig, name)  # Guarda la gráfica en outputs/figures/[cite: 1].
    return fig  # Devuelve la figura[cite: 1].

def plot_confusion_matrix(y_true, y_pred, title="Matriz de confusión", name="confusion_matrix.png"):  # Grafica la matriz de confusión[cite: 1].
    """Matriz de confusión (filas = real, columnas = predicho). Guarda la figura en outputs/figures/."""  # Docstring de la función[cite: 1].
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])  # Calcula la matriz de confusión para etiquetas numéricas 0 y 1[cite: 1].
    fig, ax = plt.subplots(figsize=(4, 4))  # Crea una figura cuadrada de 4x4 pulgadas[cite: 1].
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,  # Dibuja mapa de calor con números enteros sin barra lateral de color[cite: 1].
                xticklabels=ORDER, yticklabels=ORDER, ax=ax)  # Etiqueta ejes X e Y con los nombres de las clases[cite: 1].
    ax.set_title(title)  # Asigna el título recibido por parámetro[cite: 1].
    ax.set_xlabel("Predicho")  # Indica que las columnas representan las predicciones del modelo[cite: 1].
    ax.set_ylabel("Real")  # Indica que las filas representan las etiquetas reales[cite: 1].
    fig.tight_layout()  # Ajusta los márgenes de la figura[cite: 1].
    _save(fig, name)  # Guarda la imagen en outputs/figures/[cite: 1].
    return fig  # Retorna la figura generada[cite: 1].