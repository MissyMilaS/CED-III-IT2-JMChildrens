# Datos

**Fuente:** GoEmotions (Demszky et al., ACL 2020), versión de Kaggle:
https://www.kaggle.com/datasets/debarshichanda/goemotions

## `raw/`
`goemotions_1.csv`, `goemotions_2.csv`, `goemotions_3.csv`: archivos originales sin modificar
(una fila por anotador, 211,225 filas en total).

## `processed/`
Generado por `notebooks/01_eda.ipynb` con las funciones de `src/preprocessing.py`:

1. Se agregan las anotaciones por comentario y se conservan las emociones con acuerdo de al menos 2 anotadores (54,263 comentarios).
2. Se aplica el mapeo oficial de emociones a sentimiento (positivo / negativo / ambiguo).
3. Se conservan solo los comentarios cuyas emociones pertenecen todas al grupo positivo o todas al negativo. Se descartan los neutrales, ambiguos y mixtos.
4. Se eliminan los textos duplicados y los que tienen etiquetas en conflicto.
5. Se toma una muestra balanceada de 500 positivos y 500 negativos (`SEED = 42`) → `balanced_1000.csv`.
6. Se hace un split estratificado 80/20 → `train.csv` (800) y `test.csv` (200).

Columnas: `id`, `text` (original), `group`, `label` (1 = positivo, 0 = negativo),
`clean` (en minúsculas, sin URLs, puntuación ni stopwords, conservando las negaciones), `n_words`, `n_tokens`.
