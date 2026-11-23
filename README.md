# CED-III-IT2-JMChildrens

Proyecto de análisis y clasificación de emociones en texto basado en el conjunto de datos **GoEmotions**.

El repositorio incluye:

- Análisis exploratorio de datos.
- Preprocesamiento y preparación de textos.
- Modelos de referencia.
- Redes neuronales densas y recurrentes.
- Una extensión experimental con Transformers.
- Métricas, gráficos y modelos entrenados.

## Integrantes

Sarah Rodriguez A00411232
Maria Camila Salamanca A00411234
Nicholas Solis A00411205
Santiago Cabal A00412508
Heidy Hernandez A00411232


## Tecnologías

- Python
- Jupyter Notebook
- pandas y NumPy
- scikit-learn
- PyTorch
- NLTK y spaCy
- Matplotlib y Seaborn
- Hugging Face Transformers y Datasets
- WordCloud

## Estructura del proyecto

```text
.
├── data/
│   ├── raw/              # Copias originales de GoEmotions
│   └── processed/        # Datos preparados para entrenamiento
├── docs/                 # Informe, presentación y referencias
├── logs/                 # Registro de experimentos y entrenamiento
├── notebooks/            # Análisis y experimentos
├── outputs/
│   ├── figures/          # Gráficos generados
│   ├── metrics/          # Métricas y resultados
│   └── saved_models/     # Modelos guardados
├── prompts/              # Interacciones y registros de prompts
├── src/                  # Funciones y módulos reutilizables
├── environment.yml       # Entorno de Conda
└── requirements.txt      # Dependencias de Python
```

## Flujo de trabajo

Ejecuta los notebooks en el siguiente orden:

1. `notebooks/01_eda.ipynb` — Exploración y análisis descriptivo.
2. `notebooks/02_baseline_model.ipynb` — Modelos de referencia.
3. `notebooks/03_dense_rnn_lstm.ipynb` — Redes densas, RNN, LSTM y BiLSTM.
4. `notebooks/04_transformer_extension.ipynb` — Experimentos con Transformers.

Los módulos de `src/` contienen lógica reutilizable para el preprocesamiento, los modelos, el entrenamiento, la evaluación y la visualización.

## Configuración del entorno

Se recomienda Python 3.10.

### Opción 1: Conda

```bash
conda env create -f environment.yml
conda activate goemotions_env
python -m ipykernel install --user --name goemotions_env
```

### Opción 2: pip

```bash
python -m venv .venv
```

Activación en Windows:

```powershell
.venv\Scripts\activate
```

Instalación de dependencias:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Inicia Jupyter:

```bash
jupyter notebook
```

Abre los notebooks desde la carpeta `notebooks/` y ejecútalos en orden.

## Datos

El proyecto utiliza el conjunto de datos GoEmotions.

Los archivos originales deben conservarse sin modificaciones dentro de `data/raw/`. Las entradas procesadas se generan dentro de `data/processed/` mediante los notebooks o los módulos de `src/`.

Los CSV originales de GoEmotions (Kaggle) ya están incluidos en `data/raw/`, y los splits procesados (`train.csv` 800 / `test.csv` 200) en `data/processed/`. Ver `data/README.md`.

## Resultados (test, 200 comentarios)

| Modelo | Accuracy | Precision | Recall | F1 | Kappa |
|---|---|---|---|---|---|
| Baseline (Dummy stratified) | 0.500 | 0.500 | 0.510 | 0.505 | 0.000 |
| Dense NN | 0.648 ± 0.026 | 0.723 ± 0.037 | 0.496 ± 0.129 | 0.575 ± 0.082 | 0.296 ± 0.052 |
| Vanilla RNN | 0.558 ± 0.032 | 0.577 ± 0.045 | 0.506 ± 0.142 | 0.524 ± 0.065 | 0.116 ± 0.064 |
| LSTM | 0.625 ± 0.043 | 0.720 ± 0.073 | 0.422 ± 0.130 | 0.519 ± 0.091 | 0.250 ± 0.086 |
| BiLSTM (extra) | 0.677 ± 0.039 | 0.695 ± 0.068 | 0.670 ± 0.084 | 0.674 ± 0.019 | 0.354 ± 0.078 |
| **DistilBERT** (1 semilla) | **0.900** | **0.908** | **0.890** | **0.899** | **0.800** |

Los modelos Dense/RNN/LSTM/BiLSTM reportan media ± desviación estándar sobre 5 semillas (777–781). DistilBERT se reporta con una sola ejecución (semilla 42); su validación es el 15% de train.

### Ejecutar solo la extensión Transformer

```bash
python src/run_transformer.py --train          # entrena y evalúa DistilBERT (~3 min en CPU)
python src/run_transformer.py --update_plots   # regenera comparison.csv y la figura comparativa
```

## Otros resultados

Los resultados de experimentos disponibles incluyen:

- Gráficos de distribución y balance de clases.
- Matrices de confusión.
- Curvas de entrenamiento.
- Comparaciones de métricas.
- Registros de búsqueda de hiperparámetros.
- Pesos de los modelos entrenados.

Estos archivos se encuentran dentro de `outputs/`.

 
## Documentación

Información adicional disponible en `docs/`:

- `report.pdf`: informe final (inglés, 4 páginas).
- `presentation.pdf`: presentación final.
- Análisis comparativo.
- Archivo de referencias bibliográficas.
