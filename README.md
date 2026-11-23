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

- Heidy Yuliana Hernández Huila — A00411232
- María Camila Salamanca — A00411234
- Nicholas Solís — A0041--05
- Sarah Rodríguez — A00411438
- Santiago Cabal — A00



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

El repositorio no descarga automáticamente el conjunto de datos original. Comprueba que los archivos requeridos estén disponibles antes de ejecutar los notebooks.

## Resultados

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

- Informe académico.
- Presentación.
- Análisis comparativo.
- Archivo de referencias bibliográficas.
