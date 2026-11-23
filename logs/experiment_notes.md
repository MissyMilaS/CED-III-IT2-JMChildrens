# Notas de experimentos

## Baseline 

**Notebook:** `notebooks/02_baseline_model.ipynb`
**Datos:** `data/processed/train.csv` (800) y `test.csv` (200), generados en `01_eda.ipynb`. Sin re-split ni re-preprocesamiento.
**Semilla:** 42

### Parámetros
- Modelo: `sklearn.dummy.DummyClassifier`
- Estrategias probadas: `most_frequent`, `prior`, `stratified`, `uniform` (`random_state=42`)
- Entrada: columna `clean`, ignorada por el modelo
- Etiqueta: `label` (1 = positivo, 0 = negativo)

### Resultados (test, 200 comentarios)
| Estrategia | Accuracy | Precision | Recall | F1 | Kappa |
|---|---|---|---|---|---|
| most_frequent | 0.50 | 0.00 | 0.00 | 0.000 | 0.00 |
| prior | 0.50 | 0.00 | 0.00 | 0.000 | 0.00 |
| stratified | 0.50 | 0.50 | 0.51 | 0.505 | 0.00 |
| uniform | 0.48 | 0.48 | 0.48 | 0.480 | -0.04 |

### Decisiones
- Baseline oficial: `stratified`. Las otras estrategias dan métricas degeneradas o casi idénticas.
- Se usan las mismas funciones de `src/evaluate.py` (`compute_metrics`, `save_metrics`) para que todos los modelos guarden el mismo formato JSON en `outputs/metrics/`.

### Observaciones
- Con empate entre clases, `most_frequent` y `prior` eligen la clase 0 (negativo) y predicen siempre negativo.
- El azar da ~50% de accuracy y kappa ≈ 0, que es el piso para comparar los demás modelos.

### Archivos generados
- `outputs/metrics/baseline_dummy.json`
- `outputs/figures/baseline_confusion_<estrategia>.png` (4 figuras)

## Modelos neuronales (Dense, RNN vanilla, LSTM y BiLSTM)

### Datos y configuración
- **Notebook:** `notebooks/03_dense_rnn_lstm.ipynb`
- **Código:** `src/models.py` y `src/train.py`; métricas de `src/evaluate.py`
- **Datos:** `data/processed/train.csv` (800 filas) y `data/processed/test.csv` (200 filas), generados por el split estratificado existente. Train contiene 400 comentarios negativos y 400 positivos; test contiene 100 negativos y 100 positivos.
- **Texto usado:** columna `clean` para la ejecución principal; `text` se usa en la ablación con stopwords.
- **Semilla base:** 777.

### Preprocesamiento para redes neuronales
- El texto limpio se divide por espacios y se convierte en secuencias de ids.
- `MAX_VOCAB = 5000`: tamaño máximo configurado; el vocabulario efectivo construido con los 800 comentarios de train tiene **2,490 palabras, incluidos `<pad>` y `<unk>`**.
- `MAX_LEN = 30`: secuencias recortadas a 30 tokens.
- `<pad>` tiene id `0`; `<unk>` tiene id `1`.
- El vocabulario se construye exclusivamente con train. Las palabras no vistas de validation/test se codifican como `<unk>`.

### Arquitecturas
| Modelo | Arquitectura |
|---|---|
| Dense | `Embedding -> promedio de embeddings reales -> Linear(ReLU) -> Dropout -> Linear(1 logit)`; `n_params = 144065` |
| RNN vanilla | `Embedding -> RNN tanh unidireccional -> estado final -> Dropout -> Linear(1 logit)`; `n_params = 166913` |
| LSTM | `Embedding -> LSTM unidireccional -> estado final -> Dropout -> Linear(1 logit)`; `n_params = 154529` |
| BiLSTM | `Embedding -> LSTM bidireccional -> estados finales concatenados -> Dropout -> Linear(1 logit)`; `n_params = 167105`. Modelo extra no pedido por el enunciado. |

### Procedimiento de entrenamiento
- Optimizador: Adam.
- Función de pérdida: `BCEWithLogitsLoss`.
- Clipping de gradientes: norma máxima `1.0`.
- Early stopping sobre validation loss con **paciencia 5** y máximo de **30 épocas**.
- Los pesos restaurados son los de la mejor época de validación.
- `emb_dim = 64` y `dropout = 0.5` son fijos.
- El test nunca participa en el vocabulario, el grid ni las decisiones de early stopping. El tuning usa únicamente train y el test se evalúa una vez por semilla al final.

### Tuning
- Grid exacto:
  - `hidden`: 32, 64, 128
  - `batch_size`: 16, 32, 64
  - `lr`: 0.001, 0.003
- 18 configuraciones por modelo.
- Validación cruzada: `StratifiedKFold(n_splits=3, shuffle=True, random_state=777)`, aplicada solo sobre train.

| Modelo | Mejor hidden | Mejor batch_size | Mejor lr | val_acc | val_acc_std | val_f1 | best_epoch |
|---|---:|---:|---:|---:|---:|---:|---:|
| Dense | 32 | 64 | 0.003 | 0.6151 | 0.0500 | 0.6786 | 7.3 |
| RNN vanilla | 128 | 16 | 0.003 | 0.5425 | 0.0326 | 0.5293 | 1.0 |
| LSTM | 32 | 32 | 0.003 | 0.6663 | 0.0431 | 0.6699 | 4.3 |
| BiLSTM | 32 | 64 | 0.003 | 0.6325 | 0.0104 | 0.6859 | 5.0 |

Las mejores filas provienen de `outputs/metrics/gridsearch_<modelo>.csv`.

### Evaluación final
- **Número de semillas:** 5.
- **Semillas:** 777, 778, 779, 780 y 781.
- Para cada semilla, el 15 % de train se separa con split estratificado (`random_state=semilla`) para validar early stopping; el resto se usa para entrenamiento.
- Test se evalúa una vez por semilla y no interviene en el tuning.

| Modelo | Accuracy | Precision | Recall | F1 | Kappa | Segundos/corrida |
|---|---:|---:|---:|---:|---:|---:|
| Baseline Dummy | 0.5000 | 0.5000 | 0.5100 | 0.5050 | 0.0000 | - |
| Dense | 0.6670 ± 0.0614 | 0.7278 ± 0.1067 | 0.6040 ± 0.1297 | 0.6425 ± 0.0451 | 0.3340 ± 0.1227 | 0.3 |
| RNN vanilla | 0.5770 ± 0.0439 | 0.5769 ± 0.0528 | 0.7000 ± 0.1616 | 0.6165 ± 0.0514 | 0.1540 ± 0.0878 | 1.2 |
| LSTM | 0.6280 ± 0.0547 | 0.6225 ± 0.0572 | 0.7040 ± 0.0695 | 0.6553 ± 0.0183 | 0.2560 ± 0.1095 | 1.5 |
| BiLSTM | 0.6400 ± 0.0695 | 0.6639 ± 0.0795 | 0.6560 ± 0.1688 | 0.6381 ± 0.0778 | 0.2800 ± 0.1390 | 1.6 |

### Ablación de stopwords: clean vs full

| Modelo | Texto | Accuracy | Precision | Recall | F1 | Kappa | Segundos/corrida |
|---|---|---:|---:|---:|---:|---:|---:|
| RNN vanilla | clean | 0.5770 | 0.5769 | 0.7000 | 0.6165 | 0.1540 | 1.2 |
| RNN vanilla | full | 0.5570 | 0.5472 | 0.6660 | 0.5990 | 0.1140 | 1.5 |
| LSTM | clean | 0.6280 | 0.6225 | 0.7040 | 0.6553 | 0.2560 | 1.5 |
| LSTM | full | 0.6330 | 0.6429 | 0.5980 | 0.6151 | 0.2660 | 2.3 |
| BiLSTM | clean | 0.6400 | 0.6639 | 0.6560 | 0.6381 | 0.2800 | 1.6 |
| BiLSTM | full | 0.6170 | 0.6130 | 0.6980 | 0.6378 | 0.2340 | 2.4 |

### Observaciones
- LSTM gana en F1 con `0.6553`, seguido de Dense con `0.6425`; la diferencia es `0.0128`, menor que la desviación F1 de LSTM (`0.0183`) y de Dense (`0.0451`), por lo que no supera claramente la variabilidad entre semillas.
- Dense gana en Kappa con `0.3340`, frente a `0.2800` de BiLSTM y `0.2560` de LSTM. La ventaja sobre BiLSTM es `0.0540`, menor que las desviaciones de Kappa de Dense (`0.1227`) y BiLSTM (`0.1390`).
- Dense supera al baseline en Accuracy por `0.1670` (0.6670 vs 0.5000) y en Kappa por `0.3340`; el baseline no registra desviación entre semillas.
- RNN vanilla obtuvo el Recall más alto por modelo entre los principales (`0.7000`), pero tuvo el menor Accuracy (`0.5770`) y Kappa (`0.1540`) y además su mejor época fue `1.0` en el grid.
- En la ablación, conservar stopwords reduce el F1 de RNN vanilla de `0.6165` a `0.5990` y el Accuracy de `0.5770` a `0.5570`; en LSTM reduce el F1 de `0.6553` a `0.6151`, mientras Kappa cambia de `0.2560` a `0.2660`.
- Dense fue el más rápido con `0.3` segundos/corrida; LSTM tardó `1.5` y BiLSTM `1.6`. PENDIENTE: la brecha train/validation no puede resumirse numéricamente porque no existe un archivo persistido de historial por época.

### Reproducibilidad
1. Instalar dependencias:
   ```bash
   python -m pip install -r requirements.txt
   ```
2. Abrir y ejecutar en orden:
   ```bash
   jupyter notebook notebooks/03_dense_rnn_lstm.ipynb
   ```
- Métricas: `outputs/metrics/`
- Figuras: `outputs/figures/`
- Pesos: `outputs/saved_models/`
- Log de ejecución: `logs/training_logs.txt`


---

## Extensión con Transformers (DistilBERT)

**Notebook:** `notebooks/04_transformer_extension.ipynb`  
**Autor:** Nicholas Solís Pinilla  
**Datos:** `data/processed/train.csv` (800) y `test.csv` (200), mismos splits estratificados fijados desde el EDA.  
**Semilla:** 42  

### Configuración del Modelo y Tokenizador
- **Modelo Base:** `distilbert-base-uncased` (6 capas Transformer Encoder, 12 cabezas de atención, $d=768$).
- **Tokenización:** WordPiece con vocabulario de 30,522 sub-palabras (`AutoTokenizer`).
- **Longitud máxima:** `MAX_LEN = 64` (cubre el 99% de las secuencias de GoEmotions).
- **Parámetros entrenables:** 66,955,010.

### Hiperparámetros de Fine-Tuning
- **Optimizador / Algoritmo:** AdamW con peso decaído (`weight_decay = 0.01`).
- **Learning rate:** $3 \times 10^{-5}$ con scheduler lineal.
- **Batch size:** 16 en entrenamiento, 32 en evaluación.
- **Épocas:** 3 épocas completas (50 pasos por época con lote de 16 = 150 pasos de optimización).
- **Estrategia de evaluación:** Por época (`eval_strategy="epoch"`).

### Métricas de Evaluación
Se calcularon las 5 métricas oficiales mediante `compute_metrics` compatible con `src/evaluate.py`:
- Accuracy
- Precision
- Recall
- F1-Score
- Cohen's Kappa ($\kappa$)

### Vínculo Teórico con Máquinas de Turing
- **Mecanismo de Auto-Atención:** Opera con direccionamiento asociativo $O(1)$ directo sobre la cinta de entrada, superando el cuello de botella secuencial y desvanecimiento de gradiente característico de las RNN y LSTM ($O(T)$).
- **Preentrenamiento masivo:** Dota al modelo de representaciones semánticas profundas previas al ajuste supervisado, permitiendo generalizar con solidez incluso en datasets reducidos como esta muestra balanceada de 1,000 comentarios.

### Archivos Generados
- `outputs/metrics/distilbert.json`
- `outputs/metrics/comparison.csv` (actualizado con los 6 modelos)
- `outputs/figures/distilbert_confusion_matrix.png`
- `outputs/figures/distilbert_training_curves.png`
- `outputs/figures/all_models_metrics_comparison.png`
