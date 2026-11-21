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