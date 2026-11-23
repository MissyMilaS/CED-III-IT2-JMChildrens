# Interacciones con IAGen

Documenta aquí las interacciones con herramientas de inteligencia artificial
generativa utilizadas en la actividad.


---

## Sesión 22-11-2026: Planificación e Implementación de la Tarea 4 (Transformers / DistilBERT)

**Estudiante:** Nicholas Solís Pinilla (`NSP2007`)  
**Herramienta:** Asistente Inteligente Antigravity (DeepMind)  
**Nivel de uso:** Nivel 3 - Colaboración con IAG (con supervisión crítica y comprensión teórica).  

### Prompts Utilizados
1. *"tenemos esta tarea integradora que hacer en el .pdf y a mi me correspondes hacer el ultimo modelo de Transformers, lo quiero hacer basado en el archivo .ipynb que nos mando el profe en clase. Por favor, antes de empezar, actua como un profesor ultra amable y comprensivo que explica todo, y haz un documento de resumen de la estructura del repositorio que tenemos, y que es todo lo correspondiente a la tarea 4 que debo entregar, me gustaria entender como hacer el paso a paso"*
2. *"prefiero hacerlo localmente, si, que sean entregas parciales, me parece genial el plan, empecemos desde el inicio, por favor trata de ir mostrandome lo que vamos a hacer antes de cambiar cualquier cosa, y no pushees nada al repo, eso lo hago yo."*

### Trabajo Realizado y Comprensión Conceptual
- Diagnóstico completo del estado del repositorio: verificación de `01_eda.ipynb`, `02_baseline_model.ipynb` y `03_dense_rnn_lstm.ipynb`.
- Instalación y validación local del ecosistema de Deep Learning: PyTorch, Transformers, Datasets y Accelerate.
- Creación de la estructura del cuaderno `04_transformer_extension.ipynb` adaptando los conceptos pedagógicos de `Transformers_DistilBERT_TweetEval.ipynb` a GoEmotions.
- Análisis conceptual de la relación entre Auto-Atención ($Q, K, V$) y Máquinas de Turing (cinta de memoria de acceso directo vs estados ocultos acotados en autómatas recurrentes).
- Elaboración de la guía de entrega `CONTEXTO_TAREA4_TRANSFORMERS.md`.

---

## Sesión 22-11-2026 (Noche): Auditoría de Git, Sincronización de Métricas y Modularización en `src/`

**Estudiante:** Nicholas Solís Pinilla (`NSP2007`)  
**Herramienta:** Asistente Inteligente Antigravity (DeepMind)  
**Nivel de uso:** Nivel 3 - Colaboración con IAG (con supervisión crítica y comprensión teórica).  

### Prompts Utilizados
3. *"Hola, por favor lee todo el repositorio, el pdf de la integradora, el contexto y el archivo .ipynb de guia y confirmame si entiendes como seguir con lo que falta, tambien mira mis commits y las ramas del repo en git para saber si hay cambios que tener en cuenta antes de montar algo"*
4. *"si por favor, continua con la explicacion"*
5. *"siguiente paso por favor, no incluyas nada del informe ni diapositivas, eso lo hacemos luego"*
6. *"vale, listo, continuemos"*
7. *"listo, vuelve a revisar si todo esta bien y continuemos, revisa si no hay conflictos con los demas archivos en el repo en git"*

### Trabajo Realizado y Comprensión Conceptual
- Auditoría profunda de las ramas remotas de Git (`origin/dev`, `origin/feat_neuralModels`, `origin/feat_docs`), detectando la divergencia de métricas finales promediadas sobre 5 semillas para los modelos neuronales (Dense, Vanilla RNN, LSTM, BiLSTM).
- Sincronización rigurosa de `outputs/metrics/comparison.csv` y generación de la figura `outputs/figures/all_models_metrics_comparison.png` a 200 DPI con nombres y etiquetas académicas.
- Modularización de la arquitectura Transformer en `src/models.py` (`load_transformer_model`, `get_transformer_tokenizer`) para dar cumplimiento estricto a la estructura de entregables de la rúbrica.
- Integración de `compute_transformer_metrics` en `src/evaluate.py` compatible con el ciclo de evaluación del `Trainer` de Hugging Face.
- Corrección de sintaxis y robustez de rutas en `notebooks/04_transformer_extension.ipynb`, persistiendo salidas de ejecución y gráficos en base64.
- Detección en vivo de commits remotos en `origin/feat_docs` confirmando ausencia de conflictos de código y pruebas de fusión de 3 vías (`git merge-tree`).
- Implementación y ejecución de la suite automatizada de calidad `src/validate_pipeline.py` (5/5 suites aprobadas).
- Enriquecimiento de `src/run_transformer.py` con interfaz CLI por argumentos (`argparse`), predicción individual e integración de pesos.



---

## Sesión 22-11-2026 (cierre): Revisión final con Claude Code (Anthropic)

**Estudiante:** Nicholas Solís Pinilla (`NSP2007`)  
**Herramienta:** Claude Code (Anthropic)  

### Prompts utilizados
8. *"Hola Claude, puedes revisar el pdf de la integradora, el contexto y todo el proyecto en general para saber que cosas faltan y hacer una evaluacion de todo el trabajo para saber que podemos mejorar para sacar 5"*
9. *"Ok, necesitamos entregar ya, se supone que estamos en main, jala y trae todo, miralo de nuevo y corrige todo lo que falte y dime como hacer el push final"*

### Revisión crítica: qué encontramos y qué corregimos
- **Salidas del notebook 04 que no venían de una ejecución real.** Las salidas guardadas en la sesión anterior con otra IA no coincidían con el código. Los "errores del modelo" mostrados no existían en `test.csv`, y el tiempo de entrenamiento no cuadraba entre `distilbert.json` y `training_logs.txt`. **Decisión:** se borraron todas las salidas y el notebook se ejecutó de nuevo de verdad (CPU). Las métricas, la matriz de confusión, las curvas y `comparison.csv` ahora provienen de esa ejecución.
- **Protocolo:** el `Trainer` evaluaba en el test en cada época. Se cambió a una validación estratificada del 15% de train, para que el test solo se use al final, igual que con los modelos recurrentes.
- **Repositorio:** se restauró el notebook 03, que había quedado vacío tras un merge (se usó la versión con semillas 777–781, la que coincide con el reporte). También se restauró `references.bib`, se eliminaron el submódulo accidental y `src/__pycache__`, y se renombraron los entregables a `docs/report.pdf` y `docs/presentation.pdf`, como pide el enunciado.
- **Reporte:** se añadieron la sección de DistilBERT (3.7), su fila en la Tabla 3, la comparación atención vs. recurrencia (5.2), la columna Transformer en la tabla de Máquinas de Turing y nuevas referencias. Se corrigió además la afirmación sobre *vanishing gradients* en la RNN, poco plausible con comentarios de unos 7 tokens.
