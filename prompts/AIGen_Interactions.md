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


