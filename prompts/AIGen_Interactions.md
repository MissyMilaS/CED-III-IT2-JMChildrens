# Interacciones con IAGen

Documenta aquí las interacciones con herramientas de inteligencia artificial
generativa utilizadas en la actividad.


---

## Sesión 22-11-2026: Planificación e Implementación de la Tarea 4 (Transformers / DistilBERT)

**Estudiante:** Nicholas Solís Pinilla (`NSP2007`)  
**Herramienta:** Asistente Inteligente Antigravity (DeepMind)  
**Nivel de uso:** Nivel 3 - Colaboración con IAG (con supervisión crítica y comprensión teórica).  

### Prompts Utilizados
1. *"tenemos esta tarea integradora que hacer en el .pdf y a mi me correspondes hacer el ultimo modelo de Transformers, lo quiero hacer basado en el archivo .ipynb que nos mando el profe en clase. Por favor, antes de empezar, actua como un profesor ultra amable y comprensivo que explica todo, y haz un documento de resumen de la estructura del repositorio que tenemos, y que es todo lo correspondiente a la tarea 4 que debo entregar..."*
2. *"prefiero hacerlo localmente, si, que sean entregas parciales, me parece genial el plan, empecemos desde el inicio, por favor trata de ir mostrandome lo que vamos a hacer antes de cambiar cualquier cosa, y no pushees nada al repo, eso lo hago yo."*

### Trabajo Realizado y Comprensión Conceptual
- Diagnóstico completo del estado del repositorio: verificación de `01_eda.ipynb`, `02_baseline_model.ipynb` y `03_dense_rnn_lstm.ipynb`.
- Instalación y validación local del ecosistema de Deep Learning: PyTorch, Transformers, Datasets y Accelerate.
- Creación de la estructura del cuaderno `04_transformer_extension.ipynb` adaptando los conceptos pedagógicos de `Transformers_DistilBERT_TweetEval.ipynb` a GoEmotions.
- Análisis conceptual de la relación entre Auto-Atención ($Q, K, V$) y Máquinas de Turing (cinta de memoria de acceso directo vs estados ocultos acotados en autómatas recurrentes).
- Elaboración de la guía de entrega `CONTEXTO_TAREA4_TRANSFORMERS.md`.
