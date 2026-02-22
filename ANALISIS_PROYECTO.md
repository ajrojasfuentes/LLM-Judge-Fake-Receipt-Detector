# Análisis integral del proyecto: LLM-Judge Fake Receipt Detector

## 1) Estado actual del repositorio y rama de referencia

- Rama local activa: `work`.
- Último commit en HEAD: `9d8fac0` (merge de PR #13).
- No existe una rama local `main` en este clon, pero el historial confirma integración reciente desde `main` vía merge commit `d0581c5`.
- La revisión aquí documentada corresponde al estado más reciente disponible en este checkout (HEAD en `work`), que incluye los cambios ya fusionados desde `main`.

## 2) Propósito del proyecto

El proyecto implementa un detector de recibos falsos basado en **múltiples jueces LLM/VLM**. Cada juez analiza la imagen de un recibo y emite JSON estructurado con etiqueta (`FAKE/REAL/UNCERTAIN`), confianza, razones y señales forenses; luego un motor de votación combina los dictámenes en un veredicto final.

## 3) Arquitectura funcional

### 3.1 Flujo operativo principal (CLI)

El `main.py` expone comandos:

- `download`: descarga/extracción del dataset.
- `sample`: muestreo estratificado 10 REAL + 10 FAKE.
- `run`: ejecución de 3 jueces y votación final.
- `evaluate`: métricas (accuracy, precision, recall, F1, matriz de confusión).
- `demo`: ejecución sobre un solo recibo.
- `forensic`: preanálisis forense para un recibo.

### 3.2 Componentes

- **DatasetManager (`pipeline/dataset.py`)**: descarga robusta, validación ZIP, carga de etiquetas por split, búsqueda de imagen/OCR.
- **ReceiptSampler (`pipeline/sampler.py`)**: muestreo reproducible con semilla y guardado de muestra.
- **Jueces (`judges/`)**:
  - Qwen2.5-VL (2 personas): Forensic Accountant, Document Examiner.
  - GLM-4.5V (1 persona): Holistic Auditor.
- **Rubric (`skills/rubric.py`)**: construcción de prompt desde plantillas modulares y esquema JSON de salida.
- **VotingEngine (`judges/voting.py`)**: votación dinámica ponderada por incertidumbre (o estrategias legacy).
- **Evaluator (`pipeline/evaluator.py`)**: consolidación de resultados y métricas.
- **Forensics (`forensics/`)**: pipeline modular de evidencia (MELA, ruido, copy-move, OCR postprocess y chequeos semánticos).

## 4) Diseño de decisión y agregación

El sistema tiene 3 estrategias de voto:

1. `dynamic_weighted` (por defecto en código de VotingEngine):
   - Calcula incertidumbre por juez (confianza no lineal + skills inciertas + penalizaciones de coherencia).
   - Convierte a pesos `max(0.1, 1 - incertidumbre)`.
   - Determina ganador entre `FAKE/REAL` y fuerza `UNCERTAIN` si margen insuficiente o umbral de votos inciertos.
2. `majority`.
3. `confidence_weighted`.

Observación: `configs/judges.yaml` fija actualmente `voting.strategy: "majority"`, por lo que el comportamiento por defecto al ejecutar `run` dependerá de ese archivo de configuración.

## 5) Dataset y supuestos

El proyecto trabaja con el dataset **Find it again!** (988 recibos, altamente desbalanceado). Por eso, el muestreo usado para evaluación interna toma 20 casos balanceados (10/10), buscando evitar sesgo por clase mayoritaria.

## 6) Hallazgos técnicos relevantes (estado actual)

### 6.1 Inconsistencia de imports forenses

- `main.py` y `judges/base_judge.py` referencian `pipeline.forensic_pipeline`.
- Ese archivo fue eliminado en cambios recientes (`D pipeline/forensic_pipeline.py`) y la implementación forense activa está en `forensics/pipeline.py`.

Impacto:
- Rutas `--forensic`, comando `forensic` y tipado de `ForensicContext` pueden fallar en runtime si se ejecutan tal como están.

### 6.2 Cobertura de tests

La suite valida:

- parseo/normalización de respuestas de jueces;
- estrategias de votación;
- sampler/evaluator;
- `forensics.pipeline.build_evidence_pack`.

Ejecución local:
- `PYTHONPATH=. pytest -q` → 24 tests passing.
- `pytest -q` sin `PYTHONPATH` falla por resolución de módulos (entorno/no instalación editable).

### 6.3 Evolución reciente (commits)

Los cambios más recientes se concentran en forensics:

- refactor del pipeline forense;
- separación de templates de evidence pack por modo (`full/graphic/reading`);
- incorporación de `ocr_postprocess` y `semantic_check`;
- eliminación de componentes legacy (incluyendo `pipeline/forensic_pipeline.py`).

## 7) Cambios recientes y lectura de tendencia del proyecto

Tendencia clara del historial:

1. **Fortalecimiento del bloque forense** con modularidad y mayor tolerancia a fallos.
2. **Estandarización de evidence packs** para distintos modos de uso.
3. **Enfoque en robustez del parseo y validación** de resultados de jueces.
4. **Persistencia de deuda técnica de integración** entre CLI/Judges y nuevo módulo forense.

## 8) Propósito operativo final del sistema

Con el estado actual, el proyecto busca ser un **orquestador de decisión asistida por VLMs** para detección de fraude documental en recibos, combinando:

- razonamiento multimodal de LLMs,
- prompts forenses estructurados,
- agregación por incertidumbre,
- y (opcionalmente) señales forenses computacionales previas.

No es un clasificador puro de visión tradicional, sino un pipeline híbrido de juicio estructurado + votación.

## 9) Recomendaciones priorizadas

1. **Corregir la integración forense rota**:
   - migrar imports de `pipeline.forensic_pipeline` a una API estable en `forensics`;
   - o reintroducir wrapper de compatibilidad.
2. **Definir punto único de configuración efectiva**:
   - alinear README/config/código respecto a estrategia de voto por defecto.
3. **Mejorar ejecutabilidad test/CLI fuera de IDE**:
   - empaquetar módulo (`pip install -e .`) o ajustar `pytest.ini` para `pythonpath`.
4. **Asegurar trazabilidad de rama objetivo**:
   - mantener rama `main` local/remota sincronizada para auditorías reproducibles.

## 10) Conclusión

El proyecto está bien encaminado y muestra madurez en su diseño de prompts, parseo estructurado, y votación probabilística. Su principal punto débil hoy no parece conceptual sino de **integración posterior a refactor forense**, donde hay referencias obsoletas que pueden romper los caminos de ejecución que usan preanálisis forense.
