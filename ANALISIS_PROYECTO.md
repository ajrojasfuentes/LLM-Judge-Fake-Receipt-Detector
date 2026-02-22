# Análisis técnico del proyecto: LLM-Judge Fake Receipt Detector

## 1) Estado actual y propósito

Este repositorio implementa un sistema de **detección de recibos falsos** basado en un enfoque de **múltiples jueces LLM/VLM** que emiten veredictos estructurados (`REAL`, `FAKE`, `UNCERTAIN`) y luego se agregan con un motor de votación. El flujo principal está centrado en:

1. Descargar y preparar el dataset Find-It-Again.
2. Muestrear un subconjunto estratificado (10 reales + 10 falsos).
3. Ejecutar tres jueces multimodales sobre cada recibo.
4. Agregar resultados y evaluar métricas.

La entrada de cada juez es la imagen del recibo + prompt forense modular; la salida se normaliza a JSON para reducir respuestas no estructuradas.

---

## 2) Arquitectura funcional

### CLI principal

`main.py` expone comandos para:

- `download`
- `sample`
- `run` (con opción `--forensic`)
- `evaluate`
- `demo`
- `forensic`

Esto define un pipeline reproducible end-to-end para dataset, inferencia y evaluación.

### Subsistemas principales

- `pipeline/dataset.py`: descarga, extracción, carga de etiquetas y localización de imágenes/OCR.
- `pipeline/sampler.py`: muestreo estratificado reproducible con semilla fija.
- `judges/*`: implementación de jueces Qwen/GLM + contrato base y parseo robusto.
- `judges/voting.py`: agregación de votos con estrategia dinámica ponderada por incertidumbre.
- `pipeline/evaluator.py`: métricas (accuracy, precision, recall, F1, matriz de confusión) y casos de desacuerdo.
- `skills/rubric.py` + `skills/templates/*.md`: construcción modular del prompt forense.

---

## 3) Modelos y estrategia de decisión

### Jueces activos

- **Judge 1**: Qwen2.5-VL (rol “Forensic Accountant”, foco matemático/contextual).
- **Judge 2**: Qwen2.5-VL (rol “Document Examiner”, foco visual/tipográfico/layout).
- **Judge 3**: GLM-4.5V (rol holístico).

### Contrato de salida

Todos los jueces convergen al `JudgeResult` con:

- `label`
- `confidence`
- `reasons`
- `skill_results`
- `flags`
- `risk_level`

La clase base (`BaseJudge`) aplica normalización, filtros de flags aprobados, alias legacy (`typography`→`typography_analysis`) y fallback a `UNCERTAIN` si falla parseo/validación.

### Votación

`VotingEngine` soporta:

- `dynamic_weighted` (principal)
- `majority`
- `confidence_weighted`

La modalidad dinámica introduce incertidumbre por juez (confianza + skills inciertas + coherencia label/risk/confidence) y exige margen de peso para declarar ganador, evitando dominancia espuria.

---

## 4) Dataset y muestreo

El proyecto trabaja con **Find-It-Again**, con distribución desbalanceada (más reales que falsos). Para evaluación controlada se usa una muestra balanceada 10/10 con semilla fija (`42`).

`sampling.yaml` concentra tanto parámetros de muestreo como ubicación del dataset y metadatos de splits.

---

## 5) Cambios recientes (historial más próximo)

### Últimos commits relevantes

1. `d5725d0` — **Purge legacy files**
   - Renombrado/migración de módulo `forensics_analysis/` a `forensics/`.
   - Eliminación de carpeta legacy `forensic/` y archivos antiguos.
   - Migración de plantillas de skills `.txt` a `.md` y ajuste de `skills/rubric.py`.

2. `be56b0f` — **Skills Improvements**
   - Reestructura del sistema de skills con nuevas plantillas Markdown.
   - Refuerzo de construcción de prompts y compatibilidad.

3. `9d6e552` — **fix debug forensics_analysis module**
   - Correcciones y endurecimiento del paquete de análisis forense.

### Lectura de tendencia

La línea reciente del proyecto prioriza:

- Consolidación del módulo forense.
- Limpieza técnica (legacy purge).
- Endurecimiento de prompts/skills y formato estructurado de salida.

---

## 6) Hallazgos técnicos importantes

### 6.1 Inconsistencia de rutas forenses

Existe una inconsistencia entre imports y estructura actual:

- El código presente en `pipeline/forensic_pipeline.py` importa desde `forensic.*`.
- El árbol actual del repositorio mantiene los módulos en `forensics/*`.

Dado el commit de limpieza reciente que eliminó `forensic/`, este punto sugiere una **regresión potencial** en ejecución de `--forensic`.

### 6.2 Estrategia de votación por defecto en runtime

Aunque el README describe el enfoque dinámico, `configs/judges.yaml` declara `strategy: "majority"`. El runtime de `main.py run` carga esa configuración, por lo que el comportamiento efectivo depende del YAML vigente.

### 6.3 Cobertura de pruebas

El proyecto tiene pruebas unitarias para:

- Parseo/validación de jueces.
- Votación.
- Muestreo y carga de etiquetas.

No se observan pruebas end-to-end de inferencia real (normal por depender de APIs externas/token), pero la base de lógica local está razonablemente cubierta.

---

## 7) Ejecución y operación

Secuencia operativa recomendada (actual):

1. `pip install -r requirements.txt`
2. Configurar `.env` con `HF_TOKEN`.
3. `python main.py download`
4. `python main.py sample`
5. `python main.py run` (o `--forensic` si se corrige/valida la inconsistencia de imports)
6. `python main.py evaluate`

---

## 8) Conclusión ejecutiva

El proyecto está bien orientado para una **evaluación forense asistida por VLMs** con estructura modular (dataset/sampler/judges/voting/evaluator/skills) y buen foco en robustez de salida JSON. Los cambios recientes muestran madurez en limpieza y arquitectura.

No obstante, hay dos prioridades técnicas inmediatas para maximizar confiabilidad:

1. **Corregir la inconsistencia `forensic` vs `forensics`** en `pipeline/forensic_pipeline.py`.
2. **Alinear documentación y configuración de votación por defecto** (README vs `configs/judges.yaml`) para evitar ambigüedad operativa.
