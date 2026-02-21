# LLM-Judge Fake Receipt Detector — Context Document

> **Propósito de este archivo:** referencia técnica completa del repositorio. Documenta la arquitectura, los modelos usados, el dataset, el flujo de ejecución, y cada componente clave del código tal como están implementados en `main`.

---

## 1. Visión General

Sistema multi-juez que utiliza tres Vision-Language Models (VLMs) para clasificar recibos de compra como **REAL**, **FAKE** o **UNCERTAIN**. Cada juez analiza la imagen del recibo con un rol forense especializado y produce un veredicto estructurado en JSON. Un motor de votación agrega los tres veredictos en una decisión final ponderada por incertidumbre.

**Dataset:** *Find it again! – Receipt Dataset for Document Forgery Detection*
**Fuente:** L3i Lab, Universidad de La Rochelle — https://l3i-share.univ-lr.fr/2023Finditagain/
**Acceso a modelos:** HuggingFace Inference API (`HF_TOKEN`)

---

## 2. Flujo de Ejecución

```
Dataset (Find-It-Again)
        │
        ▼
DatasetManager.download() + extract()   → data/raw/findit2/
        │
        ▼
ReceiptSampler.sample()                 → outputs/samples.json
  (10 REAL + 10 FAKE, split=train, seed=42)
        │
        ▼
ForensicPipeline.analyze() [opcional]   → ForensicContext
  (Multi-ELA, Noise, DCT, Copy-Move, OCR Aritmético)
        │
        ▼
┌───────────────────────────────────────────────────────┐
│  Judge 1: Forensic Accountant (Qwen2.5-VL-72B)       │
│  Judge 2: Document Examiner   (Qwen2.5-VL-72B)       │
│  Judge 3: Holistic Auditor    (GLM-4.5V)              │
└────────────────────────┬──────────────────────────────┘
                         │
                         ▼
               VotingEngine.aggregate()
               (dynamic_weighted por defecto)
                         │
                         ▼
                   FinalVerdict
              REAL | FAKE | UNCERTAIN
                         │
                         ▼
             outputs/results/{id}.json
                         │
                         ▼
               Evaluator.summary()
        (accuracy, precision, recall, F1)
```

---

## 3. Dataset: Find it again!

### Estadísticas Globales

| Partición | Total | Falsos (FAKE) | Auténticos (REAL) |
|-----------|-------|---------------|-------------------|
| Train     | 577   | 94            | 483               |
| Val       | 193   | 34            | 159               |
| Test      | 218   | 35            | 183               |
| **Total** | **988** | **163 (~16.5%)** | **825 (~83.5%)** |

**Desequilibrio importante:** solo el 16.5% de los recibos son falsos → el muestreo estratificado (10+10) compensa este sesgo para la evaluación.

### Tipos de Falsificación (sobre 455 áreas modificadas)

| Código | Descripción | Frecuencia |
|--------|-------------|------------|
| **CPI** | Copy-Paste Inside (mismo documento) | ~77.6% |
| **IMI** | Inserción de texto imitando la fuente | ~7.9% |
| **CUT** | Eliminación de caracteres | ~7.9% |
| **PIX** | Modificación a nivel de píxel | ~7.3% |
| **CPO** | Copy-Paste desde fuera del documento | ~2.2% |

### Campos más Alterados

| Campo | Frecuencia |
|-------|------------|
| Total / Pago | ~51.4% |
| Productos / líneas de item | ~20.9% |
| Metadatos (fecha, hora) | ~18.0% |
| Nombre de empresa | ~5.7% |

### Implicación para los Jueces

El tipo dominante (CPI ~78%) es extremadamente sutil: se copia un dígito o región **dentro del mismo recibo** y se pega en otro lugar. Esto significa:
- Las alteraciones suelen ser de **un solo dígito** (ej. cambiar un total).
- Las trazas son **mínimas**: posibles halos en bordes, inconsistencias de compresión, o aritmética que no cuadra.
- El sistema prompt incluye advertencias explícitas sobre "hard negatives" (anotaciones manuscritas, sellos) que **no son falsificaciones**.

---

## 4. Los Tres Jueces

### 4.1 Judge 1 — Forensic Accountant

| Parámetro | Valor |
|-----------|-------|
| Clase | `QwenJudge` (instancia via `make_forensic_accountant()`) |
| Modelo | `Qwen/Qwen2.5-VL-72B-Instruct` |
| ID | `judge_1` |
| Temperatura | `0.2` (baja → determinista) |
| max_tokens | `1024` |
| Focus skills | `math_consistency`, `contextual_validation` |

**Rol:** Perito contable especializado en fraude de documentos financieros. Prioriza la consistencia matemática y anomalías numéricas. Conservador: solo reporta FAKE ante evidencia clara.

**Diferencia clave vs YAML:** El archivo `configs/judges.yaml` define temperatura `0.1` para judge_1, pero la función de fábrica `make_forensic_accountant()` en `qwen_judge.py` usa `0.2`. Los valores activos en tiempo de ejecución son los de las **factory functions**.

---

### 4.2 Judge 2 — Document Examiner

| Parámetro | Valor |
|-----------|-------|
| Clase | `QwenJudge` (instancia via `make_document_examiner()`) |
| Modelo | `Qwen/Qwen2.5-VL-72B-Instruct` |
| ID | `judge_2` |
| Temperatura | `0.6` (moderada) |
| max_tokens | `1024` |
| Focus skills | `typography_analysis`, `visual_authenticity`, `layout_structure` |

**Rol:** Examinador forense de documentos especializado en falsificaciones visuales. Orientado a anomalías tipográficas, artefactos visuales e inconsistencias de diseño.

**Diferencia clave vs YAML:** El YAML define temperatura `0.7`, la factory function usa `0.6`. Los valores activos son los de las **factory functions**.

---

### 4.3 Judge 3 — Holistic Auditor

| Parámetro | Valor |
|-----------|-------|
| Clase | `GLMJudge` |
| Modelo | `zai-org/GLM-4.5V` |
| ID | `judge_3` |
| Temperatura | `0.3` (balanceada) |
| max_tokens | `1024` |
| Focus skills | Todos (5/5 con igual peso) |
| Timeout | `120.0 s` |

**Rol:** Auditor holístico con experiencia amplia en autenticación de recibos. Aplica las 5 habilidades con el mismo peso para un veredicto independiente y cruzado.

**API call:** Usa `InferenceClient.chat.completions.create()` con fallback a `chat_completion()` para compatibilidad con versiones antiguas de `huggingface_hub`. La imagen va en el mensaje `user` como `image_url` (base64 data URL) y el `persona_description` va en un mensaje `system` separado.

---

### 4.4 Diferencias de Implementación entre QwenJudge y GLMJudge

| Aspecto | QwenJudge | GLMJudge |
|---------|-----------|----------|
| Mensaje system | Embebido en el prompt de usuario | Mensaje `system` separado |
| Orden imagen/texto | Imagen primero, luego texto | Texto primero, luego imagen |
| Fallback API | No | Sí (`chat_completion`) |
| Soporte `.webp` | No | Sí |

---

## 5. Sistema de Prompts y Habilidades (Rubric)

### 5.1 Arquitectura del Prompt

`skills/rubric.py` → `Rubric.build_prompt()` ensambla el prompt completo desde plantillas modularizadas:

```
skills/templates/
├── system_prompt.txt          ← Plantilla maestra con {placeholders}
├── math_consistency.txt       ← Habilidad 1
├── typography_analysis.txt    ← Habilidad 2
├── visual_authenticity.txt    ← Habilidad 3
├── layout_structure.txt       ← Habilidad 4
├── contextual_validation.txt  ← Habilidad 5
└── output_schema.json         ← Esquema JSON de salida
```

**Mecanismo de énfasis:** Todas las 5 habilidades están SIEMPRE presentes en el prompt. Las `focus_skills` reciben el marcador `>>> PRIMARY FOCUS <<<` al inicio de su bloque. Esto permite que cada juez se especialice sin perder cobertura transversal.

### 5.2 Estructura del Prompt (system_prompt.txt)

```
=== RECEIPT FORENSIC ANALYSIS TASK ===
Receipt ID: {receipt_id}
Analyst Role: {persona_name}

{persona_description}

=== DATASET CONTEXT (Read before analyzing) ===
[Contexto del dataset: tipo dominante CPI, campos más alterados, hard negatives]

=== FORENSIC ANALYSIS ===
[SKILL 1 - MATH_CONSISTENCY]      ← >>> PRIMARY FOCUS <<< si es focus_skill
[SKILL 2 - TYPOGRAPHY_ANALYSIS]
[SKILL 3 - VISUAL_AUTHENTICITY]
[SKILL 4 - LAYOUT_STRUCTURE]
[SKILL 5 - CONTEXTUAL_VALIDATION]

REQUIRED OUTPUT FORMAT:
{output_schema}

STRICT RULES + APPROVED FLAG CODES
```

**Si se activa el pre-análisis forense**, el bloque `ForensicContext.to_prompt_section()` se **prepende** al prompt antes de las habilidades.

### 5.3 Las 5 Habilidades Forenses

#### SKILL 1 — MATH_CONSISTENCY
Verifica consistencia aritmética:
- Total de líneas de producto (cantidad × precio unitario = subtotal por línea)
- Suma de subtotales = subtotal declarado
- Subtotal + impuesto = total
- Coherencia del total general y del cambio/pago
- Totales sospechosamente redondos
- Valores imposibles (precios negativos, cantidades cero)
- Valores duplicados (indicador de copy-paste)

#### SKILL 2 — TYPOGRAPHY_ANALYSIS
Detecta anomalías tipográficas:
- Inconsistencias de familia de fuente (mezcla serif/sans-serif)
- Espaciado de caracteres, kern, alineación de baseline
- Inconsistencias de peso y tamaño de texto
- Artefactos de renderizado y halos de superposición de texto
- Resolución de texto inconsistente con el fondo

#### SKILL 3 — VISUAL_AUTHENTICITY
Detección de falsificaciones a nivel de imagen:
- Trazas de copy-paste (bordes duros, halos, fringing de color)
- Inconsistencias de compresión JPEG
- Modificaciones a nivel de píxel (PIX)
- Áreas borradas o blanqueadas
- Superposición digital de texto
- Inconsistencias de resolución en regiones
- Consistencia de textura del papel y uniformidad de iluminación

#### SKILL 4 — LAYOUT_STRUCTURE
Validación de estructura del recibo:
- Flujo estándar: encabezado → ítems → total → pago → pie de página
- Consistencia de alineación de columnas
- Irregularidades en espaciado vertical
- Detección de líneas duplicadas
- Identificación de layouts genéricos de plantilla vs. auténticos
- Coherencia de la sección de pago y pie de página

#### SKILL 5 — CONTEXTUAL_VALIDATION
Verificación de coherencia semántica:
- Plausibilidad de fecha (no futura, no demasiado antigua, formato y día-de-semana)
- Coherencia del nombre e información de la tienda
- Plausibilidad de los ítems para el tipo de establecimiento
- Consistencia de moneda y configuración regional
- Consistencia del método de pago
- Contradicciones internas (fecha vs. horario de atención, etc.)
- Plausibilidad del número de recibo/transacción

### 5.4 Esquema de Salida JSON

```json
{
  "label": "FAKE|REAL|UNCERTAIN",
  "confidence": 87.5,
  "reasons": [
    "observación específica 1 (< 20 palabras)",
    "observación específica 2"
  ],
  "skill_results": {
    "math_consistency": "pass|fail|uncertain",
    "typography": "pass|fail|uncertain",
    "visual_authenticity": "pass|fail|uncertain",
    "layout_structure": "pass|fail|uncertain",
    "contextual_validation": "pass|fail|uncertain"
  },
  "flags": ["TOTAL_MISMATCH", "FONT_INCONSISTENCY"],
  "risk_level": "low|medium|high"
}
```

**Flags aprobados:**
`TOTAL_MISMATCH`, `TAX_ERROR`, `LINE_ITEM_ERROR`, `FONT_INCONSISTENCY`, `TEXT_OVERLAY`, `COPY_PASTE_ARTIFACT`, `COMPRESSION_ANOMALY`, `MISSING_FIELDS`, `TEMPLATE_LAYOUT`, `IMPLAUSIBLE_DATE`, `IMPLAUSIBLE_STORE`, `CURRENCY_MISMATCH`, `PAYMENT_INCONSISTENCY`, `ERASED_CONTENT`, `RESOLUTION_MISMATCH`, `SUSPICIOUS_ROUND_TOTAL`

**Reglas estrictas:** El JSON debe ser válido, sin fences de markdown. Se reintenta hasta 3 veces en caso de error de parsing. Flags no reconocidos son filtrados. Label inválido se normaliza a `UNCERTAIN`.

---

## 6. Motor de Votación (VotingEngine)

### 6.1 Estrategia por Defecto: `dynamic_weighted`

**Paso 1 — Incertidumbre por juez:**

```
Si label == "UNCERTAIN" → uncertainty = 1.0

De lo contrario:
  c = confidence / 100.0

  u_conf = (1.0 - c)                             si c >= 0.75  [lineal: rango [0.0, 0.25]]
  u_conf = 0.25 + (0.75 - c), capped a 1.0      si c < 0.75   [penalización más severa]

  u_skill = n_skills_uncertain / n_skills_total

  consistency_penalty:
    +0.12  si risk_level == "high" AND confidence > 80  (sobreconfianza)
    +0.08  si risk_level == "low"  AND confidence < 65  (exceso de cautela)

  uncertainty = 0.55 × u_conf + 0.35 × u_skill + consistency_penalty
              (clamp a [0.0, 1.0])
```

**Paso 2 — Peso por juez:**

```
weight = max(0.1, 1.0 - uncertainty)
```

**Paso 3 — Agregación:**

```
FAKE_score = suma de weights de jueces que votaron FAKE
REAL_score = suma de weights de jueces que votaron REAL

winner = label con mayor score
winner_ratio = winner_score / total_weight

Si winner_ratio > 0.40 → label = winner
De lo contrario → label = UNCERTAIN (margen insuficiente)

Forzado UNCERTAIN si >= 2 jueces votaron UNCERTAIN (uncertain_threshold)
```

**Paso 4 — Incertidumbre del veredicto final:**

```
u_spread  = 1.0 - (winner_weight / total_weight)
u_avg     = media de las uncertainties individuales de los jueces
u_dissent = weight de jueces que votaron en contra del winner / total_weight

verdict_uncertainty = 0.40 × u_spread + 0.35 × u_avg + 0.25 × u_dissent
                    (clamp a [0.0, 1.0])
```

### 6.2 Estrategias Alternativas

| Estrategia | Descripción |
|------------|-------------|
| `majority` | Mayoría simple (≥2 de 3). Empate o umbral de UNCERTAIN → UNCERTAIN |
| `confidence_weighted` | Suma de confidences por label. Ganador necesita suma ≥ 50 |
| `dynamic_weighted` | **Default.** Ponderación por incertidumbre (ver arriba) |

### 6.3 Estructura de Salida: `FinalVerdict`

```python
@dataclass
class FinalVerdict:
    receipt_id: str
    label: str                          # "REAL" | "FAKE" | "UNCERTAIN"
    tally: str                          # ej. "FAKE (w=1.52/2.30)"
    vote_counts: Dict[str, int]         # conteo bruto de votos
    avg_confidence: float               # media de confidences declaradas
    strategy_used: str                  # estrategia efectivamente usada
    all_flags: List[str]                # unión de flags de todos los jueces
    judge_results: List[JudgeResult]
    verdict_uncertainty: float          # [0, 1] — 0=certero, 1=totalmente incierto
    judge_uncertainties: Dict[str, float]  # judge_id → uncertainty score
```

---

## 7. Análisis Forense Pre-Procesado

### 7.1 ForensicPipeline (pipeline/forensic_pipeline.py)

Capa opcional que ejecuta `ForensicAnalyzer` sobre cada imagen **antes** de llamar a los jueces VLM. Los resultados se convierten en texto estructurado (`ForensicContext.to_prompt_section()`) que se **prepende al prompt del juez**, amplificando las señales de falsificación.

**Activación:** `python main.py run --forensic`

### 7.2 Los 5 Análisis Forenses

#### 1. Multi-ELA (Error Level Analysis multi-calidad)
- Comprime la imagen a múltiples calidades JPEG (70, 85, 95) y mide la **varianza píxel a píxel** entre compresiones.
- Una región con "memoria JPEG" previa (pegada desde otra fuente) muestra mayor varianza cross-quality.
- Salida: ratio de píxeles sospechosos, varianza media, bloques divergentes (>2σ).

#### 2. Local Noise Map
- Análisis de varianza de ruido por bloques (32×32 px por defecto).
- Bloques con varianza anómala (>2σ de la media) sugieren regiones con diferente historia de compresión.

#### 3. Frequency Analysis (DCT/FFT)
- Transformada de coseno discreta (DCT) por bloques.
- Ratio de energía de alta frecuencia: bordes artificiales de inserción de texto generan discontinuidades de frecuencia.

#### 4. Copy-Move Detection (ORB keypoint matching)
- Detector ORB (Oriented FAST and Rotated BRIEF) con 3000 features.
- Matching dentro de la misma imagen para detectar regiones copiadas internamente (CPI).
- Calibrado para recibos: umbral 200+ matches y 5+ clusters para alta confianza.
- Distancia mínima entre puntos: 150 px (evita falsos positivos por texto repetitivo).

#### 5. OCR Text Extraction + Arithmetic Verification
- Lee el archivo `.txt` de OCR pareado del dataset.
- Extrae campos estructurados: empresa, fechas, totales, ítems, montos.
- Verifica aritmética: suma de ítems vs. total declarado. Tolerancia del 25% (acomoda impuestos, redondeo).

### 7.3 Parámetros de ForensicPipeline

```python
ForensicPipeline(
    output_dir="outputs/forensic",
    mela_qualities=(70, 85, 95),
    mela_block_size=16,
    mela_variance_threshold=5.0,
    noise_block_size=32,
    freq_block_size=32,
    orb_features=3000,
    match_threshold=0.55,
    min_match_distance=150.0,
    cluster_eps=30.0,
    min_cluster_size=8,
    save_images=True,    # guarda imágenes intermedias en output_dir
    verbose=False,
)
```

### 7.4 Interpretación de Niveles en el Prompt

| Señal | Umbral LOW | Umbral MODERATE | Umbral HIGH |
|-------|------------|-----------------|-------------|
| Multi-ELA (ratio) | < 3% | 3–10% | > 10% |
| Noise Map (ratio) | < 5% | 5–15% | > 15% |
| DCT anomalous blocks | < 20 | 20–50 | > 50 |
| Copy-Move (confidence) | < 0.2 | 0.2–0.5 | > 0.5 |

---

## 8. Clases de Datos Principales

### JudgeResult (judges/base_judge.py)

```python
@dataclass
class JudgeResult:
    judge_id: str
    judge_name: str
    receipt_id: str
    label: str                      # "REAL" | "FAKE" | "UNCERTAIN"
    confidence: float               # [0.0 – 100.0]
    reasons: list[str]              # 2–4 observaciones cortas
    skill_results: dict[str, str]   # skill → "pass|fail|uncertain"
    flags: list[str]                # códigos de flag aprobados
    risk_level: str                 # "low" | "medium" | "high"
    raw_response: str               # respuesta raw del LLM (debug)
    parse_error: str | None         # error de parsing si aplica
```

**Constantes de validación:**
- `VALID_LABELS = {"REAL", "FAKE", "UNCERTAIN"}`
- `VALID_SKILL_RESULTS = {"pass", "fail", "uncertain"}`
- `VALID_RISK_LEVELS = {"low", "medium", "high"}`
- 16 flags aprobados (ver sección 5.4)

**Fallback:** `JudgeResult.error_result()` devuelve un resultado UNCERTAIN con confidence=0.0 y todos los skills en "uncertain" cuando el parsing falla definitivamente.

### ForensicContext (pipeline/forensic_pipeline.py)

```python
@dataclass
class ForensicContext:
    image_path: str
    # Multi-ELA
    multi_ela_suspicious_ratio: float | None
    multi_ela_mean_variance: float | None
    multi_ela_max_variance: float | None
    multi_ela_divergent_blocks: int | None
    multi_ela_total_blocks: int | None
    # Noise
    noise_anomalous_ratio: float | None
    noise_anomalous_blocks: int | None
    noise_total_blocks: int | None
    # Frequency
    freq_anomalous_blocks: int | None
    freq_hf_mean: float | None
    # Copy-Move
    cm_num_matches: int | None
    cm_num_clusters: int | None
    cm_confidence: float | None
    # OCR
    ocr_cleaned_text: str | None
    ocr_company: list[str] | None
    ocr_dates: list[str] | None
    ocr_totals: list[str] | None
    ocr_items: list[str] | None
    ocr_amounts: list[float] | None
    ocr_quality: float | None
    ocr_arithmetic_report: dict | None
    errors: dict[str, str]
```

---

## 9. Pipeline de Evaluación

### Métricas Calculadas (Evaluator — pipeline/evaluator.py)

| Métrica | Fórmula |
|---------|---------|
| Accuracy | (TP + TN) / Total |
| Precision (FAKE) | TP / (TP + FP) |
| Recall (FAKE) | TP / (TP + FN) |
| F1 (FAKE) | 2 × (P × R) / (P + R) |
| Confusion Matrix | TP, TN, FP, FN, UNCERTAIN |

**UNCERTAIN cuenta como incorrecto.**

### Análisis de Desacuerdo

`Evaluator.disagreement_cases(n)` devuelve hasta n casos donde los jueces no coincidieron todos (set de labels > 1). Incluye los reasons y confidences de cada juez para post-análisis.

---

## 10. Comandos CLI (main.py)

```bash
# Descargar y extraer dataset Find-It-Again (~1 GB ZIP)
python main.py download

# Muestrear 20 recibos (10 REAL + 10 FAKE, seed=42, split=train)
python main.py sample

# Ejecutar los 3 jueces sobre los 20 recibos muestreados
python main.py run

# Ejecutar con pre-análisis forense (Multi-ELA, copy-move, OCR)
python main.py run --forensic

# Calcular métricas de evaluación
python main.py evaluate

# Demo rápida sobre un recibo individual
python main.py demo X00016469622
python main.py demo X00016469622 --forensic

# Pre-análisis forense sobre un recibo individual (imprime el contexto)
python main.py forensic X00016469622
```

---

## 11. Configuración

### configs/judges.yaml

Define los 3 jueces, modelos, personas y estrategia de votación.
**Nota:** Los valores de temperatura en el YAML (`judge_1: 0.1`, `judge_2: 0.7`) **difieren** de los de las factory functions (`judge_1: 0.2`, `judge_2: 0.6`). En tiempo de ejecución se usan los valores de `qwen_judge.py`.

```yaml
voting:
  strategy: "dynamic_weighted"
  uncertain_threshold: 2      # >= 2 votos UNCERTAIN → forzar UNCERTAIN
  tiebreak: "confidence_avg"  # solo para confidence_weighted
```

### configs/sampling.yaml

```yaml
sampling:
  random_seed: 42
  total_samples: 20
  real_count: 10
  fake_count: 10
  split: "train"
  output_file: "outputs/samples.json"

dataset:
  download_url: "http://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip"
  raw_dir: "data/raw"
  findit2_dir: "data/raw/findit2"
  metadata_dir: "data/dataset/findit2"
```

---

## 12. Estructura del Repositorio

```
LLM-Judge-Fake-Receipt-Detector/
├── main.py                        # CLI: download | sample | run | evaluate | demo | forensic
├── forensic_analysis.py           # ForensicAnalyzer: Multi-ELA, Noise, DCT, Copy-Move, OCR
├── requirements.txt
├── .env.example                   # Plantilla: HF_TOKEN=hf_...
│
├── configs/
│   ├── judges.yaml               # Definición de jueces, modelos, personas, voting
│   └── sampling.yaml             # Dataset URLs, splits, parámetros de muestreo
│
├── judges/
│   ├── __init__.py
│   ├── base_judge.py             # BaseJudge (ABC) + JudgeResult + constantes
│   ├── qwen_judge.py             # QwenJudge + make_forensic_accountant() + make_document_examiner()
│   ├── glm_judge.py              # GLMJudge (Holistic Auditor)
│   └── voting.py                 # VotingEngine + FinalVerdict
│
├── pipeline/
│   ├── __init__.py
│   ├── dataset.py                # DatasetManager: download, extract, load_labels, find_image
│   ├── sampler.py                # ReceiptSampler: stratified sample, save/load
│   ├── evaluator.py              # Evaluator: accuracy, F1, confusion matrix, disagreement
│   └── forensic_pipeline.py     # ForensicPipeline + ForensicContext
│
├── skills/
│   ├── __init__.py
│   ├── rubric.py                 # Rubric: build_prompt() desde templates
│   └── templates/
│       ├── system_prompt.txt          # Plantilla maestra del prompt
│       ├── math_consistency.txt       # Habilidad 1
│       ├── typography_analysis.txt    # Habilidad 2
│       ├── visual_authenticity.txt    # Habilidad 3
│       ├── layout_structure.txt       # Habilidad 4
│       ├── contextual_validation.txt  # Habilidad 5
│       └── output_schema.json         # Esquema JSON esperado
│
├── data/
│   ├── dataset/findit2/          # CSVs de metadata pre-recolectados
│   │   ├── train_data.csv        # 577 filas
│   │   ├── val_data.csv          # 193 filas
│   │   ├── test_data.csv         # 217 filas
│   │   ├── train.txt             # Labels: {id} {0|1}
│   │   ├── val.txt
│   │   └── test.txt
│   └── raw/                      # Vacío hasta ejecutar download
│
├── outputs/
│   ├── samples.json              # Muestra de 20 recibos (generado por sample)
│   ├── results/                  # JSONs de FinalVerdict por recibo
│   └── forensic/                 # Imágenes intermedias de análisis forense
│
├── tests/
│   ├── test_judges.py            # Unit tests: parsing JSON, normalización de labels
│   ├── test_voting.py            # Unit tests: estrategias de votación, umbral UNCERTAIN
│   └── test_pipeline.py          # Unit tests: sampler, carga de labels
│
└── notebooks/
    ├── 01_dataset_exploration.ipynb      # EDA del dataset
    ├── 02_evaluation_analysis.ipynb      # Análisis de resultados
    └── 03_end_to_end_demo.ipynb          # Demo completo del pipeline
```

---

## 13. Dependencias

```
# Core
huggingface-hub>=0.23.0    # InferenceClient (Qwen2.5-VL + GLM-4.5V)
requests>=2.31.0
pyyaml>=6.0

# Análisis forense
Pillow>=10.0.0
opencv-python-headless>=4.9.0   # ELA, DCT/FFT, copy-move (ORB)
numpy>=1.26.0

# Dataset y EDA
pandas>=2.1.0
matplotlib>=3.8.0
seaborn>=0.13.0
jupyter>=1.0.0
notebook>=7.0.0
ipykernel>=6.0.0

# Opcional
streamlit>=1.32.0

# Testing
pytest>=8.0.0
```

---

## 14. Setup Rápido

```bash
# 1. Instalar dependencias
pip install -r requirements.txt

# 2. Configurar token de HuggingFace
cp .env.example .env
# Editar .env y añadir: HF_TOKEN=hf_tu_token_aqui

# 3. Descargar dataset
python main.py download

# 4. Muestrear
python main.py sample

# 5. Ejecutar jueces
python main.py run --forensic

# 6. Evaluar
python main.py evaluate
```

---

## 15. Decisiones de Diseño Clave

| Decisión | Justificación |
|----------|---------------|
| Sin OCR como dependencia principal | Los VLMs razonan directamente desde la imagen, capturando anomalías visuales que el OCR perdería |
| JSON estricto + retry (max 3) | Elimina ruido de texto libre y asegura respuestas parseables |
| 2 instancias Qwen + 1 GLM | Diversidad de modelo sin necesitar 3 cuentas de API distintas |
| Dynamic weighted voting | La ponderación por incertidumbre es más robusta que mayoría simple; refleja la calidad del voto |
| Análisis forense opcional | Los señales (Multi-ELA, copy-move) amplifican trazas sutiles de CPI para los VLMs |
| 5 habilidades ordenadas | Fuerza razonamiento paso a paso antes de emitir el veredicto, reduciendo sesgos de anclaje |
| Mecanismo de énfasis (focus_skills) | Especialización por juez sin perder cobertura cruzada |
| Muestreo estratificado 10+10 | Compensa el desequilibrio de clases (16.5% FAKE) para evaluación imparcial |
| Seed 42 fijo | Reproducibilidad total del experimento |
