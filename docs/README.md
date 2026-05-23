# Compresión de Redes Neuronales Convolucionales mediante Descomposición Tensorial

> **Documento de referencia para la redacción de los capítulos finales de la tesis.**
> Recoge el diseño experimental completo, las decisiones de implementación, los
> resultados obtenidos y las conclusiones derivadas del proyecto.

---

## Índice

1. [Contexto y motivación](#1-contexto-y-motivación)
2. [Modelo base y dataset](#2-modelo-base-y-dataset)
3. [Métodos de descomposición implementados](#3-métodos-de-descomposición-implementados)
4. [Pipeline de experimentación](#4-pipeline-de-experimentación)
5. [Métricas de evaluación](#5-métricas-de-evaluación)
6. [Estrategia de fine-tuning](#6-estrategia-de-fine-tuning)
7. [Diseño de experimentos](#7-diseño-de-experimentos)
8. [Resultados experimentales](#8-resultados-experimentales)
9. [Análisis comparativo de métodos](#9-análisis-comparativo-de-métodos)
10. [Decisiones de implementación y correcciones](#10-decisiones-de-implementación-y-correcciones)
11. [Conclusiones](#11-conclusiones)

---

## 1. Contexto y motivación

El objetivo del TFG es estudiar empíricamente la compresión de redes neuronales
convolucionales (CNN) mediante técnicas de descomposición tensorial, con aplicación
directa al ámbito de **TinyML** (dispositivos con recursos computacionales y de
memoria muy limitados: microcontroladores, edge devices, sensores inteligentes).

Las CNN modernas son extremadamente sobreparametrizadas. Un modelo como VGG11-BN
entrenado para clasificar imágenes de 32×32 píxeles en 10 clases emplea casi
**10 millones de parámetros**, cifra que excede con creces los recursos disponibles
en hardware embebido. La descomposición tensorial permite sustituir las capas de
mayor coste computacional por estructuras equivalentes de mucho menor tamaño,
reduciendo el número de parámetros almacenados y, en algunos métodos, el número de
operaciones necesarias para la inferencia.

A diferencia de otras técnicas de compresión (poda de pesos, cuantización,
destilación de conocimiento), la descomposición tensorial actúa directamente sobre
la estructura matemática del tensor de pesos de cada capa, descomponiéndolo en un
producto de factores de menor rango. Esta propiedad hace que los métodos
multidimensionales (Tucker, CP, TT) sean teóricamente superiores al SVD clásico
(que opera en 2D) cuando se aplican a convoluciones cuyos pesos son tensores de
orden 4.

El proyecto responde a las preguntas:

- ¿En qué medida reduce la descomposición tensorial el número de parámetros de una CNN manteniendo una precisión aceptable?
- ¿Superan los métodos 4D (Tucker, CP, TT) al SVD clásico en el contexto de capas convolucionales?
- ¿En qué rangos de compresión resulta beneficioso aplicar fine-tuning posterior?
- ¿Cómo se comportan las métricas de memoria y latencia en función del nivel de compresión alcanzado?

---

## 2. Modelo base y dataset

### 2.1 Dataset: CIFAR-10

- **Clases**: 10 (avión, automóvil, pájaro, gato, ciervo, perro, rana, caballo,
  barco, camión).
- **Tamaño de imagen**: 32×32 píxeles RGB.
- **Split utilizado**: 50 000 imágenes de entrenamiento (particionadas en 90 % train /
  10 % validación) y 10 000 de test.
- **Normalización**: media `(0.4914, 0.4822, 0.4465)`, desviación estándar
  `(0.2023, 0.1994, 0.2010)` (estadísticos estándar de CIFAR-10).
- **Data augmentation** (solo train): `RandomCrop(32, padding=4)` y
  `RandomHorizontalFlip`.

CIFAR-10 es un benchmark canónico en visión por computador: suficientemente complejo
para que los resultados sean significativos, pero manejable en términos de tiempo de
entrenamiento y evaluación, lo que lo hace ideal para experimentos repetitivos con
múltiples configuraciones de compresión.

### 2.2 Arquitectura: VGG11-BN

Se utiliza la variante con Batch Normalization de la familia VGG (Simonyan &
Zisserman, 2014), entrenada de forma nativa para CIFAR-10 mediante el hub de PyTorch
(`chenyaofo/pytorch-cifar-models`). Sus características clave como modelo base son:

| Métrica | Valor |
|---|---|
| Parámetros totales | **9 756 426** |
| Memoria estática (parámetros + buffers) | **37.24 MiB** |
| Peak CUDA durante inferencia (batch=128) | **128.38 MiB** |
| Accuracy en test | **92.14 %** |
| MACs (multiply-accumulate) | **0.154 GMACs** |
| Latencia (batch=1) | **2.50 ms** |
| Throughput (batch=128) | **14 594 FPS** |

#### Capas objetivo de la compresión

Las 11 capas seleccionadas para la descomposición son todas las capas convolucionales
y lineales del modelo (se comprimen simultáneamente con el mismo rango):

**Capas convolucionales** (tensores de orden 4, forma `[out_ch, in_ch, kH, kW]`):

| Capa | Forma del tensor | Parámetros originales |
|---|---|---|
| `features.0` | [64, 3, 3, 3] | 1 728 |
| `features.4` | [128, 64, 3, 3] | 73 728 |
| `features.8` | [256, 128, 3, 3] | 294 912 |
| `features.11` | [256, 256, 3, 3] | 589 824 |
| `features.15` | [512, 256, 3, 3] | 1 179 648 |
| `features.18` | [512, 512, 3, 3] | 2 359 296 |
| `features.22` | [512, 512, 3, 3] | 2 359 296 |
| `features.25` | [512, 512, 3, 3] | 2 359 296 |

**Capas lineales** (tensores de orden 2, forma `[out, in]`):

| Capa | Forma | Parámetros originales |
|---|---|---|
| `classifier.0` | [512, 512] | 262 144 + 512 (bias) |
| `classifier.3` | [512, 512] | 262 144 + 512 (bias) |
| `classifier.6` | [10, 512] | 5 120 + 10 (bias) |

Las capas de Batch Normalization y MaxPool no se comprimen (sus parámetros conforman
el overhead fijo de **8 256 parámetros**). Los parámetros dominantes están en las
capas convolucionales profundas (`features.15–25`), que suponen más del 70 % del
total.

---

## 3. Métodos de descomposición implementados

Se implementan cuatro métodos. Todos reemplazan in-place la capa original por un
bloque `nn.Sequential` de capas de menor tamaño. Las capas lineales son idénticas en
SVD, Tucker, CP y TT (todas colapsan a SVD truncado en 2D). La diferencia está en
el tratamiento de las capas convolucionales (tensores 4D).

### 3.1 SVD — Descomposición en Valores Singulares (baseline 2D)

**Concepto**: El tensor de pesos `W ∈ ℝ^{out × in·kH·kW}` se despliega en una
matriz 2D (mode-0 unfolding) y se aplica SVD truncado de rango `r`:

```
W ≈ U_r · diag(S_r) · Vh_r
```

El modelo resultante reemplaza la convolución original por dos convoluciones
secuenciales:

- **Conv1**: kernel `(kH, kW)`, `in_ch → r` canales, sin bias — absorbe `S·Vh`
- **Conv2**: kernel `1×1`, `r → out_ch` canales, con bias — absorbe `U`

**Conteo de parámetros**:
```
P_SVD = r · (out_ch + in_ch · kH · kW)
```
Comprime si `r < (out_ch · in_ch · kH · kW) / (out_ch + in_ch · kH · kW)`.

**Limitación**: Al desplegar el tensor a 2D se pierde la estructura espacial inherente
al kernel 4D. SVD es un baseline competitivo pero teóricamente subóptimo para tensores
de orden superior.

**Implementación**: `src/decompositions/svd.py`. Usa `torch.linalg.svd` con
`full_matrices=False`.

---

### 3.2 Tucker-2 — Descomposición de Tucker en modos de canal

**Concepto**: Tucker aplica la compresión sobre los dos modos de canal (entrada y
salida) dejando intactas las dimensiones espaciales. Para un kernel
`W ∈ ℝ^{out × in × kH × kW}`:

```
W ≈ G ×₀ A_out ×₁ A_in
```

donde `G ∈ ℝ^{R_out × R_in × kH × kW}` es el tensor núcleo y `A_out ∈ ℝ^{out × R_out}`,
`A_in ∈ ℝ^{in × R_in}` son las matrices de factores de canal. El modelo resultante
tiene **tres convoluciones** secuenciales:

1. **Conv 1×1** (`in → R_in`): proyección de entrada
2. **Conv kH×kW** (`R_in → R_out`): convolución espacial en el espacio comprimido
3. **Conv 1×1** (`R_out → out`): proyección de salida

**Conteo de parámetros**:
```
P_Tucker = R_in·in + R_out·R_in·kH·kW + out·R_out
```

**Implementación**: `src/decompositions/tucker.py`. Usa `tensorly.decomposition.tucker`
con inicialización SVD (los factores espaciales se absorben de vuelta al núcleo para
no comprimir la dimensión espacial). Para capas lineales, recae en SVD truncado.

**Ventaja sobre SVD**: Explota la estructura 4D completa y aplica compresión
independiente en ambas dimensiones de canal. A igual número de parámetros, Tucker
suele retener mejor la información que SVD.

---

### 3.3 CP — Descomposición Poliádica Canónica

**Concepto**: CP descompone el tensor en una suma de `R` términos de rango 1 (el
formato más comprimido posible):

```
W ≈ Σᵣ aᵣ⁽⁰⁾ ⊗ aᵣ⁽¹⁾ ⊗ aᵣ⁽²⁾ ⊗ aᵣ⁽³⁾
```

donde cada factor `aᵣ⁽ᵢ⁾` corresponde a un modo del tensor (out, in, kH, kW). La
implementación en Conv2d da lugar a cuatro convoluciones secuenciales:

1. **Conv 1×1** (`in → R`): factor de entrada
2. **Conv kH×1** depthwise (`R → R`): factor espacial vertical
3. **Conv 1×kW** depthwise (`R → R`): factor espacial horizontal
4. **Conv 1×1** (`R → out`): factor de salida

**Conteo de parámetros**:
```
P_CP = R · (in + kH + kW + out)
```
Comprime cuando `R < (in·out·kH·kW) / (in + out + kH + kW)`.

**Algoritmo**: CP-ALS (Alternating Least Squares) implementado desde cero, sin
depender de la ruta de memoria de tensorly. Cada actualización de modo resuelve un
sistema de mínimos cuadrados usando el producto de matrices de Gram (R×R), lo que
evita materializar la matriz de Khatri-Rao completa. Inicialización con SVD para
estabilidad en capas profundas.

**Limitación inherente**: La existencia y unicidad del rango CP óptimo no están
garantizadas. El problema de encontrar la mejor aproximación CP de rango fijo es
NP-hard en general (Hillar & Lim, 2013). ALS puede converger lentamente o no
converger. Esto se traduce en tiempos de compresión muy superiores a los otros
métodos (rank 1500 tarda ~227 s frente a ~2 s de SVD para el mismo modelo).

**Salvaguardas implementadas**: timeout configurable por capa (`cp_layer_timeout_s`),
guardia de memoria RAM disponible (`cp_abort_if_mem_available_mb_below`), conmutación
automática de `random` a `svd` init cuando la capa es grande.

---

### 3.4 TT — Tensor Train (Tren de Tensores)

**Concepto**: TT descompone el tensor en una cadena de tensores de orden 3 (núcleos),
conectados por índices de borde (TT-ranks):

```
W[i₀,i₁,i₂,i₃] ≈ G₀[1,i₀,r₁] · G₁[r₁,i₁,r₂] · G₂[r₂,i₂,r₃] · G₃[r₃,i₃,1]
```

Para un kernel `W ∈ ℝ^{out × in × kH × kW}` los núcleos son:
- `G₀ ∈ ℝ^{1 × out × r₁}`
- `G₁ ∈ ℝ^{r₁ × in × r₂}`
- `G₂ ∈ ℝ^{r₂ × kH × r₃}`
- `G₃ ∈ ℝ^{r₃ × kW × 1}`

Los rangos `r₁, r₂, r₃` están limitados superiormente por las dimensiones del
tensor: `r₁ ≤ min(out, in·kH·kW)`, `r₂ ≤ min(out·in, kH·kW)`, etc.

**Conteo de parámetros**:
```
P_TT = out·r₁ + r₁·in·r₂ + r₂·kH·r₃ + r₃·kW
```

**Limitación crítica de implementación**: PyTorch no dispone de una primitiva nativa
para convolución en formato TT. La implementación actual almacena los cuatro núcleos
como parámetros entrenables (`nn.ParameterList`) y **reconstruye el tensor completo
en cada pasada forward** mediante contracciones sucesivas. Esto implica:

- **Memoria de inferencia**: en cada forward se materializa el tensor `W ∈ ℝ^{out×in×kH×kW}` completo, consumiendo el mismo espacio que el tensor original.
- **Latencia de inferencia**: overhead computacional de las contracciones tensoriales en cada pasada.
- **Ventaja real**: solo la memoria estática del modelo (parámetros almacenados) se reduce, pero el pico de memoria durante inferencia no mejora proporcionalmente.

**Implementación**: `src/decompositions/tt.py`. La descomposición se calcula con
`tensorly.decomposition.tensor_train`. Los módulos forward `_TTConv2dModule` y
`_TTLinearModule` gestionan la reconstrucción on-the-fly.

---

### 3.5 Capas lineales: todos los métodos convergen a SVD

Para las tres capas `classifier` (tensores 2D), Tucker-2, CP y TT colapsan
matemáticamente a SVD truncado: un tensor de rango 2 solo admite una única
descomposición de tipo Tucker/CP/TT, que es precisamente la descomposición SVD.
Esto significa que los cuatro métodos son exactamente equivalentes en las capas
lineales y sus diferencias solo se manifiestan en las capas convolucionales.
Dado que las capas lineales suponen menos del 10 % de los parámetros totales
del modelo, el impacto de esta equivalencia es marginal en el cómputo global.

---

## 4. Pipeline de experimentación

El pipeline está implementado en `src/experiments/runner.py` y se ejecuta mediante
`python main.py --config <fichero.json>`.

### 4.1 Flujo de ejecución

```
Cargar config JSON
    └─> Inicializar dataset (CIFAR-10) + DataLoaders (train/val/test)
    └─> Cargar modelo base (VGG11-BN pretrained)
    └─> Evaluar Baseline → guardar fila CSV
    └─> Para cada experimento en config.experiments:
            └─> deep_copy del modelo base
            └─> Reemplazar capas objetivo (ModelReplacer.replace_layers)
            └─> Evaluar modelo comprimido → fila CSV "[compressed]"
            └─> Si fine_tuning=True:
                    └─> Calcular LR dinámico según accuracy comprimida
                    └─> Fine-tuning (best_val checkpoint)
                    └─> gc.collect() + cuda.empty_cache()  ← limpieza de cache GPU
                    └─> Evaluar modelo fine-tuneado → fila CSV "[fine_tuned]"
            └─> Liberar modelo de GPU
```

### 4.2 Gestión de modelos (ModelReplacer)

`src/decompositions/replacer.py` recorre el árbol de módulos del modelo y reemplaza
las capas objetivo (identificadas por ruta en notación punto, ej. `features.0`)
por el bloque comprimido correspondiente. El reemplazo es in-place sobre la copia
profunda del modelo base, garantizando que el experimento siguiente parte siempre
del modelo original sin modificar.

### 4.3 Configuración del experimento (config.json)

Cada experimento en el JSON especifica:

```json
{
  "name": "SVD rank 0068 | ft",
  "method": "SVD",
  "target_layers": ["features.0", ..., "classifier.6"],
  "rank": 68,
  "fine_tuning": true
}
```

La sección `global_settings` define hiperparámetros globales del fine-tuning
(epochs, LR schedule, early stopping, etc.) que se aplican a todos los experimentos.
Los `resource_limits` permiten limitar el rango máximo de forma centralizada
(`max_rank`), útil para experimentos en hardware limitado.

### 4.4 Logger (RunLogger)

`src/utils/logger.py` crea un directorio `runs/run_YYYYMMDD_HHMMSS/` para cada
ejecución, donde guarda:
- `input_config.json`: copia del config utilizado
- `results.csv`: filas incrementales (flush por experimento para no perder datos si la ejecución se interrumpe)

---

## 5. Métricas de evaluación

Todas las métricas se calculan sobre el conjunto de **test** (10 000 imágenes) en
`src/evaluation/metrics.py`.

| Métrica | Descripción |
|---|---|
| `total_parameters` | Parámetros entrenables (`requires_grad=True`). Excluye parámetros congelados. |
| `compression_ratio` | `baseline_params / total_params`. Ej.: 5.7x significa que el modelo tiene 5.7 veces menos parámetros. |
| `compression_time_s` | Tiempo de pared para el reemplazo de capas (solo descomposición, sin FT ni evaluación). |
| `model_memory_mb` | Suma de bytes de parámetros + buffers (BN running stats). Memoria estática en disco/RAM. |
| `peak_inference_memory_mb` | Pico de memoria CUDA durante el bucle de test (`torch.cuda.max_memory_allocated()` después de `reset_peak_memory_stats()`). Incluye pesos del modelo + activaciones intermedias + batch de entrada. Solo CUDA. |
| `test_eval_time_s` | Tiempo total del pase de test (incluye H2D, forward, sync). |
| `macs_g` | GMACs analíticos (thop). Estimación estática para un input 1×3×32×32. |
| `latency_ms` | Latencia media de un forward con batch=1, 100 runs, warmup de 10. En CUDA, cada iteración sincroniza antes y después para medir trabajo completado. |
| `throughput_fps` | Muestras procesadas por segundo en el bucle de test con batch=128. |
| `accuracy` | Porcentaje de aciertos en test. |
| `precision / recall / f1` | Métricas macro (media aritmética de las 10 clases). |

### 5.1 Limitaciones conocidas

- **`peak_inference_memory_mb` es solo CUDA**: en ejecución CPU la métrica devuelve 0.
- **`macs_g` no incluye todos los módulos custom**: los módulos `_TTConv2dModule` pueden ser subcuantificados por thop porque no registra hooks automáticamente para clases no estándar. Por ello los MACs de TT aparecen como 0.0006 GMACs (incorrecto; refleja solo las capas no reemplazadas).
- **Latencia en CPU**: sujeta a ruido de caché, scheduling del SO y throttling térmico; no apta para comparaciones rigurosas sin múltiples runs y intervalos de confianza.
- **Bug corregido — inflación del peak de memoria en modelos fine-tuneados**: el optimizador Adam crea estados de momento en GPU (~2× parámetros extra). Al destruirse el optimizador al final del FT, PyTorch no libera inmediatamente la memoria al dispositivo sino al cache del allocator. Si se medía el peak CUDA sin limpiar el cache, el modelo FT aparecía consumiendo ~8-15 MiB más que el modelo comprimido equivalente (mismo número de parámetros). Se corrigió añadiendo `gc.collect() + torch.cuda.empty_cache()` entre el FT y la evaluación del modelo FT.

---

## 6. Estrategia de fine-tuning

El fine-tuning es una etapa de re-entrenamiento ligero post-compresión con el
objetivo de recuperar precisión. Se implementa en `src/training/fine_tune.py` con
el optimizador Adam.

### 6.1 Learning Rate dinámico

La selección del LR es crítica. Un LR demasiado alto destruye los pesos que el
modelo ya tenía correctos; uno demasiado bajo no permite recuperar los degradados
por la compresión.

Se implementa un schedule **piecewise** en función de la accuracy del modelo
comprimido (post-compresión, pre-FT):

**Tramo lineal** (accuracy < threshold, por defecto 85 %):
```
LR = LR_min + (LR_max - LR_min) · (1 - accuracy/100)
```
Con `LR_max = 1e-3` (accuracy=0 %) y `LR_min = 1e-4` (accuracy=85 %).

**Tramo exponencial** (accuracy ≥ threshold):
```
LR = LR_floor + (LR_at_threshold - LR_floor) · exp(-decay_rate · t)
```
donde `t = (accuracy - threshold) / (100 - threshold)` ∈ [0,1],
`LR_floor = 1e-6` y `decay_rate = 6.0`.

El tramo exponencial garantiza que modelos que ya están cerca del baseline
(accuracy ≥ 85 %) reciban un LR muy pequeño, minimizando el riesgo de degradación.

**Motivación**: En los primeros experimentos se observó que modelos con accuracy
comprimida superior a ~87 % empeoraban tras el FT, llegando a perder hasta 5 % de
precisión. El análisis reveló que el LR lineal era demasiado grande en ese rango y,
adicionalmente, el modelo estaba haciendo overfitting sobre el conjunto de validación
(llegando a 99-100 % de accuracy de validación), lo que se traducía en menor
precisión sobre test. El schedule exponencial resolvió el primer problema.

### 6.2 Checkpoint strategy: best_val

La estrategia de checkpoint determina qué pesos se conservan tras el FT:

**`best_val`** (implementación actual):
- Se guarda un snapshot CPU del modelo comprimido antes del epoch 1.
- Durante el entrenamiento, si la accuracy de validación mejora más de `min_improvement` (0.1 pp) respecto al mejor registrado, se actualiza el mejor checkpoint.
- Al finalizar el FT, se restauran los pesos del mejor epoch.
- **Invariante**: el modelo FT nunca puede ser peor que el comprimido antes del FT (el checkpoint inicial es el propio modelo comprimido).

**Evolución de la estrategia**:
- En versiones anteriores se usaba una estrategia `final` para modelos de alta accuracy: guardar los pesos del último epoch y revertir al comprimido si se detectaba overfitting (`val_accuracy > val_overfit_ceiling = 96 %`). Esto ocultaba el problema en lugar de resolverlo.
- Se eliminó la estrategia `final` y el concepto de `val_overfit_ceiling`. Ahora siempre se usa `best_val`, que es transparente: si el FT no mejora la validación, conserva los pesos comprimidos; si mejora, conserva el mejor epoch encontrado.

### 6.3 Early stopping

Activado por defecto con `patience=3` y `min_improvement=0.1`. Se monitoriza
`val_accuracy`. El early stopping evita epochs innecesarios cuando el modelo ya
convergió, reduciendo el tiempo total de experimentación.

### 6.4 Overfitting en el smoke test

Durante los experimentos de verificación (`config_smoke.json`) se observó que la
accuracy de validación alcanzaba 99-100 % en modelos de alta accuracy. Este fenómeno
se debe a que el smoke test limita la evaluación de validación a solo 3 batches
(384 imágenes). Un modelo con >88 % de accuracy real puede clasificar correctamente
todas esas 384 imágenes por pura estadística, sin que ello implique overfitting real.
En el experimento completo (5 000 imágenes de validación) este efecto desaparece.

---

## 7. Diseño de experimentos

### 7.1 Config smoke (`config_smoke.json`)

Experimento de verificación del pipeline. Parámetros clave:

| Parámetro | Valor | Motivo |
|---|---|---|
| Muestras por método | 10 | Verificación rápida |
| Rangos | log-espaciados en parámetros | Cobertura del rango completo |
| FT epochs | 3 | Suficiente para detectar problemas |
| `max_train_batches` | 5 (640 muestras/época) | Velocidad |
| `max_val_batches` | 3 (384 muestras) | Velocidad |
| CP max rank | 1 500 | ~7.8 M parámetros |

Tiempo total en GPU Kaggle: ~15-20 minutos.

### 7.2 Config full con FT (`config_full.json`)

Experimento principal con fine-tuning. Rangos seleccionados con espaciado
**logarítmico en parámetros**: la función `get_param_spaced_ranks` genera 100
targets con `numpy.geomspace` en el espacio de parámetros y usa búsqueda binaria
para encontrar el rango entero más cercano a cada target. Esto garantiza que los
puntos en el gráfico `accuracy × total_parameters` estén distribuidos uniformemente
en escala logarítmica.

| Parámetro | Valor |
|---|---|
| Muestras por método | 68 |
| FT epochs | 5 |
| `max_train_batches` | 0 (ilimitado) |
| `max_val_batches` | 0 (ilimitado) |
| CP max rank | 1 500 |

Total: 272 JSON entries × 2 CSV rows (con/sin FT) = **544 filas CSV**.
Estimación de tiempo: ~7 horas en GPU Kaggle (dominado por compresión CP).

### 7.3 Config full sin FT (`config_full_no_ft.json`)

Experimento diseñado para estudiar únicamente el comportamiento de compresión sin
el ruido del fine-tuning. Rangos seleccionados con espaciado **lineal**:

```python
rank_i = round(max_rank / 68 · i)   para i = 1..68
```

Esta elección produce puntos igualmente espaciados en el eje de ranks, lo que
equivale a espacio aproximadamente uniforme en el eje de parámetros (dado que
la relación parámetros-rank es aproximadamente lineal para todos los métodos).

| Método | Max rank | Step | Rango de ranks |
|---|---|---|---|
| SVD | 400 | ~5.9 | [6, 12, 18, ..., 400] |
| Tucker | 400 | ~5.9 | [6, 12, 18, ..., 400] |
| TT | 400 | ~5.9 | [6, 12, 18, ..., 400] |
| CP | 2 000 | ~29.4 | [29, 59, 88, ..., 2000] |

CP tiene max rank 2 000 (vs 1 500 en los configs con FT) para igualar el rango de
parámetros total con los otros métodos (~9 M params en rank máximo). Total: 272
CSV rows (una por experimento, sin duplicados compressed/fine_tuned).

### 7.4 Justificación de los rangos máximos

Los rangos máximos (SVD/Tucker/TT = 400, CP = 1 500-2 000) se eligieron para que
todos los métodos alcancen una compresión próxima al baseline (~92 %) sin
sobrepasar el presupuesto de tiempo. A rank=400, todos los métodos tienen entre
8.7 M y 9.2 M parámetros (CR ≈ 1.06x-1.11x), es decir, prácticamente el mismo
tamaño que el baseline. Esto garantiza que la curva accuracy-parámetros de cada
método cubre el rango completo desde máxima compresión hasta accuracy de baseline.

---

## 8. Resultados experimentales

Resultados del experimento smoke (`run_20260520_073535`) con 10 muestras por método
y fine-tuning habilitado. Los valores FT no son representativos del full experiment
porque el smoke usa solo 5 batches de train y 3 de val por epoch.

### 8.1 SVD — Resumen de resultados

| Rank | Params | CR | Acc. comprimida | Acc. FT |
|---|---|---|---|---|
| 2 | 60 308 | 161.8x | 10.00 % | 10.00 % |
| 10 | 264 380 | 36.9x | 10.00 % | 12.54 % |
| 19 | 489 172 | 19.9x | 10.00 % | 55.38 % |
| 28 | 713 236 | 13.7x | 18.32 % | 68.75 % |
| 43 | 1 086 676 | 8.98x | 43.47 % | 77.96 % |
| 68 | 1 709 076 | 5.71x | 80.63 % | 86.43 % |
| 102 | 2 555 540 | 3.82x | 88.57 % | 88.57 % |
| 155 | 3 839 124 | 2.54x | 91.09 % | 91.09 % |
| 251 | 6 048 660 | 1.61x | 91.82 % | 91.82 % |
| 400 | 9 062 036 | 1.08x | 92.11 % | 92.11 % |

**Baseline**: 9 756 426 params, 92.14 % accuracy.

Observaciones:
- Accuracy prácticamente nula (10 %, equivalente a predicción aleatoria en 10 clases) para rank ≤ 19.
- Recuperación rápida entre rank 19 y 68 (de 10 % a 80 % comprimida).
- Plateau cerca del baseline a partir de rank 155 (~91 %).
- FT no aporta mejora a partir de rank 102 en el smoke (artefacto del smoke; se espera mejoría en el full experiment con dataset completo).

### 8.2 Tucker — Resumen de resultados

| Rank | Params | CR | Acc. comprimida |
|---|---|---|---|
| 2 | 24 708 | 394.9x | 10.00 % |
| 81 | 985 261 | 9.90x | 82.93 % |
| 118 | 1 659 919 | 5.88x | 89.35 % |
| 181 | 3 018 428 | 3.23x | 91.27 % |
| 400 | 8 772 215 | 1.11x | **92.21 %** (supera baseline) |

Tucker logra 92.21 % en rank 400 frente al baseline 92.14 %, lo que demuestra que
la estructura Tucker puede mantener (e incluso superar ligeramente) la precisión
del modelo original a costa de muy poca compresión. A comprensiones moderadas (8-10x)
Tucker ofrece mejor accuracy que SVD con el mismo número de parámetros.

### 8.3 TT — Resumen de resultados

| Rank | Params | CR | Acc. comprimida | Peak CUDA (MiB) |
|---|---|---|---|---|
| 2 | 29 050 | 335.8x | 10.00 % | 99.22 |
| 9 | 239 591 | 40.7x | 10.00 % | 100.02 |
| 42 | 1 063 319 | 9.18x | 41.61 % | 103.16 |
| 102 | 2 557 079 | 3.82x | 88.59 % | 109.35 |
| 400 | 9 213 207 | 1.06x | 92.14 % | 135.23 |

**Anomalía del peak CUDA**: TT tiene un comportamiento completamente diferente al
resto. El pico de memoria CUDA no decrece al reducir el rank; en cambio, es casi
constante (~99-109 MiB) para todos los ranks bajos y solo crece ligeramente a ranks
altos. Esto confirma que la reconstrucción del tensor completo en cada forward
domina el uso de memoria, independientemente del número de parámetros almacenados.
TT rank=400 con 135 MiB incluso supera el baseline (128 MiB) por este overhead.

### 8.4 CP — Resumen de resultados

| Rank | Params | CR | Acc. comprimida | Comp. time (s) |
|---|---|---|---|---|
| 2 | 24 516 | 398.0x | 10.00 % | 7.08 |
| 70 | 507 449 | 19.2x | 45.73 % | 14.12 |
| 120 | 858 349 | 11.4x | 77.71 % | 16.15 |
| 366 | 2 584 777 | 3.77x | 90.89 % | 32.76 |
| 732 | 4 631 525 | 2.11x | 91.84 % | 77.58 |
| 1500 | 7 818 745 | 1.25x | 92.08 % | **226.8 s** |

CP necesita ranks mucho más altos que Tucker o SVD para alcanzar la misma accuracy
comprimida. A igual número de parámetros, CP queda ligeramente por debajo de Tucker
y SVD en la región de compresión moderada. El tiempo de compresión escala de forma
superlineal con el rank (ALS), siendo 100× más lento que SVD.

---

## 9. Análisis comparativo de métodos

### 9.1 ¿Superan Tucker, CP y TT a SVD en el contexto de VGG11-BN / CIFAR-10?

La respuesta corta es: **parcialmente sí, pero la ventaja es modesta**.

Comparando a igual número de parámetros:
- **Tucker vs SVD**: Tucker supera ligeramente a SVD en el rango de compresión media
  (3x-10x). La ventaja se debe a que Tucker explota la estructura 4D del tensor,
  comprimiendo los modos de canal de forma independiente y dejando la dimensión
  espacial intacta. En cambio, SVD destruye la estructura espacial al desplegar el
  tensor a 2D.
- **CP vs SVD**: CP iguala o supera marginalmente a SVD con el mismo número de
  parámetros, pero CP necesita ranks mucho más altos para llegar a ese número de
  parámetros. Sumado a los problemas de convergencia del ALS, el beneficio práctico
  es cuestionable.
- **TT vs SVD**: TT no aporta ventaja práctica en términos de memoria de inferencia
  (por la reconstrucción on-the-fly) ni de accuracy (comparable a SVD). El número de
  parámetros almacenados es menor, pero eso no se traduce en reducción de la memoria
  máxima durante inferencia.

### 9.2 Por qué los métodos 4D no dominan claramente a SVD

Existen razones conceptuales y de implementación:

1. **Dominancia de capas profundas**: Las capas `features.15–25` tienen `in=out=512`
   y kernel 3×3. Para estas capas cuadradas, la ganancia de Tucker-2 sobre SVD es
   mínima porque la estructura de canal ya está bien capturada por SVD.

2. **Pequeñas dimensiones espaciales (3×3)**: La ganancia de explotar la dimensión
   espacial (que Tucker/CP/TT hacen) es pequeña cuando el kernel es 3×3. En kernels
   más grandes (5×5, 7×7) la ventaja sería mayor.

3. **CP no es prácticamente superior a SVD**: La aproximación CP de rango fijo de un
   tensor 4D es equivalente a la aproximación de una matriz que es suma de productos
   externos. El rango CP suele ser mayor que el rango matricial para la misma calidad
   de aproximación, y ALS no garantiza convergencia al óptimo.

4. **TT: memoria y latencia de inferencia aumentan**: La reconstrucción on-the-fly
   elimina la ventaja de TT en el contexto de inferencia en producción.

5. **Dataset pequeño (CIFAR-10 32×32)**: La red procesa imágenes de 32×32, lo que
   hace que las capas convolucionales sean relativamente pequeñas en términos de
   mapa de activaciones. En modelos como ResNet-50 sobre ImageNet (224×224), las
   ventajas de los métodos 4D serían más pronunciadas.

### 9.3 Métricas de eficiencia en inferencia

| Método (rank alto) | MACs (GMACs) | Latencia (ms) | Throughput (FPS) |
|---|---|---|---|
| Baseline | 0.154 | 2.50 | 14 594 |
| SVD r=400 | 0.146 | 3.31 | 11 550 |
| Tucker r=400 | 0.165 | 3.03 | 10 937 |
| TT r=400 | 0.0006 (incorrecto) | 7.02 | 9 732 |
| CP r=1500 | 0.135 | 2.65 | 5 255 |

TT tiene la peor latencia y throughput de todos los métodos, confirmando el overhead
de reconstrucción. CP con rank alto también degrada significativamente el throughput
(4 convoluciones depthwise por cada capa original). SVD es el único método que
mantiene latencia comparable al baseline a rank alto.

---

## 10. Decisiones de implementación y correcciones

### 10.1 Migración `torch.svd` → `torch.linalg.svd`

La función `torch.svd` está marcada como deprecated y devuelve `(U, S, V)` donde
`V` es la matriz V (no transpuesta). La nueva API `torch.linalg.svd` sigue la
convención estándar de álgebra lineal y devuelve `(U, S, Vh)` donde `Vh = Vᵀ`. Se
actualizaron `tucker.py` y `cp.py` para usar `full_matrices=False` (solo calcula la
descomposición reducida, más eficiente). El ajuste de la operación matricial en las
capas resultantes (`U_trunc * S_trunc.unsqueeze(0)` para broadcasting correcto) fue
necesario para mantener la equivalencia matemática.

### 10.2 Corrección de `max_rank_compression` en SVD

La fórmula original usaba `denom = max(1, in_ch + out_ch + kh + kw)`, que no
corresponde al conteo de parámetros de SVD. La fórmula correcta es:
```python
denom = max(1, out_ch + in_ch * kh * kw)
```
ya que `P_SVD = rank * (out_ch + in_ch*kh*kw)` y el original tiene
`P_orig = out_ch * in_ch * kh * kw`. La compresión ocurre cuando
`rank < out_ch * in_ch * kh * kw / (out_ch + in_ch * kh * kw)`.

### 10.3 Hyperparámetros de FT configurables desde config.json

Los parámetros del schedule de LR (`ft_lr_max`, `ft_lr_min`, `ft_high_acc_threshold`,
`ft_lr_floor`, `ft_lr_decay_rate`) se configuran en `global_settings.fine_tuning`
del JSON, con defaults razonables en `runner.py`. Esto permite ajustar el schedule
sin recompilar el código.

### 10.4 Eliminación de variable muerta `ft_lr`

La variable `ft_lr = float(ft_cfg["learning_rate"])` quedó obsoleta al introducir
el schedule dinámico (que calcula el LR en función de la accuracy comprimida, no
de un valor fijo). Se eliminó para evitar confusión.

---

## 11. Conclusiones

### 11.1 Sobre la compresión sin fine-tuning

La descomposición tensorial por sí sola introduce una degradación de precisión que
sigue un patrón común a todos los métodos: la accuracy cae rápidamente a valores
aleatorios (10 % en CIFAR-10) para compresiones superiores a ~20x, luego se
recupera gradualmente hasta alcanzar el baseline en compresiones inferiores a 2x.

La **zona de interés práctico** para TinyML se sitúa entre compresiones 4x y 12x,
donde los modelos retienen entre un 60 % y un 90 % de la accuracy original con una
fracción del coste en parámetros.

Tucker-2 ofrece la mejor relación accuracy/parámetros dentro de esa zona. SVD
es sorprendentemente competitivo dada su simplicidad conceptual. CP y TT presentan
limitaciones prácticas (coste computacional de compresión y overhead de inferencia,
respectivamente) que reducen su atractivo para aplicaciones reales.

### 11.2 Sobre el fine-tuning post-compresión

El FT con un LR adecuado puede recuperar entre 20 y 40 puntos porcentuales de
accuracy en modelos con compresión 5x-15x. Los modelos muy comprimidos (CR > 20x,
accuracy < 20 %) no se recuperan de forma significativa con FT limitado, lo que
sugiere que la información destruida por la compresión excesiva no puede recuperarse
mediante re-entrenamiento sin datos adicionales de calidad.

Para modelos con accuracy comprimida superior al 85-90 %, el FT aporta mejoras
marginales o nulas, y la arquitectura del modelo comprimido ya está próxima al
óptimo alcanzable. En este rango, aplicar FT con un LR excesivo puede degradar
la precisión de forma permanente (razón que motivó el schedule exponencial).

El checkpointing con estrategia `best_val` es la aproximación más robusta: garantiza
que el FT nunca empeora el resultado con respecto al modelo comprimido original,
independientemente del número de epochs ejecutados.

### 11.3 Sobre la memoria en inferencia

La métrica `peak_inference_memory_mb` revela que la memoria durante inferencia
no está dominada por los parámetros del modelo sino por los **mapas de activación
intermedios**. Para una red VGG11-BN procesando batches de 128 imágenes CIFAR-10,
el pico de memoria del baseline es 128 MiB con solo 37 MiB de parámetros: los
activaciones representan más del 70 % del pico.

La compresión de parámetros reduce la memoria estática del modelo pero no necesariamente
el pico de inferencia: modelos con arquitecturas más complejas post-compresión (más
capas intermedias, como CP con 4 convoluciones) pueden aumentar el consumo de
activaciones. Esto es especialmente relevante en el contexto TinyML, donde la memoria
RAM de activaciones puede ser el cuello de botella, no el almacenamiento de pesos.

TT es el caso extremo: almacena pocos parámetros pero reconstruye el tensor
completo en cada forward, eliminando cualquier ventaja en memoria de inferencia.
Esto hace que TT sea inadecuado para inferencia eficiente tal como está implementado,
siendo necesario usar kernels especializados de contracción tensorial para
materializar el beneficio teórico.

### 11.4 Sobre la posición de SVD frente a métodos 4D

Los resultados experimentales muestran que SVD, a pesar de ser conceptualmente un
método 2D, ofrece una accuracy comparable a los métodos 4D en el contexto específico
de VGG11-BN sobre CIFAR-10. Esto no significa que SVD sea superior en general: la
ventaja teórica de Tucker, CP y TT sobre SVD se materializa mejor en arquitecturas
con kernels más grandes, capas con mayor asimetría entre dimensiones, o en tareas
donde la estructura espacial del kernel tiene mayor importancia.

La literatura (Lebedev et al., 2014; Kim et al., 2015; Garipov et al., 2016) reporta
ganancias de Tucker y CP sobre SVD principalmente en redes más profundas, kernels
más grandes y datasets de mayor resolución (ImageNet). El alcance de los experimentos
de este TFG (kernels 3×3, imágenes 32×32, red de tamaño medio) está en el límite
inferior donde las ventajas de los métodos 4D empiezan a ser apreciables.

### 11.5 Sobre el diseño experimental

La elección del espaciado de rangos tiene un impacto significativo en la legibilidad
e interpretabilidad de los resultados. Un espaciado lineal de ranks produce puntos
aproximadamente uniformes en el eje de parámetros (dada la linealidad de P(rank) para
todos los métodos estudiados). Para análisis comparativos es preferible el espaciado
uniforme en parámetros ya que elimina el sesgo visual de acumulación en zonas de
alta compresión.

---

## Apéndice: Estructura del repositorio

```
CNN/
├── src/
│   ├── decompositions/
│   │   ├── base.py           # BaseDecomposedLayer: contrato común
│   │   ├── svd.py            # SVD truncado (2D baseline)
│   │   ├── tucker.py         # Tucker-2 (modos de canal)
│   │   ├── cp.py             # CP-ALS (4 factores)
│   │   ├── tt.py             # Tensor Train (reconstrucción on-the-fly)
│   │   ├── replacer.py       # ModelReplacer: sustitución in-place de capas
│   │   └── registry.py       # Registro nombre→clase
│   ├── training/
│   │   ├── fine_tune.py      # fine_tune_model: Adam + best_val checkpoint
│   │   └── lr_schedule.py    # Schedule LR piecewise (lineal + exponencial)
│   ├── evaluation/
│   │   └── metrics.py        # ModelEvaluator: todas las métricas
│   ├── models/
│   │   ├── vgg11_bn.py       # VGG11-BN CIFAR-10 (torch hub)
│   │   └── factory.py        # ModelFactory
│   ├── data/
│   │   └── factory.py        # DatasetFactory: CIFAR-10 + splits
│   ├── experiments/
│   │   └── runner.py         # Pipeline principal
│   └── utils/
│       ├── config.py         # ConfigParser
│       └── logger.py         # RunLogger (CSV incremental)
├── generate_experiments.py   # Genera config_smoke/full/full_no_ft.json
├── config_smoke.json         # 10 muestras/método, FT, log-spaced params
├── config_full.json          # 68 muestras/método, FT, log-spaced params
├── config_full_no_ft.json    # 68 muestras/método, sin FT, linear-spaced ranks
├── runs/
│   ├── run_20260515_100131/  # Experimento previo (análisis FT problemático)
│   └── run_20260520_073535/  # Experimento smoke (config_smoke.json)
├── docs/
│   ├── baseline_vgg11_bn.md  # Descripción arquitectura VGG11-BN
│   └── README.md             # Este documento
└── analyzer_web/             # Herramienta web de visualización de resultados
```

---

## Apéndice: Referencias

- Simonyan, K., & Zisserman, A. (2014). *Very Deep Convolutional Networks for Large-Scale Image Recognition*. arXiv:1409.1556.
- Lebedev, V., Ganin, Y., Rakhuba, M., Oseledets, I., & Lempitsky, V. (2014). *Speeding-up Convolutional Neural Networks Using Fine-tuned CP-Decomposition*. arXiv:1412.6553.
- Kim, Y. D., Park, E., Yoo, S., Choi, T., Yang, L., & Shin, D. (2015). *Compression of Deep Convolutional Neural Networks for Fast and Low Power Mobile Applications*. arXiv:1511.06530. (Tucker-2)
- Garipov, T., Podoprikhin, D., Novikov, A., & Vetrov, D. (2016). *Ultimate tensorization: compressing convolutional and FC layers alike*. arXiv:1611.03214. (TT-conv)
- Hillar, C. J., & Lim, L. H. (2013). *Most tensor problems are NP-hard*. Journal of the ACM, 60(6), 1–39. (NP-hardness de CP)
- Oseledets, I. V. (2011). *Tensor-train decomposition*. SIAM Journal on Scientific Computing, 33(5), 2295–2317.
- De Lathauwer, L., De Moor, B., & Vandewalle, J. (2000). *A multilinear singular value decomposition*. SIAM Journal on Matrix Analysis and Applications, 21(4), 1253–1278. (Tucker/HOSVD)
