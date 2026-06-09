# Métricas del experimento de compresión

Este documento describe todas las métricas que se registran en cada experimento de compresión de redes neuronales, cómo se calculan conceptualmente y en qué gráficos aparecen.

---

## Estructura general del CSV de resultados

Cada fila del fichero `results.csv` corresponde a **una configuración de experimento** (un método de descomposición, un rango, y si ha habido fine-tuning o no). Las columnas son:

```
experiment_name, method, target_layers, rank, fine_tuning_enabled,
fine_tuning_time_s, total_parameters, compression_ratio, compression_time_s,
model_memory_mb, peak_inference_memory_mb, test_eval_time_s,
macs_g, latency_ms, throughput_fps,
accuracy, precision, recall, f1_score
```

Siempre hay una fila especial, el **Baseline**, que corresponde al modelo original sin comprimir y sirve de referencia para todas las comparaciones.

---

## 1. Parámetros del modelo (`total_parameters`)

**¿Qué mide?** El número total de parámetros entrenables del modelo comprimido (pesos de convoluciones, capas lineales, batch normalization, etc.).

**¿Cómo se computa?** Se suman todos los elementos de todos los tensores de parámetros del modelo tras aplicar la descomposición:

$$P = \sum_{l \in \text{capas}} |\theta_l|$$

donde $|\theta_l|$ denota el número de parámetros de la capa $l$ (por ejemplo, para una capa convolucional estándar con $C_{out}$ filtros de tamaño $C_{in} \times k \times k$, su contribución es $C_{out} \cdot C_{in} \cdot k \cdot k$).

Cuando se aplica una descomposición tensorial de rango $r$, cada capa descompuesta sustituye sus pesos originales por un conjunto de tensores más pequeños, reduciendo así el total. El cálculo exacto depende del método (véase [`DECOMPOSITIONS.md`](README.md) para las fórmulas por método).

**¿Dónde aparece?**
- Plots de **Accuracy vs Parámetros** (`accuracy_vs_parameters.png`, `accuracy_vs_parameters_by_ft.png`): eje X en millones de parámetros.
- Plots de **Memoria vs Parámetros** (`memory_vs_compression.png`): panel (a), eje Y.

---

## 2. Ratio de compresión (`compression_ratio`)

**¿Qué mide?** Cuántas veces más pequeño es el modelo comprimido respecto al modelo original en términos de número de parámetros. Es la métrica principal para cuantificar el nivel de compresión.

**¿Cómo se computa?**

$$CR = \frac{P_{\text{baseline}}}{P_{\text{comprimido}}}$$

donde $P_{\text{baseline}}$ es el número de parámetros del modelo original y $P_{\text{comprimido}}$ es el número de parámetros del modelo comprimido.

Un $CR = 1.0$ significa que el modelo no ha sido comprimido (el baseline). Un $CR = 4.0$ significa que el modelo tiene 4 veces menos parámetros. Valores muy altos (e.g., $CR > 50$) indican compresión agresiva y generalmente implican una pérdida notable de precisión.

> **Nota sobre el break-even:** existe un rango máximo por encima del cual la descomposición produce *más* parámetros que el original. Este límite se llama *break-even rank* y se computa analíticamente antes de lanzar el experimento para evitar configuraciones sin sentido.

**¿Dónde aparece?**
- Es el **eje X principal** de los plots de `accuracy_vs_compression.png`, `accuracy_vs_compression_by_ft.png`, `compute_vs_compression.png`, `tradeoff_map.png`.

---

## 3. Precisión del clasificador (`accuracy`, `precision`, `recall`, `f1_score`)

Estas cuatro métricas evalúan la calidad del modelo como clasificador, medidas sobre el conjunto de test completo (CIFAR-10: 10.000 imágenes de 10 clases).

### 3.1 Accuracy (`accuracy`)

**¿Qué mide?** La fracción de imágenes del conjunto de test clasificadas correctamente.

$$\text{Accuracy} = \frac{\text{Número de predicciones correctas}}{\text{Total de muestras}} \times 100$$

Es la métrica más directa y la que se usa **en todos los plots que comparan modelos**. El baseline tiene un accuracy de referencia (e.g., 94.25%) y el objetivo de la compresión es mantenerse lo más cerca posible de él.

### 3.2 Precision (`precision`)

**¿Qué mide?** De todas las muestras que el modelo predice como clase $c$, ¿cuántas realmente son de la clase $c$? Promediado sobre todas las clases (macro-average):

$$\text{Precision} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FP_c}$$

donde $TP_c$ son los verdaderos positivos de la clase $c$ y $FP_c$ son los falsos positivos. Con 10 clases balanceadas, en condiciones ideales (Baseline correcto) Precision ≈ Accuracy.

### 3.3 Recall (`recall`)

**¿Qué mide?** De todas las muestras que realmente son de la clase $c$, ¿cuántas identifica correctamente el modelo? Promediado (macro-average):

$$\text{Recall} = \frac{1}{C} \sum_{c=1}^{C} \frac{TP_c}{TP_c + FN_c}$$

donde $FN_c$ son los falsos negativos de la clase $c$. Igual que precision, con clases balanceadas Recall ≈ Accuracy cuando el modelo funciona bien.

### 3.4 F1-Score (`f1_score`)

**¿Qué mide?** La media armónica de precision y recall, que equilibra ambas métricas. Útil cuando precision y recall divergen.

$$F_1 = \frac{1}{C} \sum_{c=1}^{C} \frac{2 \cdot \text{Precision}_c \cdot \text{Recall}_c}{\text{Precision}_c + \text{Recall}_c}$$

Con CIFAR-10, que tiene clases perfectamente balanceadas, las cuatro métricas de clasificación tienden a converger. La divergencia entre ellas en modelos muy comprimidos indica que el modelo colapsa: predice siempre una misma clase (precision baja, recall alto para esa clase, recall ~0 para las demás).

**¿Dónde aparece?**
- `accuracy` es el **eje Y** en prácticamente todos los plots: `accuracy_vs_compression.png`, `accuracy_vs_parameters.png`, `tradeoff_map.png`, `pareto_dashboard.png`, `pareto_3d.png`, `pareto_bubble.png`, `memory_vs_compression.png`.
- `precision`, `recall`, `f1_score` están en el CSV pero no se representan explícitamente en ningún plot; sirven como métricas de diagnóstico auxiliares.

---

## 4. Tiempo de compresión (`compression_time_s`)

**¿Qué mide?** El tiempo en segundos que tarda el proceso de descomposición tensorial en sí, es decir, el tiempo necesario para factorizar los pesos de todas las capas objetivo.

**¿Cómo se computa?** Se mide el tiempo de reloj (wall clock time) desde que se inicia la descomposición de la primera capa hasta que termina la última. **No incluye** el fine-tuning posterior.

Este tiempo **depende directamente del rango $r$** elegido: a mayor rango, la descomposición tiene que manipular tensores más grandes, aunque en la práctica el tiempo crece de forma no lineal y depende mucho del método (SVD utiliza álgebra lineal estándar; CP requiere algoritmos iterativos como ALS que pueden ser lentos a rangos altos; Tucker es intermedio).

**¿Dónde aparece?**
- Plot `compute_vs_compression.png`, panel **(a)**: eje Y en escala logarítmica, frente al ratio de compresión en el eje X. Se usa la escala logarítmica porque los tiempos pueden abarcar varios órdenes de magnitud (de milisegundos a varios minutos).

---

## 5. Tiempo de fine-tuning (`fine_tuning_time_s`)

**¿Qué mide?** El tiempo en segundos que dura el proceso de fine-tuning (ajuste fino) tras la compresión. Solo es no-cero en las filas donde `fine_tuning_enabled = True`.

**¿Cómo se computa?** Tiempo de reloj total de todas las épocas de fine-tuning, incluyendo el tiempo de forward pass, backward pass, actualización de pesos y validación por época. Si el early stopping detiene el entrenamiento antes, el tiempo registrado es el de las épocas efectivamente ejecutadas.

**¿Dónde aparece?** Solo en el CSV, no se representa directamente en ningún plot. Es útil para estimar el coste total de la compresión (compresión + fine-tuning) y para comparar el overhead de recuperación de precisión entre métodos.

---

## 6. Memoria del modelo (`model_memory_mb`)

**¿Qué mide?** El tamaño en megabytes que ocupan los **pesos del modelo** almacenados en memoria, asumiendo representación en `float32` (4 bytes por parámetro).

**¿Cómo se computa?**

$$\text{Model Memory (MB)} = \frac{P \times 4 \text{ bytes}}{1024^2}$$

donde $P$ es el número total de parámetros. Es una métrica puramente estática: no depende de si el modelo está realizando inferencia o no, solo del número de parámetros. Es directamente proporcional a `total_parameters`.

> **Distinción importante:** `model_memory_mb` mide el tamaño de los pesos en reposo. Esto es diferente de la memoria pico durante la inferencia (`peak_inference_memory_mb`), que incluye adicionalmente los mapas de activación intermedios.

**¿Dónde aparece?** En el CSV como referencia. No se usa directamente en los plots principales (se prefiere `peak_inference_memory_mb` por ser más representativo del consumo real).

---

## 7. Memoria pico de inferencia (`peak_inference_memory_mb`)

**¿Qué mide?** El pico máximo de memoria GPU (o RAM) consumida durante una pasada de inferencia completa sobre un batch. Incluye no solo los pesos del modelo sino también los **mapas de activación intermedios** de todas las capas durante el forward pass.

**¿Cómo se computa?** Se registra el pico de memoria de la GPU (usando `torch.cuda.max_memory_allocated()`) durante la inferencia de un batch de tamaño fijo. El valor depende tanto de la arquitectura del modelo como del tamaño del batch.

Esta métrica es mucho más relevante que `model_memory_mb` para caracterizar el consumo real en producción, ya que los mapas de activación pueden dominar el consumo de memoria en redes profundas, especialmente en las capas iniciales con resolución espacial alta.

**¿Dónde aparece?**
- Plot `memory_vs_compression.png`: eje X de ambos paneles. Muestra cómo evoluciona el consumo de memoria real con el nivel de compresión.
- Plot `pareto_dashboard.png`: eje X del panel 1 (Accuracy vs Memoria) y eje X del panel 3 (Memoria vs Latencia).
- Plot `pareto_3d.png`: eje X del espacio 3D.
- Plot `pareto_bubble.png`: el **tamaño de la burbuja** es proporcional a `peak_inference_memory_mb`. A mayor burbuja, más memoria consume ese punto.

---

## 8. MACs (`macs_g`)

**¿Qué mide?** El número de **operaciones de multiplicación-acumulación** (*Multiply-Accumulate Operations*, MACs) necesarias para una pasada de inferencia. Se expresa en giga-MACs (GMACs, es decir, $10^9$ MACs).

**¿Cómo se computa?** Se contabiliza analíticamente el número de operaciones aritméticas del grafo computacional del modelo dado un tensor de entrada de dimensión fija (una imagen de CIFAR-10: $3 \times 32 \times 32$). Para una capa convolucional estándar con $C_{out}$ filtros de tamaño $k \times k$, aplicada sobre un mapa de entrada $C_{in} \times H \times W$:

$$\text{MACs}_{\text{conv}} = C_{out} \cdot C_{in} \cdot k^2 \cdot H_{out} \cdot W_{out}$$

Los MACs son la medida estándar del coste computacional teórico de un modelo. A diferencia de la latencia, los MACs son independientes del hardware y del software de ejecución.

> **MACs vs FLOPs:** Frecuentemente se confunden. 1 MAC = 1 multiplicación + 1 acumulación = 2 FLOPs. En muchos papers se usan indistintamente; aquí se reportan MACs.

**¿Dónde aparece?** En el CSV. No se representa explícitamente en los plots actuales, aunque es un indicador complementario al `latency_ms`. La relación entre MACs y latencia real puede ser no lineal por efectos de caché, paralelismo y eficiencia del hardware.

---

## 9. Latencia de inferencia (`latency_ms`)

**¿Qué mide?** El tiempo en milisegundos que tarda el modelo en procesar **un batch completo** de imágenes durante la inferencia. Es una medida de rendimiento en tiempo real.

**¿Cómo se computa?** Se realiza un warm-up previo (para estabilizar la GPU y los cachés del sistema), y luego se mide el tiempo total de múltiples pasadas de inferencia, promediando el resultado:

$$\text{Latency (ms)} = \frac{\text{Tiempo total de N pasadas}}{N} \times 1000$$

La latencia real depende del hardware (GPU/CPU, frecuencia, memoria de caché), del tamaño del batch, y de si el modelo tiene operaciones que se benefician del paralelismo masivo. Un modelo con muchos MACs no necesariamente tiene mayor latencia si sus operaciones son altamente paralelizables (por ejemplo, convoluciones grandes en GPU).

> **Nota importante:** Tras la descomposición tensorial, un modelo comprimido puede tener *más* latencia que el original aunque tenga *menos* parámetros y MACs. Esto ocurre porque la descomposición introduce capas adicionales con tamaños no estándar que el hardware no paraleliza tan eficientemente. Por este motivo la latencia se muestra junto a la precisión.

**¿Dónde aparece?**
- Plot `compute_vs_compression.png`, panel **(b)**: barras de latencia media ± desviación estándar por método (sin fine-tuning, ya que la latencia depende solo de la estructura del modelo, no de los pesos).
- Plot `pareto_dashboard.png`: eje X del panel 2 (Accuracy vs Latencia), eje Y del panel 3 (Memoria vs Latencia).
- Plot `pareto_3d.png`: eje Y del espacio 3D.
- Plot `pareto_bubble.png`: eje X del plot 2D.

---

## 10. Throughput (`throughput_fps`)

**¿Qué mide?** El número de imágenes que el modelo puede procesar por segundo (*frames per second*, FPS). Es la inversa de la latencia por muestra.

**¿Cómo se computa?** A partir de la latencia medida sobre el batch completo de tamaño $B$:

$$\text{Throughput (FPS)} = \frac{B}{\text{Latency (s)}} = \frac{B \times 1000}{\text{Latency (ms)}}$$

Un throughput mayor indica que el modelo puede procesar más imágenes por unidad de tiempo, lo cual es relevante para aplicaciones en tiempo real o procesamiento masivo de datos.

**¿Dónde aparece?** En el CSV como métrica complementaria de rendimiento. No se representa en los plots actuales, aunque conceptualmente es equivalente a la inversa de `latency_ms`.

---

## 11. Tiempo de evaluación en test (`test_eval_time_s`)

**¿Qué mide?** El tiempo total en segundos necesario para evaluar el modelo sobre todo el conjunto de test (10.000 imágenes en CIFAR-10). Incluye los forward passes de todos los batches y el cómputo de las métricas de clasificación.

**¿Cómo se computa?** Tiempo de reloj de principio a fin de la evaluación completa del test set. Es proporcional al tamaño del dataset y a la latencia por batch, pero también incluye el overhead de Python para calcular accuracy, precision, recall y F1.

**¿Dónde aparece?** Solo en el CSV. Es un indicador del coste de evaluación y puede usarse para estimar el tiempo necesario para validar nuevas configuraciones.

---

## Resumen: métricas por gráfico

| Gráfico | Eje X | Eje Y | Eje Z / Tamaño | Filtra |
|---|---|---|---|---|
| `accuracy_vs_compression.png` | `compression_ratio` | `accuracy` | — | FT y No FT por separado |
| `accuracy_vs_compression_by_ft.png` | `compression_ratio` | `accuracy` | — | Ambos en mismo plot |
| `accuracy_vs_parameters.png` | `total_parameters` | `accuracy` | — | FT y No FT por separado |
| `accuracy_vs_parameters_by_ft.png` | `total_parameters` | `accuracy` | — | Ambos en mismo plot |
| `compute_vs_compression.png` (a) | `compression_ratio` | `compression_time_s` (log) | — | Solo sin FT |
| `compute_vs_compression.png` (b) | método | `latency_ms` (media±SD) | — | Solo sin FT |
| `memory_vs_compression.png` (a) | `peak_inference_memory_mb` | `total_parameters` (M) | — | Solo sin FT |
| `memory_vs_compression.png` (b) | `peak_inference_memory_mb` | `accuracy` | — | Solo sin FT |
| `tradeoff_map.png` | `compression_ratio` | `accuracy` | — | FT y No FT con cuadrantes |
| `pareto_dashboard.png` (1) | `peak_inference_memory_mb` | `accuracy` | — | Solo con FT |
| `pareto_dashboard.png` (2) | `latency_ms` | `accuracy` | — | Solo con FT |
| `pareto_dashboard.png` (3) | `peak_inference_memory_mb` | `latency_ms` | tamaño ∝ `accuracy` | Solo con FT |
| `pareto_bubble.png` | `latency_ms` | `accuracy` | tamaño ∝ `peak_inference_memory_mb` | Solo con FT |
| `pareto_3d.png` | `peak_inference_memory_mb` | `latency_ms` | `accuracy` (eje Z) | Solo con FT |

---

## Decisiones de diseño notables

### Fine-tuning vs. sin fine-tuning
Muchos plots muestran dos curvas por algoritmo: con fine-tuning (FT) y sin él. La diferencia ilustra la **capacidad de recuperación de precisión** mediante fine-tuning. En general, el fine-tuning es esencial para rangos bajos (alta compresión) donde la descomposición destruye mucha información. A rangos altos (baja compresión), ambas curvas convergen al baseline.

### Escala logarítmica en tiempos de compresión
El plot de `compression_time_s` usa escala logarítmica en el eje Y porque los tiempos varían varios órdenes de magnitud: SVD (algebra lineal directa) puede completarse en décimas de segundo, mientras que CP (que usa ALS iterativo) puede tardar decenas de segundos a rangos altos.

### Filtro de Pareto en memoria
En el plot de memoria (`memory_vs_compression.png`), se aplica un filtro Pareto sobre las curvas: solo se representan las configuraciones donde reducir parámetros efectivamente reduce la memoria pico. Esto elimina artefactos de medición donde configuraciones con más parámetros tienen paradójicamente menos memoria pico (algo que puede ocurrir por diferencias en el tamaño de los mapas de activación intermedios).

### Cuadrantes del Trade-off Map
El `tradeoff_map.png` divide el espacio (Compression Ratio, Accuracy) en cuatro zonas usando como umbrales la mediana del ratio de compresión y el 97% de la accuracy del baseline:
- **Ideal**: alta compresión *y* alta precisión — la esquina superior derecha.
- **Accuracy-first**: alta precisión pero poca compresión — poco comprimido pero fiel.
- **Compression-first**: alta compresión pero precisión reducida — útil en edge computing.
- **Weak**: ni comprime bien ni mantiene la precisión — configuraciones a evitar.
