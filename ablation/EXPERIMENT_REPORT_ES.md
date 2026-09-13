# Estudio de Ablación de Capas (Action Expert): Informe Técnico

Este documento detalla el diseño experimental, los protocolos de evaluación, los registros brutos de ejecución y los resultados empíricos del **Estudio de Ablación de Capas** (*Action Expert Layercut*) realizado sobre el modelo Vision-Language-Action SmolVLA.

---

## 1. Diseño y Realización del Experimento

### Objetivo y Motivación Teórica
El Action Expert de SmolVLA consta de 16 capas transformer encargadas de procesar tokens visuales-lingüísticos y decodificar autoregresivamente las acciones de la trayectoria articular del robot. Para integrar nuevas modalidades sensoriales (como nubes de puntos de profundidad estéreo en SmolVLA-D y memoria visual en SmolVLA-MD) sin degradar la distribución de acciones preentrenada, es indispensable mapear empíricamente qué capas presentan tolerancia natural frente a cuáles constituyen cuellos de botella estructuralmente críticos.

```
Action Expert de SmolVLA (16 Capas Transformer):
┌───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┬───────┐
│ L0    │ L1    │ L2    │ L3    │ L4    │ L5    │ L6    │ L7    │ L8    │ L9    │ L10   │ L11   │ L12   │ L13   │ L14   │ L15   │
├───────┼───────┴───────┴───────┼───────┼───────┼───────┴───────┴───────┼───────┼───────┼───────┼───────┼───────┴───────┼───────┤
│ Entr. │    CRÍTICAS (1-3)     │ Inter │ Crit  │ VENTANA TOLERANTE(6-8)│ Trans │ Trans │ Inter │ Trans │  CRÍTICAS(13-14)│ Sal.  │
└───────┴───────────────────────┴───────┴───────┴───────────────────────┴───────┴───────┴───────┴───────┴───────────────┴───────┘
                                                ▲ Ventana de Inyección Residual (SmolVLA-D)
```

### Mecanismo de Ablación (`smolvla_layercut`)
- **Passthrough por Conexión de Identidad (Skip-Block):** En lugar de anular pesos a cero o podar módulos (lo que descalibra las estadísticas de activación intermedia), las capas ablacionadas se sustituyen por `None` en `get_model_layers`. El tensor de activaciones atraviesa la capa intacto a través de la conexión residual pura, sin computar atención ni bloques MLP.
- **Parche de Denoising sin Prefijo:** En `forward_attn_layer` se gestiona el flujo para evitar errores de listas vacías en `torch.cat` cuando `inputs_embeds=[None, suffix]` durante los pasos de difusión o flow-matching.
- **Checkpoint del Modelo:** Todas las evaluaciones físicas emplearon el checkpoint entrenado para la tarea real (`outputs/train/tfm_layer_ablation_expert_only_v3/checkpoints/last/pretrained_model`), entrenado sobre los datasets físicos combinados (`Esk1z0/tfm_layer_ablation_batch_1` + `batch_2`).
- **Verificación de Sanidad:** Cada configuración se comprobó previamente con [`verify_layercut.py`](verify_layercut.py) para certificar que exactamente las capas deseadas quedaban enmascaradas antes de interactuar con el robot.

### Fases del Estudio y Configuraciones (21 Modelos, 315 Evaluaciones Físicas)
El estudio se estructuró en tres niveles complementarios:

1. **Fase 0 — Barrido Numérico Offline (`xai_ablation.py`):**
   - Evaluación de 135 combinaciones de ventanas continuas (tamaño de ventana de 1 a 15 capas sobre todas las posiciones válidas) sobre un batch multimodal sintético.
   - Medición de la divergencia en la predicción de acciones (MSE y MAE) respecto al baseline sin modificar para mapear la sensibilidad relativa antes de ocupar tiempo en el robot físico.
2. **Fase 1 — Ablación Física de Capa Única:**
   - 17 configuraciones físicas: el modelo `base` de referencia sin ablacionar más 16 modelos con ablación individual (`layercut_0` a `layercut_15`).
3. **Fase 2 — Ablación Física de Rango Multicapa:**
   - 4 configuraciones continuas (`layercut_range_6_7`, `layercut_range_6_7_8`, `layercut_range_7_8_9`, `layercut_range_9_10_11`) evaluando la ventana tolerante y sus fronteras para contrastar si la tolerancia individual se preserva al desactivar bloques contiguos.

### Setup de Hardware y Escenarios de Evaluación
- **Hardware Robótico:** Brazo articulado de 6 grados de libertad **SO-101**, equipado con pinza paralela, cámara cenital de contexto y cámara estéreo dual SVPRO (2560x800) montada en la muñeca.
- **Diseño de Evaluación Física:** Evaluado en **15 intentos físicos por modelo** (14 intentos en `layercut_6` y `layercut_13` debido a datos faltantes en una ejecución; total de 315 ejecuciones físicas).
- **Escenarios Controlados:** Cada modelo ejecutó 3 disposiciones físicas fijas (5 repeticiones por condición):
  - `training`: Disposiciones estándar vistas en la distribución de entrenamiento.
  - `no_visto`: Posiciones no vistas que exigen generalización espacial.
  - `stress`: Configuraciones complejas con objetos muy juntos o rotaciones críticas.
- Cuatro objetos sobre la rejilla 3x3 (`A1` a `C3`): Estrella Negra (`starBlack`), Estrella Naranja (`starOrange`), Cubo Negro (`cubeBlack`) y Cubo Naranja (`cubeOrange`).

---

## 2. Protocolo y Rúbrica de Evaluación

La tarea evaluada consiste en ordenar los 4 objetos: agarrar y depositar las 2 estrellas dentro del recipiente, y agarrar y posicionar los 2 cubos sobre sus marcas delimitadas.

### 9 Criterios Binarios de Evaluación
Cada intento físico se califica mediante 9 criterios binarios estrictos (1 = cumplido, 0 = no cumplido):

```
                                 Rúbrica Binaria de 9 Puntos
 ┌───────────────────────────────────────┬───────────────────────────────────────┐
 │        Ordenación de Estrellas (4 pts)│         Colocación de Cubos (4 pts)   │
 ├───────────────────────────────────────┼───────────────────────────────────────┤
 │ 1. Agarre y elevación Estrella Negra  │ 5. Agarre y elevación Cubo Negro      │
 │ 2. Colocación en bandeja Est. Negra   │ 6. Colocación en marca Cubo Negro     │
 │ 3. Agarre y elevación Estrella Naranja│ 7. Agarre y elevación Cubo Naranja    │
 │ 4. Colocación en bandeja Est. Naranja │ 8. Colocación en marca Cubo Naranja   │
 └───────────────────────────────────────┴───────────────────────────────────────┘
                                     │
                   9. Finalización Limpia del Episodio (1 pt)
```

### Definición Detallada de Criterios

#### 1. Agarre y Elevación (`*_agarre`)
- **Puntúa 1:** La pinza cierra firmemente sobre el objeto objetivo, este pierde completamente contacto con la mesa y se mantiene estable durante el inicio del transporte.
- **Puntúa 0:** El robot solo empuja o arrastra el objeto sin elevarlo, el objeto cae inmediatamente tras elevarse o queda aprisionado contra otro elemento sin sujeción real.

#### 2. Colocación de Estrellas (`estrella_*_destino`)
- **Puntúa 1:** La estrella se libera dentro de los límites del recipiente y permanece dentro al terminar el intento sin ser expulsada por acciones posteriores.
- **Puntúa 0:** La estrella queda sobre el borde, cae fuera o es desplazada fuera durante movimientos siguientes.

#### 3. Colocación de Cubos (`cubo_*_destino`)
- **Puntúa 1:** El cubo se libera sobre la zona delimitada por la marca con su base completamente contenida dentro del área.
- **Puntúa 0:** El cubo cae fuera, sobresale de los límites de la marca o es desplazado fuera posteriormente.

#### 4. Finalización Limpia (`finalizacion_limpia`)
- **Puntúa 1:** El robot concluye la secuencia completa dentro del tiempo límite sin quedar bloqueado cinemáticamente, sin bucles repetitivos de indecisión y sin requerir intervención humana.
- **Puntúa 0:** El robot se bloquea, oscila indefinidamente, colisiona violentamente contra el entorno o requiere parada de emergencia manual.

### Métricas Agregadas
- **Puntuación Normalizada:** $\text{Puntuación} = \frac{\text{Puntos Brutos}}{9} \in [0.0, 1.0]$.
- **Tasa de Éxito Completo:** $\text{Éxito} = 1 \iff \text{Puntos Brutos} = 9$ (100% de subtareas completadas limpiamente).
- **Taxonomía de Fallos:** Clasificación por modo (`modo_fallo`: `fallo_agarre`, `colision`, `bloqueo`, `desorden`, `otro`), fase de ejecución (`fase_fallo`) y objeto afectado (`objeto_fallo`).

---

## 3. Resultados y Conclusiones Arquitectónicas

### Tabla Maestra de Resultados (21 Configuraciones de Modelo)

| Variante de Modelo | N | Puntuación Media $\mu$ | Desv. $\sigma$ | Éxito Completo % | Agr. ★N | Agr. ★O | Agr. CN | Agr. CO | Col. ★N | Col. ★O | Col. CN | Col. CO |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| **`base` (SmolVLA)** | 15 | **0.830** | 0.234 | **60.0%** | 0.867 | 0.933 | 0.800 | 0.867 | 0.867 | 0.933 | 0.800 | 0.800 |
| `layercut_0` | 15 | 0.645 | 0.310 | 26.7% | 0.667 | 0.800 | 0.667 | 0.667 | 0.667 | 0.733 | 0.667 | 0.667 |
| `layercut_1` | 15 | 0.176 | 0.091 | 0.0% | 0.067 | 0.400 | 0.333 | 0.000 | 0.067 | 0.400 | 0.333 | 0.000 |
| `layercut_2` | 15 | 0.235 | 0.196 | 0.0% | 0.333 | 0.333 | 0.267 | 0.133 | 0.333 | 0.333 | 0.267 | 0.133 |
| `layercut_3` | 15 | 0.257 | 0.142 | 0.0% | 0.467 | 0.333 | 0.333 | 0.067 | 0.467 | 0.267 | 0.333 | 0.067 |
| `layercut_4` | 15 | 0.607 | 0.383 | 40.0% | 0.600 | 0.800 | 0.533 | 0.600 | 0.600 | 0.800 | 0.533 | 0.600 |
| `layercut_5` | 15 | 0.191 | 0.183 | 0.0% | 0.400 | 0.267 | 0.200 | 0.000 | 0.400 | 0.267 | 0.200 | 0.000 |
| **`layercut_6`** | 14 | **0.613** | 0.247 | 14.3% | 0.714 | 0.786 | 0.429 | 0.714 | 0.714 | 0.786 | 0.429 | 0.714 |
| **`layercut_7`** | 15 | **0.585** | 0.272 | 20.0% | 0.667 | 0.733 | 0.600 | 0.533 | 0.667 | 0.733 | 0.600 | 0.533 |
| **`layercut_8`** | 15 | **0.533** | 0.319 | 20.0% | 0.600 | 0.800 | 0.467 | 0.467 | 0.600 | 0.733 | 0.467 | 0.467 |
| `layercut_9` | 15 | 0.473 | 0.300 | 13.3% | 0.467 | 0.733 | 0.533 | 0.333 | 0.467 | 0.733 | 0.533 | 0.333 |
| `layercut_10` | 15 | 0.429 | 0.349 | 13.3% | 0.333 | 0.533 | 0.467 | 0.533 | 0.333 | 0.533 | 0.467 | 0.533 |
| `layercut_11` | 15 | 0.541 | 0.342 | 20.0% | 0.533 | 0.800 | 0.533 | 0.467 | 0.533 | 0.800 | 0.533 | 0.467 |
| `layercut_12` | 15 | 0.355 | 0.250 | 0.0% | 0.533 | 0.733 | 0.200 | 0.133 | 0.533 | 0.733 | 0.200 | 0.133 |
| `layercut_13` | 14 | 0.251 | 0.146 | 0.0% | 0.429 | 0.267 | 0.133 | 0.267 | 0.400 | 0.267 | 0.133 | 0.267 |
| `layercut_14` | 15 | 0.169 | 0.171 | 0.0% | 0.067 | 0.267 | 0.133 | 0.333 | 0.067 | 0.267 | 0.133 | 0.267 |
| `layercut_15` | 15 | 0.369 | 0.173 | 0.0% | 0.800 | 0.600 | 0.200 | 0.533 | 0.467 | 0.533 | 0.000 | 0.200 |
| `layercut_range_6_7` | 15 | 0.383 | 0.284 | 13.3% | 0.267 | 0.667 | 0.467 | 0.267 | 0.267 | 0.667 | 0.467 | 0.267 |
| `layercut_range_6_7_8` | 15 | 0.354 | 0.212 | 0.0% | 0.333 | 0.467 | 0.467 | 0.333 | 0.333 | 0.467 | 0.467 | 0.333 |
| `layercut_range_7_8_9` | 15 | 0.198 | 0.104 | 0.0% | 0.267 | 0.333 | 0.333 | 0.000 | 0.200 | 0.333 | 0.333 | 0.000 |
| `layercut_range_9_10_11` | 15 | **0.125** | 0.097 | 0.0% | 0.200 | 0.200 | 0.267 | 0.000 | 0.200 | 0.133 | 0.133 | 0.000 |

### Conclusiones Principales del Experimento

1. **Cuellos de Botella Estructurales:**
   - **Capas Iniciales (1, 2, 3, 5):** Desactivar las capas 1 a 3 degrada severamente el rendimiento ($\le 0.257$ de puntuación media, $0\%$ de éxito). Estas capas generan las representaciones intermedias motoras fundamentales.
   - **Capas Finales (13, 14):** La capa 14 registra la puntuación media individual más baja de toda la red ($0.169$). Estas capas decodifican los comandos directos de posición y velocidad articular.
2. **Meseta de Tolerancia Contigua (Capas 6–8):**
   - Las capas 6, 7 y 8 presentan la mayor conservación de capacidad funcional de toda la red intermedia ($\mu = 0.613, 0.585, 0.533$ con tasas de éxito de $14.3\%–20.0\%$).
   - Aunque `layercut_0` ($0.645$) y `layercut_4` ($0.607$) obtienen buenas puntuaciones, son puntos aislados con caídas drásticas en capas adyacentes.
3. **No Aditividad de la Tolerancia:**
   - La desactivación de múltiples capas consecutivas demuestra que la tolerancia no es lineal. Suprimir el bloque 6–8 completo (`range_6_7_8`) reduce la puntuación a $0.354$, mientras que desplazar la ventana hacia capas posteriores (`range_9_10_11`) causa un colapso total ($\mu = 0.125$, $0\%$ éxito).
4. **Decisión de Diseño para SmolVLA-D:**
   - La ventana tolerante contigua de las **capas 6, 7 y 8** fue seleccionada para la inyección de profundidad 3D (`depth_injection_layers=[6, 7, 8]`).
   - La integración se realiza mediante **adaptadores de cross-attention residuales inicializados a cero**, preservando íntegramente las competencias motoras preentrenadas mientras se introduce progresivamente la información geométrica 3D.

---

## 4. Recursos del Directorio

- [`resultados_ablacion.ipynb`](resultados_ablacion.ipynb): Notebook interactivo con pruebas estadísticas completas y gráficos.
- [`prepare_layercut_checkpoint.py`](prepare_layercut_checkpoint.py): Script para generar y exportar checkpoints layercut.
- [`verify_layercut.py`](verify_layercut.py): Script de verificación de configuraciones de ablación.
- [`evaluations/`](evaluations/): Registros CSV brutos de los 21 modelos evaluados.
- [`graphs/`](graphs/): Visualizaciones y scripts de generación de figuras.
- [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md): Versión en inglés de este informe técnico.
