# Gráficos del estudio de ablación (skip-block) — Action Expert de SmolVLA

Generados por [`generar_graficos_ablacion.py`](generar_graficos_ablacion.py) a partir de los CSV en
`../evaluations/`. Resumen del experimento: se evalúa el modelo `base` (SmolVLA sin modificar) y 20 variantes
en las que se **desactiva (bypass/skip) una o varias de las 16 capas del Action Expert**, sobre la misma tarea
robótica (colocar estrellas en el recipiente y cubos en la marca), con 15 ejecuciones por modelo repartidas en
3 condiciones (`training`, `no_visto`, `stress`). Ver `../EVALUATION_RUBRIC.md` para el detalle de puntuación.

**Por qué importa para el TFM**: `depth_injection_layers = [6, 7, 8]` en `configuration_smolvla_d.py` /
`configuration_smolvla_md.py` no es un valor arbitrario — estos gráficos son la evidencia empírica que motivó
esa elección. Ver también `../../wip/smolvla_variants_especificaciones.md` §4 para el mecanismo de inyección.

## Las dos fases del estudio (no confundir)

| Fase | Modelos | Qué se hace |
|---|---|---|
| **1. Capa única** | `layercut_0` … `layercut_15` | Se desactiva **una sola** de las 16 capas del Action Expert por ejecución. Es un barrido completo capa a capa para mapear qué tan crítica es cada una. |
| **2. Rango / varias capas** | `layercut_range_6_7`, `layercut_range_6_7_8`, `layercut_range_7_8_9`, `layercut_range_9_10_11` | Se desactivan **2 o 3 capas consecutivas simultáneamente**, para ver si el daño de varias capas "poco críticas" sigue siendo pequeño cuando se combinan. |

> Nota de datos: el CSV `eval_checklist_layercut_range_7_8_9.csv` tiene un typo en la columna `modelo`
> (`lyercut_range_7_8_9`, sin la primera "a"). El script lo normaliza automáticamente a
> `layercut_range_7_8_9` antes de graficar.

## Figuras

### 01 — `01_ablacion_capa_unica_puntuacion.png`
Puntuación normalizada media (±desviación estándar) al desactivar cada una de las 16 capas, una por una,
frente al modelo `base` (línea discontinua azul, 0.83). Colorea en **verde** las capas 6/7/8 (elegidas para
la inyección de profundidad) y en **rojo** las capas "críticas" (puntuación media < 0.28 al desactivarlas
solas: capas 1, 2, 3, 5, 13, 14). El resto queda en gris.

### 02 — `02_ablacion_capa_unica_exito.png`
Lo mismo que 01 pero con la métrica binaria estricta (tasa de éxito completo, sin crédito parcial). Usa la
**misma clasificación de color por capa que en 01** (basada siempre en la puntuación, no en el éxito) para
que ambos gráficos sean directamente comparables capa a capa.

**Lectura conjunta de 01+02**: las capas 1, 2, 3, 5, 13 y 14 son las más frágiles (puntuación ≤0.26, 0%
de éxito completo) — tocarlas destruye la política. Las capas 6, 7 y 8 están entre las que mejor toleran
ser desactivadas individualmente (puntuación 0.53–0.61, tasa de éxito 14–20%, muy por encima de la mayoría
de las demás capas intermedias/tardías).

### 03 — `03_individual_vs_combinado.png`
Panel izquierdo: capas 6 a 11 desactivadas **una a la vez**. Panel derecho: las mismas capas desactivadas
**en combinaciones de 2-3 a la vez**. Mismo eje Y (0–1) para comparar directamente.

**Conclusión visual**: capas 6/7/8 individualmente rondan 0.5–0.6 de puntuación; en cuanto se combinan 2 o
más (6+7, 6+7+8, 7+8+9, 9+10+11) la puntuación cae a 0.12–0.38 y la tasa de éxito completo se va a **0% en
3 de las 4 combinaciones**. El daño no es aditivo, es abrupto.

### 04 — `04_degradacion_no_lineal.png`
Para cada combinación de capas, compara el valor **"esperado"** (media simple de las puntuaciones/tasas de
éxito individuales de esas capas, como si el daño fuese independiente) contra el valor **observado** al
desactivarlas juntas de verdad. La brecha esperado→observado en puntuación **crece** según la ventana se
aleja de 6-8 hacia 9-11: 0.22 (6+7) → 0.22 (6+7+8) → 0.33 (7+8+9) → 0.36 (9+10+11).

**Esta es la pieza central del argumento de diseño**: el Action Expert tiene redundancia distribuida en
capas 6-11 que se agota rápido en cuanto se tocan varias a la vez, y esa fragilidad combinatoria es *peor*
cuanto más se avanza hacia las capas 9-11. Por eso el mecanismo de profundidad de D/MD **nunca elimina ni
reemplaza** capas 6/7/8 — solo les suma un delta pequeño con inicialización en cero (`zero_proj`), que en
`t=0` es un no-op exacto y converge gradualmente durante el entrenamiento (ver
`smolvla_variants_especificaciones.md` §4.3).

### 05 — `05_curva_degradacion_capas.png`
Las 16 capas en orden (0 a 15), como curva continua, con la ventana 6-8 sombreada. Da la vista de conjunto:
un "mapa de fragilidad" del Action Expert. Se aprecian dos valles muy profundos (capas 1-3 y 13-14) que
rodean una meseta relativamente más tolerante entre las capas 6 y 11, dentro de la cual 6-8 es la
sub-ventana con mejor comportamiento.

### 06 — `06_heatmap_config_capas_clave.png`
Puntuación media por tipo de configuración (`training` / `no_visto` / `stress`) para los modelos clave:
`base`, capas 6/7/8 individuales, y las 4 combinaciones de rango. Muestra que la degradación de las
combinaciones no es solo peor en promedio, sino **consistentemente peor en las tres condiciones**,
incluyendo generalización (`no_visto`) y robustez (`stress`) — no es un artefacto de una sola condición.

### 07 — `07_modos_fallo_capas_clave.png`
Modo de fallo dominante (`modo_fallo`) entre las ejecuciones fallidas, para los mismos modelos clave que en
06. En `base`/capas individuales el fallo se reparte entre `bloqueo`, `timeout`, `sin_aproximacion`, etc.
En las combinaciones de rango el fallo se concentra cada vez más en `agarre_fallido` y `bloqueo` puros
(las combinaciones 7+8+9 y 9+10+11 fallan al 100% por bloqueo) — indicio de que la política pierde
capacidad de control motor básico, no solo precisión fina.

## Resumen para la memoria del TFM

1. El barrido de capa única identifica qué capas del Action Expert son intercambiables/redundantes
   (6, 7, 8, y en menor medida 0, 4, 11) frente a las que son estructuralmente críticas (1, 2, 3, 5, 13, 14).
2. El barrido de rango demuestra que esa redundancia es limitada y no se puede explotar sumando ablaciones
   individuales: combinar capas "seguras" destruye la política igualmente.
3. Conclusión de diseño: capas 6-8 son el mejor punto disponible para **añadir** una señal nueva (profundidad)
   al Action Expert, pero solo mediante una intervención no destructiva (suma residual con compuerta
   zero-init), nunca mediante poda, reemplazo o fine-tuning agresivo de esas capas.
