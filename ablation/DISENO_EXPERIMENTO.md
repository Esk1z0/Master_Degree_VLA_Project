# Diseño del experimento de ablación (skip-block del Action Expert)

> Ver también: [`graphs/README.md`](graphs/README.md) para el análisis de resultados y gráficas, y
> [`../final_evaluations/DISENO_EVALUACION.md`](../final_evaluations/DISENO_EVALUACION.md) para el
> experimento de validación conductual que sigue a este.

## Objetivo

Determinar qué capas de las 16 del Action Expert de SmolVLA son "prescindibles" (el modelo tolera
desactivarlas) frente a cuáles son estructuralmente críticas, **antes** de decidir dónde inyectar
mecanismos nuevos (memoria temporal, profundidad). La lógica: un punto de inyección seguro debe caer en
una capa donde el modelo ya demuestre tolerancia a perturbaciones, no en una capa crítica para el control
motor básico.

El experimento tiene **tres fases**, de menor a mayor coste (de una sweep numérica offline en segundos a
ejecuciones físicas de minutos por episodio):

## Fase 0 — Barrido offline por divergencia de salida (`xai_ablation.py`)

- Carga `lerobot/smolvla_base` (pesos preentrenados de HuggingFace, sin fine-tuning específico del TFM) y
  genera un batch **sintético** (imágenes/estado/lenguaje aleatorios vía `generate_mock_batch`).
- Calcula la predicción de acciones "baseline" (sin ablación) y luego, para **cada ventana contigua posible**
  de capas del Action Expert — tamaño de ventana de 1 a 15, y cada posición de inicio válida para ese
  tamaño — parchea `get_model_layers` para que esas capas actúen como identidad (`None` → passthrough, sin
  atención ni MLP) y recalcula la predicción.
- Métrica: MSE y MAE entre la predicción ablacionada y la baseline. Cuanto mayor el MSE, más cambia la
  salida del modelo al quitar esas capas → proxy numérico de "cuán crítica" es esa combinación de capas.
- Resultado: 135 combinaciones evaluadas (ventanas 1–15), guardadas en
  `../results/layer_basic/ablation_results.json`, con dos gráficas de apoyo:
  `../results/layer_basic/mse_heatmap.png` (ventana × posición inicial) y `mse_lines.png` (evolución del MSE
  para ventanas de tamaño 1, 2 y 3).
- **Límite de esta fase**: usa el modelo base sin fine-tuning y una entrada sintética aleatoria, no una
  tarea real ni un checkpoint entrenado para la tarea del TFM. Sirve para tener una primera intuición barata
  de qué zonas de la red son más sensibles antes de gastar tiempo de robot real, pero no sustituye la
  validación física (fases 1 y 2).
- El plan combinatorio completo (las 120 combinaciones teóricas de ventana×posición que cubren esta fase y
  las siguientes) está enumerado en [`../docs/smolvla_layer_ablation_plan.md`](../docs/smolvla_layer_ablation_plan.md).
  Solo una fracción de ese plan (single-layer completo + 4 combinaciones de rango) llegó a ejecutarse
  físicamente — ver fases 1 y 2.

## Infraestructura física: `smolvla_layercut`

Las fases 1 y 2 sí usan un checkpoint entrenado para la tarea real
(`outputs/train/tfm_layer_ablation_expert_only_v3/checkpoints/last/pretrained_model`, entrenado sobre los
datasets `Esk1z0/tfm_layer_ablation_batch_1` y `_batch_2` combinados) y lo ejecutan en el robot SO-101
físico.

- `prepare_layercut_checkpoint.py` no reentrena nada: crea un directorio "wrapper" que reutiliza los mismos
  pesos (hardlink de `model.safetensors`, 0 bytes extra) pero cambia `type` a `smolvla_layercut` en
  `config.json` y añade `ablate_layer_indices` / `ablate_layer_range`.
- `SmolVLALayercutPolicy` (`../lerobot/src/lerobot/policies/smolvla_layercut/modeling_smolvla_layercut.py`)
  hace *monkey-patch* de `get_model_layers` para que las capas del Action Expert en los índices ablacionados
  se sustituyan por `None`. Una capa `None` actúa como **conexión de identidad pura** (el tensor pasa sin
  modificarse, sin atención ni MLP) — no se pone a cero, no se reentrena nada, es exactamente "saltarse el
  bloque" (*skip-block*, de ahí el nombre de la rúbrica). Solo se tocan capas del Action Expert; el VLM
  nunca se ablaciona.
- También parchea `forward_attn_layer` para evitar errores cuando, en un paso de denoising sin prefix
  (`inputs_embeds=[None, suffix]`), la capa ablacionada dejaría una lista vacía para `torch.cat`.
- `verify_layercut.py` es la comprobación de sanidad: confirma que exactamente los índices esperados quedan
  en `None` en `expert_layers` antes de lanzar una evaluación física, evitando ejecutar un experimento con la
  configuración equivocada.

## Fase 1 — Ablación física de **una sola capa** (`layercut_0` … `layercut_15`)

Cada una de las 16 capas del Action Expert se desactiva **individualmente** (`ablate_layer_indices=[i]`) y
se evalúa en el robot real, más un modelo `base` sin ninguna capa desactivada como referencia. Total: 17
configuraciones.

## Fase 2 — Ablación física de **varias capas a la vez** (`layercut_range_*`)

Se desactivan 2–3 capas consecutivas simultáneamente: `6-7`, `6-7-8`, `7-8-9`, `9-10-11`
(`ablate_layer_range=[start, end]`, intervalo cerrado). Estas cuatro combinaciones no son arbitrarias:
cubren la ventana que la fase 1 señaló como más tolerante (6-8) y su frontera inmediata (9-11), para
comprobar si esa tolerancia se mantiene, se degrada gradualmente o colapsa al combinar capas — pregunta que
la fase 1 (una capa cada vez) no puede responder por sí sola. Ver [`graphs/04_degradacion_no_lineal.png`](graphs/04_degradacion_no_lineal.png)
para el resultado (colapsa de forma no lineal, más severamente cuanto más se acerca a 9-11).

## Diseño de las escenas físicas

- `combinations.py` genera un **pool** de 240 escenas balanceadas (`../data/layouts/so101_layouts_001_060.csv`
  … `_181_240.csv`) más un lote adicional de 60 (`../data/layouts/so101_layouts_skip_block.csv`): posiciones
  de 4 objetos (`starBlackPos`, `starOrangePos`, `cubeBlackPos`, `cubeOrangePos`) sobre una rejilla 3×3
  (`A1`…`C3`) y una rotación (`low`/`mid`/`high`), elegidas con un muestreo *greedy* (500 candidatos por
  episodio) que penaliza la repetición de zona por objeto, la repetición global de zona y el agrupamiento
  espacial excesivo de los 4 objetos — para que el conjunto de escenas cubra el tablero de forma uniforme en
  vez de agruparse en unas pocas zonas.
- Sin embargo, las evaluaciones **realmente ejecutadas** (`evaluations/eval_checklist_*.csv`) no usan ese
  pool aleatorio: usan **3 escenas fijas, elegidas a mano** (`layout_id=manual`, `fuente=manual`), una por
  cada condición (`training`, `no_visto`, `stress`), repetida **5 veces (intentos)** cada una, para un total
  de **15 ejecuciones por modelo** (14 en `layercut_6` y `layercut_13` por datos faltantes en 1 intento). Con
  ~20 configuraciones de modelo a evaluar físicamente, fijar las escenas por condición convierte el layout
  en una variable controlada en vez de una fuente extra de varianza entre modelos, a costa de menor
  cobertura de escenas que el pool generado por `combinations.py`.
- Las 3 condiciones (`tipo_configuracion`) representan niveles de dificultad/familiaridad distintos, aunque
  el detalle fino de qué distingue cada una debe consultarse en las notas de captura del proyecto; a nivel
  de diseño del CSV son simplemente tres grupos independientes de 5 repeticiones cada uno por modelo.

## Rúbrica de puntuación

Definida en [`EVALUATION_RUBRIC.md`](EVALUATION_RUBRIC.md): 9 criterios binarios (agarre y colocación de 2
estrellas + 2 cubos, más finalización limpia) → `puntuacion_bruta` (0-9) → `puntuacion_normalizada` (0-1) →
`exito_completo` (1 solo si `puntuacion_bruta == 9`). Se calculan automáticamente con
[`evaluations/calculate_evaluation_scores.py`](evaluations/calculate_evaluation_scores.py), que también
imprime el resumen por modelo y el análisis de modos de fallo (`modo_fallo`, `fase_fallo`, `objeto_fallo`)
volcado en [`evaluations/results.txt`](evaluations/results.txt). Regla explícita de la rúbrica: los
criterios de puntuación no se pueden cambiar a mitad de proceso, para que las comparaciones entre las ~20
configuraciones sigan siendo válidas. El límite de tiempo máximo por ejecución quedó marcado como pendiente
de definir (`[PENDIENTE DE DEFINIR]`) en la rúbrica.

## De los resultados a la decisión de diseño

Resumen (detalle completo y gráficas en [`graphs/README.md`](graphs/README.md)):

1. Fase 1 identifica capas claramente críticas (1, 2, 3, 5, 13, 14 — puntuación media <0.28, 0% éxito) frente
   a capas relativamente tolerantes, entre ellas 6, 7 y 8 (puntuación 0.53–0.61, 14–20% de éxito, muy por
   encima de la mayoría del resto).
2. Fase 2 muestra que esa tolerancia **no es aditiva**: combinar 2-3 de esas capas relativamente seguras
   colapsa el rendimiento muy por debajo de lo que la fase 1 haría esperar, y ese colapso es **peor** cuanto
   más se aleja la ventana de 6-8 hacia 9-11.
3. Conclusión de diseño usada en `smolvla_d` / `smolvla_md`: las capas 6, 7 y 8 son el mejor punto disponible
   del Action Expert para **añadir** una señal nueva (profundidad), pero solo mediante una intervención no
   destructiva — suma residual con adaptador inicializado a cero (`zero_proj`), nunca poda, reemplazo o
   fine-tuning agresivo de esas capas. Ver
   [`../docs/models/smolvla_d/architecture.md`](../docs/models/smolvla_d/architecture.md) §7 para el
   mecanismo exacto de inyección que resultó de esta decisión.

## Relación con la evaluación final

Este experimento responde "¿dónde del Action Expert puedo intervenir sin romper el modelo?" — es la
evidencia que sostiene la elección arquitectónica `depth_injection_layers=[6,7,8]`. No dice nada sobre si la
profundidad o la memoria realmente ayudan a resolver la tarea; solo dice dónde es seguro tocar la red. Esa
segunda pregunta ("¿la memoria y la profundidad que añadimos realmente cambian el comportamiento del robot
de forma útil?") es el objeto de la evaluación final, documentada en
[`../final_evaluations/DISENO_EVALUACION.md`](../final_evaluations/DISENO_EVALUACION.md).
