# Diseño de la evaluación final (validación conductual de M y D)

> Ver también: [`../ablation/DISENO_EXPERIMENTO.md`](../ablation/DISENO_EXPERIMENTO.md) para el experimento
> de ablación de capas que precede y motiva arquitectónicamente a este, y [`README_evaluacion.md`](README_evaluacion.md)
> para el flujo operativo de relleno/cálculo de resultados.

## Objetivo

Mientras que el experimento de ablación (`ablation/`) es una herramienta de **diseño arquitectónico**
(dónde es seguro intervenir en el Action Expert), la evaluación final es una prueba de **validación
conductual**: comprobar en el robot real si las capacidades que SmolVLA-M (memoria temporal) y SmolVLA-D
(profundidad) están diseñadas para aportar se manifiestan realmente en la tarea, y no solo sobre el papel.
Compara los modelos finales (vanilla, M, D, MD, y presumiblemente el `base` de SmolVLA) tal como quedan tras
el entrenamiento, sin tocar su arquitectura.

## Los 4 escenarios y qué capacidad prueba cada uno

El diseño de escenario es la pieza central de este experimento: cada uno de los 4 tipos de prueba está
construido para estresar específicamente una de las capacidades nuevas, de forma aislada primero y combinada
después:

| Escenario | Episodios | Qué prueba | Relación con el modelo |
|---|---|---|---|
| `destreza_general` | 5 (`G-01`…`G-05`) | Manipulación básica (agarrar y colocar 2 estrellas + 2 cubos) sin ningún estresor añadido | Línea base común a los 4 modelos — sin esto, cualquier diferencia en memoria/profundidad no sería interpretable |
| `autooclusion` | 5 (`O-01`…`O-05`) | El brazo, al manipular un objeto, **tapa visualmente** otro objeto pendiente (`objeto_ocluido`) durante parte de la ejecución; se mide si el robot recuerda y termina ese objeto tras la oclusión | Diseñado para SmolVLA-M: sin memoria temporal (K frames de historia), un modelo puramente reactivo de un solo frame no tiene forma de "recordar" un objeto que ya no ve |
| `fotografia` | 5 (`P-01`…`P-05`) | Se coloca una **fotografía** de un objeto (`foto_objeto`) junto al objeto real; se mide si el robot va a por el objeto real y evita intentar agarrar la foto (una imagen 2D no tiene profundidad real) | Diseñado para SmolVLA-D: un modelo sin señal de profundidad solo tiene RGB, que no distingue intrínsecamente una foto plana de un objeto 3D real |
| `combinado` | 3 (`C-01`…`C-03`) | Ambos estresores a la vez (oclusión de un objeto + distractor fotográfico de otro) | Diseñado para SmolVLA-MD: prueba si memoria y profundidad siguen funcionando cuando compiten por la atención del modelo en la misma ejecución |

Total: **18 episodios por modelo** (5+5+5+3), consistente con "los 18 intentos" mencionados en
[`README_evaluacion.md`](README_evaluacion.md).

## Rúbrica extendida y validez del registro

A diferencia de la rúbrica de ablación (9 criterios homogéneos para todos los episodios), aquí la
puntuación se compone de **tres bloques**, activos según el escenario
([`calcular_resultados_evaluacion.py`](calcular_resultados_evaluacion.py)):

- **Destreza** (siempre, máx. 9 puntos): mismos 8 criterios de agarre/colocación + finalización limpia que en
  ablación.
- **Memoria** (solo `autooclusion` y `combinado`, máx. 2 puntos): `recuerda_objeto_ocluido` (el robot
  redirige su atención al objeto pendiente tras la oclusión) + `completa_objeto_ocluido` (termina de
  manipularlo correctamente).
- **3D / Fotografía** (solo `fotografia` y `combinado`, máx. 2 puntos): `selecciona_real_antes_foto` (la
  primera intención de agarre es sobre el objeto físico) + `evita_intento_agarre_foto` (nunca cierra la
  pinza sobre la fotografía). También se cuentan (sin puntuar) `aproximaciones_foto` e
  `intentos_agarre_foto` como métricas de frecuencia del error.

La puntuación máxima por episodio es, por tanto, variable: 9 en `destreza_general`, 11 en `autooclusion` y
`fotografia`, 13 en `combinado`. `puntuacion_normalizada = puntuacion_total / puntuacion_max` corrige esto
para que los escenarios sean comparables entre sí.

**Validez del registro** (`registro_valido`) es un control de calidad experimental adicional que no existía
en el estudio de ablación: un episodio de `autooclusion`/`combinado` solo cuenta para las métricas agregadas
si, además de `setup_validado=1` (las posiciones en la mesa coinciden con lo planeado), se confirma
`objeto_visible_inicio=1` (el objeto a ocluir era visible antes del movimiento) y `oclusion_verificada=1`
(el brazo realmente llegó a taparlo). Esto evita que un episodio donde la oclusión no llegó a ocurrir de
verdad se cuente como si el modelo hubiera "fallado" o "acertado" en memoria. `exito_completo` es aún más
estricto que en ablación: exige registro válido + puntuación máxima exacta + sin `timeout` + sin
`intervencion_emergencia` (parada de seguridad manual).

## Diseño de las 18 escenas

A diferencia del pool balanceado y generado algorítmicamente de la fase de ablación (`ablation/combinations.py`),
las 18 escenas de la evaluación final están **curadas a mano**, una por una, con un propósito explícito
anotado en `notas_setup` de cada fila de la plantilla — por ejemplo, `O-01` fija la trayectoria del brazo
por la columna 1 explícitamente para validar que al actuar sobre `C1` se tape `A1`; `P-01`/`P-02` colocan el
objeto real y su fotografía en la misma fila en posiciones intercambiadas para descartar sesgos de lateralidad;
`C-02`/`C-03` combinan oclusión y fotografía con notas explícitas de ajuste de posición para evitar
solapamientos accidentales entre marcadores. Esto tiene sentido dado que cada escena aquí necesita cumplir una
condición geométrica/causal específica (que el brazo realmente ocluya tal objeto, que la foto y el original
no se confundan de posición) que un muestreo aleatorio balanceado no puede garantizar.

## Estado actual

`evaluations/` solo contiene la plantilla (`plantilla_evaluacion_modelo.csv`), con las 18 filas de diseño de
escena ya definidas pero **todas las columnas de resultado vacías** (puntuaciones, tiempos, modos de fallo).
Según [`README_evaluacion.md`](README_evaluacion.md), el flujo previsto es copiar esa plantilla a un
`evaluacion_<modelo>.csv` por cada modelo a evaluar (por ejemplo `evaluacion_smolvla_vanilla.csv`,
`evaluacion_smolvla_m.csv`, `evaluacion_smolvla_d.csv`, `evaluacion_smolvla_md.csv`), rellenarla durante las
ejecuciones físicas, y procesar todos los archivos a la vez con
`calcular_resultados_evaluacion.py evaluacion_*.csv` para obtener `resumen_global_evaluacion.csv`
comparando los modelos. **Esta evaluación todavía no se ha ejecutado.**

## Relación con el experimento de ablación

La ablación (`ablation/`) responde "¿dónde del Action Expert puedo intervenir sin romper el modelo?" — es la
evidencia que sostiene la elección arquitectónica `depth_injection_layers=[6,7,8]` usada en SmolVLA-D/MD.
Esta evaluación final responde una pregunta distinta y posterior: "¿la memoria y la profundidad que
añadimos realmente cambian el comportamiento del robot de forma útil?". En otras palabras: la ablación
justifica el **cómo/dónde** de SmolVLA-D/MD a nivel de arquitectura; esta evaluación debe justificar el
**para qué** a nivel de comportamiento observado en el robot. Ver
[`../ablation/DISENO_EXPERIMENTO.md`](../ablation/DISENO_EXPERIMENTO.md) para el detalle de esa primera
fase, ya completada.
