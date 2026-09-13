# Evaluación Final en Hardware Robótico: Informe Técnico

Este documento detalla el diseño experimental, los protocolos de evaluación, los registros brutos de ejecución y los resultados empíricos de la evaluación final en hardware físico comparando las cuatro variantes del modelo SmolVLA sobre el brazo robótico **SO-101**.

---

## 1. Diseño y Realización del Experimento

### Objetivo y Alcance
La evaluación final es una prueba de **validación conductual de extremo a extremo** diseñada para comprobar si las capacidades multimodales introducidas en SmolVLA-M (memoria visual temporal) y SmolVLA-D (inyección de profundidad estéreo 3D) se traducen en mejoras tangibles de manipulación, precisión espacial y robustez ante distractores en hardware robótico real de bajo coste.

```
Variantes de Modelo Evaluadas (Todos con Checkpoints v2):
┌────────────────────┬─────────────────────────────────────────────────┬──────────────────────┐
│ Variante de Modelo │ Arquitectura y Modalidades Sensoriales          │ Estado del Checkpoint│
├────────────────────┼─────────────────────────────────────────────────┼──────────────────────┤
│ 1. SmolVLA-Vanilla │ Baseline preentrenado RGB monocular/estéreo     │ 22.500 p. (100,0%)   │
│ 2. SmolVLA-M       │ Memoria Visual Temporal Causal (K=6 frames)     │ 24.000 p. (~53,3%)   │
│ 3. SmolVLA-D       │ Profundidad 3D en capas 6-8 (Cross-Attention)   │ 22.500 p. (100,0%)   │
│ 4. SmolVLA-MD      │ Multimodal Combinado (Profundidad + Memoria)    │ 24.000 p. (~53,3%)   │
└────────────────────┴─────────────────────────────────────────────────┴──────────────────────┘
```

### Inferencia en Vivo y Pipeline de Profundidad 3D
- **Plataforma de Hardware:** Brazo robótico de 6-DoF **SO-101** controlado mediante LeRobot. Disposición de doble cámara: cámara cenital fija de contexto (`camera1`) y cámara estéreo dual montada en la muñeca (`camera2`, 2560x800).
- **Inyección de Nube de Puntos en Tiempo Real:** Ejecutado mediante `lerobot-record` con el parámetro `--depth_calib_path camera/calib_stereo.npz`. En cada ciclo del bucle de control se calculan las disparidades mediante OpenCV SGBM+WLS, proyectando la nube de puntos geométrica 3D hacia los adaptadores residuales de la política.

### Diseño de Escenarios de Evaluación (12 Episodios Físicos por Modelo)
Para aislar rigurosamente los efectos de la memoria y la profundidad sin varianza espuria, los 4 modelos se evaluaron en **12 intentos físicos estandarizados** (6 sin distractor fotográfico y 6 con distractor fotográfico), ejecutados en bloque:

```
Disposición Física en la Mesa (Rejilla A1–C3):
┌──────────────┬──────────────┬──────────────┐
│      A1      │      A2      │      A3      │
│   [Vacío]    │Est. Naranja  │  Cubo Negro  │
├──────────────┼──────────────┼──────────────┤
│      B1      │      B2      │      B3      │
│ [Recipiente] │Est. Negra    │ *Cubo Naranja*│ ◄ Objeto Ocluido Objetivo
├──────────────┼──────────────┼──────────────┤
│      C1      │      C2      │      C3      │
│ [Foto Dist.] │   [Vacío]    │ [Zona Marca] │
└──────────────┴──────────────┴──────────────┘
```

1. **`combinado_sin_foto` (Episodios `C-01` a `C-06`):**
   - Objetivo: Aislar la memoria temporal.
   - Configuración: El cubo naranja real en posición `B3` queda ocluido dinámicamente por el propio brazo del robot al recoger las figuras centrales (`B2`/`A2`). Sin fotografía presente en la mesa.
2. **`combinado_con_foto` (Episodios `C-07` a `C-12`):**
   - Objetivo: Prueba de estrés máximo (Profundidad + Memoria).
   - Configuración: Misma disposición base, añadiendo una fotografía impresa en 2D de alta fidelidad del cubo naranja en la casilla `C1`.

---

## 2. Protocolo y Rúbrica Extendida de Evaluación

Dado que los episodios con fotografía evalúan capacidades perceptivas adicionales, la rúbrica de puntuación se estructura en bloques modulares:

### Estructura de Puntuación (Máx. 11 pts sin Foto / Máx. 13 pts con Foto)

```
                            Rúbrica de Evaluación Modular
 ┌──────────────────────────────────────────┬──────────────────────────────────────────┐
 │       Bloque B: Destreza (9 pts)         │        Bloque C: Memoria (2 pts)         │
 ├──────────────────────────────────────────┼──────────────────────────────────────────┤
 │ • 4 Criterios de Agarre (Estrellas/Cubos)│ • recuerda_objeto_ocluido (1/0)          │
 │ • 4 Criterios de Colocación (Bandeja/Zon)│ • completa_objeto_ocluido (1/0)          │
 │ • finalizacion_limpia (1/0)              │                                          │
 └──────────────────────────────────────────┴──────────────────────────────────────────┘
                                     │
 ┌──────────────────────────────────────────┴──────────────────────────────────────────┐
 │    Bloque D: 3D / Distractor Fotográfico (2 pts, solo en episodios C-07 a C-12)     │
 ├─────────────────────────────────────────────────────────────────────────────────────┤
 │ • selecciona_real_antes_foto (1/0)   │ • evita_intento_agarre_foto (1/0)            │
 │ • intentos_agarre_foto (escala 0/1/2)│                                              │
 └─────────────────────────────────────────────────────────────────────────────────────┘
```

### Definición Detallada de Criterios y Controles de Calidad

#### 1. Controles de Calidad Experimental (Bloque A)
- `setup_validado` (1/0): Certifica que la colocación física de los objetos coincide con la plantilla.
- `objeto_visible_inicio` (1/0): El objeto a ocluir era claramente visible antes del inicio del movimiento.
- `oclusion_verificada` (1/0): El brazo del robot llegó a tapar físicamente el objeto durante la ejecución.
- `registro_valido` (1/0): Indicador de validez global ($1$ solo si setup, visibilidad y oclusión fueron verificados).

#### 2. Destreza y Ejecución de Tarea (Bloque B — 9 pts)
- Criterios de agarre y colocación de los 4 objetos según la rúbrica binaria estandarizada.
- `finalizacion_limpia` (1/0): Ejecución sin asistencia humana, sin bloqueos cinemáticos ni colisiones violentas.

#### 3. Memoria Temporal (Bloque C — 2 pts, evaluado en los 12 episodios)
- `recuerda_objeto_ocluido` (1/0): Tras producirse la oclusión, el robot reorienta suavemente su atención y trayectoria hacia el objeto pendiente.
- `completa_objeto_ocluido` (1/0): El robot finaliza con éxito la manipulación y colocación del objeto ocluido.

#### 4. Discriminación Geométrica 3D (Bloque D — 2 pts, evaluado en `C-07` a `C-12`)
- `selecciona_real_antes_foto` (1/0): La primera intención de agarre se dirige hacia el objeto físico real en lugar de la fotografía 2D.
- `evita_intento_agarre_foto` (1/0): La pinza nunca llega a cerrarse sobre la fotografía ni a colisionar con ella en todo el episodio.
- `intentos_agarre_foto` (Escala 0/1/2): Registro categórico ($0$ = solo aproximación visual sin cierre, $1$ = un intento claro de agarre en la foto, $2$ = múltiples intentos de agarre en la foto).

#### 5. Indicadores de Fallo y Seguridad (Bloque E)
- `timeout` (1/0): El episodio superó el tiempo máximo establecido ($120$ s) o entró en bucle infinito.
- `intervencion_emergencia` (1/0): Parada manual de seguridad por parte del operador ante movimientos erráticos o peligrosos.

### Normalización de Puntuación
$$\text{Puntuación}_{\text{sin\_foto}} = \frac{\text{Puntos Brutos}}{11} \in [0.0, 1.0], \quad \text{Puntuación}_{\text{con\_foto}} = \frac{\text{Puntos Brutos}}{13} \in [0.0, 1.0]$$
- **Éxito Completo Estricto:** $\text{Éxito} = 1 \iff \text{registro\_valido} = 1 \land \text{Puntuación} = 1.0 \land \text{timeout} = 0 \land \text{intervencion} = 0$.

---

## 3. Resultados y Análisis Comparativo de Modelos

### Tabla Resumen de Rendimiento

| Variante de Modelo | Puntuación Global Normalizada | Puntuación Sin Foto | Puntuación Con Foto | Tasa de Éxito Completo | Timeouts | Paradas de Emergencia |
|---|---|---|---|---|---|---|
| **SmolVLA-Vanilla** | 0.356 | 0.545 | 0.167 | 2 / 12 (16,7%) | 10 / 12 | 0 / 12 |
| **SmolVLA-M** | 0.398 | 0.591 | 0.205 | 1 / 12 (8,3%) | 11 / 12 | 0 / 12 |
| **SmolVLA-D** | **0.490** | **0.864** | 0.115 | **3 / 12 (25,0%)** | 9 / 12 | 0 / 12 |
| **SmolVLA-MD** | 0.057 | 0.076 | 0.038 | 0 / 12 (0,0%) | 2 / 12 | **10 / 12** |

---

### Perfiles Conductuales por Modelo

```
Distribución del Rendimiento por Condición:
  1.0 ┌─────────────────────────────────────────────────────────────┐
      │                                    [0.864]                  │
  0.8 │                                    ██████                   │
      │                  [0.545]  [0.591]  ██████                   │
  0.6 │                  ██████   ██████   ██████                   │
      │                  ██████   ██████   ██████                   │
  0.4 │                  ██████   ██████   ██████                   │
      │                  ██████   ██████   ██████                   │
  0.2 │ [0.167] [0.205]  ██████   ██████   ██████  [0.115]  [0.076] │
      │  ░░░░░   ░░░░░   ██████   ██████   ██████   ░░░░░    ██████ │
  0.0 └─────────────────────────────────────────────────────────────┘
         Vanilla           SmolVLA-M        SmolVLA-D      SmolVLA-MD
        (Con / Sin Foto) (Con / Sin Foto) (Con / Sin Foto)(Con / Sin Foto)
```

1. **SmolVLA-D (Inyección de Profundidad Estéreo — Mejor Rendimiento Global):**
   - Logra un récord de **0.864** en entornos limpios (**+58,5% de mejora relativa frente al baseline Vanilla**).
   - Registró 3 ejecuciones perfectas de 6 en `combinado_sin_foto`.
   - Mayor tasa de recuerdo y compleción de objetos ocluidos ($58,3\%$ de recuerdo, $50,0\%$ de compleción).
   - Comportamiento marcadamente bimodal: excelente en manipulación 3D limpia, pero vulnerable a la ambigüedad plana de la foto 2D (timeouts en los 6 intentos con foto).
2. **SmolVLA-M (Memoria Visual Temporal — Máxima Estabilidad):**
   - El modelo más regular: nunca obtuvo cero puntos en ningún episodio; inició con éxito la manipulación de la estrella negra en el 100% de los intentos.
   - El recuerdo de oclusión fue elevado sin foto ($83,3\%$), pero colapsó al competir la foto por la atención visual ($0,0\%$).
3. **SmolVLA-Vanilla (Modelo Baseline):**
   - Rendimiento intermedio ($0.356$ global, $0.545$ sin foto, 2 éxitos completos).
   - Fallo absoluto en discriminación de la foto (`selecciona_real_antes_foto = 0/6`).
4. **SmolVLA-MD (Multimodal Combinado — Colapso por Inestabilidad):**
   - Fuerte degradación conductual ($0.057$ global, $0$ éxitos, 10 de 12 episodios terminados por intervención de emergencia).
   - Manifestó oscilaciones motoras de alta frecuencia ("temblores" y bloqueos cinemáticos).
   - Los análisis sugieren que la inyección de profundidad por cross-attention y los estados ocultos de la memoria recurrente causaron interferencias de gradiente durante el entrenamiento incompleto.

### Conclusiones Transversales de la Investigación
- **Efecto Suelo del Distractor Fotográfico 2D:** Todos los modelos fallaron en la discriminación inicial (`selecciona_real_antes_foto = 0/6`). Las fotografías planas actúan como atractores visuales muy potentes para los backbones VLA estándar.
- **Aporte Decisivo de la Profundidad 3D:** La inyección de nubes de puntos 3D proporciona una ganancia sustancial de precisión para tareas de contacto y recuperación de oclusiones en ausencia de distractores adversarios 2D.
- **Interferencia Memoria-Distractor:** Los distractores visuales intensos interfieren directamente sobre la trayectoria de estados ocultos de los módulos de memoria temporal.

---

## 4. Recursos del Directorio

- [`calcular_resultados_evaluacion.py`](calcular_resultados_evaluacion.py): Script de cálculo de métricas y generación de resúmenes agregados desde los CSV.
- [`analisis_significancia.py`](analisis_significancia.py): Script para pruebas de significancia estadística entre modelos.
- [`evaluations/`](evaluations/): Registros CSV brutos y resúmenes de cada variante evaluada.
- [`graphs/`](graphs/): Gráficos comparativos de rendimiento y scripts de generación.
- [`EXPERIMENT_REPORT.md`](EXPERIMENT_REPORT.md): Versión en inglés de este informe técnico.
