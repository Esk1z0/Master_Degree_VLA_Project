# Rúbrica de Evaluación — Skip-Block Analysis de SmolVLA

**Tarea evaluada:** Colocar las estrellas en el recipiente y los cubos sobre la marca delimitada.
**Tiempo máximo por ejecución:** `[PENDIENTE DE DEFINIR]`
**Puntuación máxima:** 9 puntos (9 criterios binarios, cada uno vale 1 punto).
**Puntuación normalizada:** puntuación bruta / 9 ∈ [0.0, 1.0].

---

## 1. Criterios de puntuación (9 puntos binarios)

Cada criterio se puntúa con `1` (cumplido) o `0` (no cumplido). No hay puntuación parcial.

| # | Columna CSV | Descripción |
|---|---|---|
| 1 | `estrella_negra_agarre` | Agarre y elevación de la estrella negra |
| 2 | `estrella_negra_destino` | Colocación de la estrella negra en el recipiente |
| 3 | `estrella_naranja_agarre` | Agarre y elevación de la estrella naranja |
| 4 | `estrella_naranja_destino` | Colocación de la estrella naranja en el recipiente |
| 5 | `cubo_negro_agarre` | Agarre y elevación del cubo negro |
| 6 | `cubo_negro_destino` | Colocación del cubo negro sobre la marca |
| 7 | `cubo_naranja_agarre` | Agarre y elevación del cubo naranja |
| 8 | `cubo_naranja_destino` | Colocación del cubo naranja sobre la marca |
| 9 | `finalizacion_limpia` | Finalización limpia del episodio |

---

## 2. Definición de cada criterio

### 2.1 Agarre y elevación (`*_agarre`)

Se puntúa `1` cuando:
- La pinza sujeta el objeto firmemente.
- El objeto pierde completamente el contacto con la mesa.
- El objeto permanece sujeto durante el inicio del transporte.

Se puntúa `0` cuando:
- El robot únicamente empuja o arrastra el objeto sin elevarlo.
- El objeto se eleva momentáneamente y cae de inmediato sin desplazamiento real.
- El objeto queda atrapado contra otro elemento sin estar realmente sujeto por la pinza.

> **Nota:** Un agarre puede puntuarse como `1` aunque el objeto caiga posteriormente durante el transporte. En ese caso se concede el punto de agarre (`*_agarre = 1`) pero no el de colocación (`*_destino = 0`).

### 2.2 Colocación de estrellas (`estrella_*_destino`)

Se puntúa `1` cuando:
- La estrella se libera dentro del recipiente.
- Permanece dentro del recipiente al finalizar la ejecución.
- No es extraída ni desplazada fuera durante acciones posteriores del robot.

Se puntúa `0` cuando:
- La estrella queda apoyada en el borde del recipiente.
- La estrella termina fuera del recipiente al finalizar la ejecución.
- La estrella fue colocada correctamente pero el robot la desplazó fuera en un movimiento posterior.

### 2.3 Colocación de cubos (`cubo_*_destino`)

Se puntúa `1` cuando:
- El cubo se libera sobre la zona delimitada por la marca.
- La base del cubo queda completamente dentro de la zona marcada.
- El cubo permanece en dicha posición al finalizar la ejecución.

Se puntúa `0` cuando:
- El cubo cae fuera de la zona marcada.
- Parte de la base del cubo sobresale de la marca (criterio estricto por defecto).
- El cubo fue colocado correctamente pero el robot lo desplazó fuera en un movimiento posterior.

> **Tolerancia espacial:** Por defecto se usa el criterio estricto (base completamente dentro). Si se decide aplicar una tolerancia porcentual (p. ej., ≥80 % de la base dentro de la marca), esta **debe definirse antes de comenzar las evaluaciones** y mantenerse idéntica para todos los modelos evaluados.

### 2.4 Finalización limpia del episodio (`finalizacion_limpia`)

Se puntúa `1` cuando:
- El robot completa la ejecución sin intervención humana.
- No queda bloqueado en ningún momento.
- No entra en bucles o repeticiones innecesarias de movimiento.
- No permanece durante demasiado tiempo sin progresar hacia el siguiente objetivo.
- No muestra pausas prolongadas que indiquen indecisión sobre el siguiente paso.
- No realiza movimientos erráticos, innecesarios o claramente incompatibles con una política adecuada.
- La ejecución finaliza dentro del tiempo máximo establecido.

Se puntúa `0` cuando:
- El robot queda bloqueado.
- Repite movimientos sin avanzar.
- Permanece demasiado tiempo sin actuar.
- Parece no decidir cuál es el siguiente paso.
- Realiza una secuencia claramente errática.
- Necesita intervención humana para detener o reiniciar la ejecución.
- Supera el tiempo máximo establecido.

> **Tiempo máximo por ejecución:** `[PENDIENTE DE DEFINIR]`

---

## 3. Categorías de `modo_fallo`

Usar preferentemente una de las categorías siguientes. Si el fallo no encaja en ninguna, usar `otro` y describir en `notas`.

| Valor | Descripción |
|---|---|
| `ninguno` | Ejecución completada sin fallo relevante. |
| `sin_aproximacion` | El robot no alcanza o no intenta alcanzar correctamente el objeto objetivo. |
| `agarre_fallido` | La pinza no consigue sujetar y elevar el objeto. |
| `objeto_caido` | El objeto se agarra correctamente pero cae durante el transporte. |
| `destino_incorrecto` | El objeto se coloca en un destino que no le corresponde (p. ej., cubo en el recipiente). |
| `colocacion_imprecisa` | El destino es correcto pero el objeto queda fuera del criterio espacial establecido. |
| `colision` | Se produce una colisión relevante con objetos, recipiente, mesa u otros elementos. |
| `bloqueo` | La política queda detenida, repite movimientos cíclicos, no progresa hacia el objetivo, muestra pausas prolongadas, indecisión aparente o incapacidad de continuar la secuencia. |
| `timeout` | La ejecución no finaliza dentro del tiempo máximo. |
| `otro` | Fallo no cubierto por las categorías anteriores; describir en `notas`. |

---

## 4. Categorías de `objeto_fallo`

Indica qué objeto protagonizó el fallo principal de la ejecución.

| Valor | Cuándo usarlo |
|---|---|
| `ninguno` | No hubo fallo. |
| `estrella_negra` | El fallo se asocia principalmente a la estrella negra. |
| `estrella_naranja` | El fallo se asocia principalmente a la estrella naranja. |
| `cubo_negro` | El fallo se asocia principalmente al cubo negro. |
| `cubo_naranja` | El fallo se asocia principalmente al cubo naranja. |
| `varios` | El fallo afectó a más de un objeto sin un objeto principal. |
| `no_aplica` | No aplica (p. ej., `modo_fallo = bloqueo` antes de cualquier manipulación). |

---

## 5. Categorías de `fase_fallo`

Indica en qué fase del ciclo de manipulación ocurrió el fallo.

| Valor | Descripción |
|---|---|
| `ninguna` | No hubo fallo. |
| `aproximacion` | El robot no se aproxima correctamente al objeto objetivo. |
| `agarre` | El robot alcanza el objeto pero no lo sujeta. |
| `elevacion` | El objeto se sujeta pero no se eleva completamente de la mesa. |
| `transporte` | El objeto se eleva correctamente pero cae durante el trayecto al destino. |
| `colocacion` | El robot llega al destino pero no libera el objeto correctamente. |
| `liberacion` | El objeto se libera pero queda fuera del criterio espacial. |
| `finalizacion` | El robot no completa la ejecución autónomamente tras haber manipulado objetos. |
| `varias` | El fallo abarca más de una fase o es difícil de asignar a una única fase. |

---

## 6. Procedimiento de evaluación por ejecución

1. Preparar la escena según `starBlackPos`, `starOrangePos`, `cubeBlackPos`, `cubeOrangePos` y `rot` del CSV.
2. Lanzar el modelo indicado en `modelo` con los bloques desactivados según `bloques_desactivados`.
3. Dejar correr la ejecución hasta que finalice autónomamente o se alcance el límite de tiempo máximo.
4. Rellenar los 9 criterios binarios inmediatamente al finalizar.
5. Ejecutar `calculate_evaluation_scores.py` para calcular `puntuacion_bruta`, `puntuacion_normalizada` y `exito_completo`.
6. Completar `modo_fallo`, `objeto_fallo`, `fase_fallo` y `notas` si hubo incidencias.

> **Regla de oro:** Nunca modificar los criterios de puntuación durante el proceso de evaluación. Cualquier cambio de criterio invalida las comparaciones entre modelos ya evaluados.
