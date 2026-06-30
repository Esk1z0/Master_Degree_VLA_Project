# Evaluación Final VLA

Este directorio contiene las herramientas necesarias para registrar y analizar las pruebas físicas de los modelos Vision-Language-Action (VLA).

## Flujo de Trabajo

### 1. Preparar la evaluación de un modelo

1. La plantilla base es `plantilla_evaluacion_modelo.csv`. Este archivo **no debe rellenarse directamente**.
2. Copia y renombra la plantilla para el modelo que vas a evaluar. Utiliza un nombre identificativo claro, por ejemplo: `evaluacion_M0.csv`, `evaluacion_SmolVLA_base.csv`. El nombre del archivo se utilizará como identificador del modelo en los resultados globales.

### 2. Rellenar los campos durante el experimento

Para cada uno de los 18 intentos, debes rellenar los siguientes bloques de datos:

#### A. Validación del Setup
* `setup_validado` (1/0): Indica si las posiciones de los objetos en la mesa coinciden con la configuración (columnas `Pos`).
* `objeto_visible_inicio` (1/0): *Solo en autooclusión y combinado.* Indica si el objeto que debe ser ocluido es visible por el robot antes de iniciar el movimiento.
* `oclusion_verificada` (1/0): *Solo en autooclusión y combinado.* Indica si durante la ejecución el brazo llegó a tapar visualmente el objeto objetivo de la métrica.
* `notas_setup`: Texto libre para observaciones sobre la configuración antes del inicio.

#### B. Puntuación de Destreza (1/0)
* `[objeto]_agarre`: 1 si el robot sujeta el objeto correctamente e inicia su transporte.
* `[objeto]_destino`: 1 si la estrella cae en el recipiente o el cubo queda estable sobre la zona azul.
* `finalizacion_limpia`: 1 si el robot finaliza correctamente sin desordenar elementos, sin colisiones graves y sin requerir intervención humana.

#### C. Métricas de Memoria (1/0)
* *Solo aplican a `autooclusion` y `combinado`.*
* `recuerda_objeto_ocluido`: 1 si tras la oclusión el robot redirige la atención/movimiento hacia el objeto pendiente.
* `completa_objeto_ocluido`: 1 si el robot termina correctamente la manipulación del objeto que fue ocluido.

#### D. Fotografía y Profundidad (1/0 y recuento)
* *Solo aplican a `fotografia` y `combinado`.*
* `selecciona_real_antes_foto` (1/0): 1 si la primera intención de agarre es sobre el objeto físico y no la foto.
* `evita_intento_agarre_foto` (1/0): 1 si en toda la ejecución no hay cierre de pinza ni impacto directo sobre la fotografía.
* `aproximaciones_foto` (número): Cuántas veces el brazo se aproxima intencionadamente a la fotografía.
* `intentos_agarre_foto` (número): Cuántas veces el robot intenta ejecutar un agarre (cierre de pinza) sobre la fotografía.

#### E. Fallos y Seguridad
* `tiempo_total_s` (número): Duración de la ejecución en segundos.
* `timeout` (1/0): 1 si el robot excede el tiempo límite establecido o entra en bucle infinito.
* `intervencion_emergencia` (1/0): 1 si el operador humano tuvo que detener físicamente al robot para evitar daños.
* `modo_fallo`, `objeto_fallo`, `fase_fallo`: Categorización del error (ej. *agarre_fallido*, *estrella_negra*, *transporte*).

---

## Análisis de Resultados

### 1. Ejecutar el script

Para procesar un solo modelo:
```bash
python calcular_resultados_evaluacion.py evaluacion_M0.csv
```

Para procesar múltiples modelos y obtener una comparativa:
```bash
python calcular_resultados_evaluacion.py evaluacion_M0.csv evaluacion_M1.csv evaluacion_M2.csv
```

### 2. Archivos generados

* `<modelo>_calculado.csv`: Para cada archivo de entrada se genera un archivo de salida que contiene las 18 filas originales más todas las métricas derivadas calculadas fila por fila (`puntuacion_destreza`, `puntuacion_normalizada`, `exito_completo`, etc.).
* `resumen_global_evaluacion.csv`: Se genera **solo cuando se evalúan dos o más modelos**. Contiene promedios y agregaciones para cada escenario y una fila `global` por modelo. Permite comparar rápidamente tasas de éxito, precisión y métricas específicas.

---

## Conceptos Importantes del Análisis

### Validez del intento (`registro_valido`)
Las métricas promedio del resumen se calculan **únicamente sobre los registros válidos**. Un registro se considera válido cuando la configuración inicial y las condiciones del escenario se cumplieron correctamente en la mesa:
* Para `destreza_general` y `fotografia`: `setup_validado == 1`
* Para `autooclusion` y `combinado`: `setup_validado == 1 AND objeto_visible_inicio == 1 AND oclusion_verificada == 1`

### Éxito completo (`exito_completo`)
Un intento se considera un éxito absoluto (1) cuando:
1. Es un registro válido.
2. La puntuación total es igual a la puntuación máxima posible en ese escenario.
3. No hubo `timeout`.
4. No hubo `intervencion_emergencia`.
Si falla cualquiera de estas condiciones, el éxito completo es (0).
