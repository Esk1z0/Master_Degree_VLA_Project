# Repositorio TFM

Este repositorio contiene el código, datos y experimentos para el Trabajo de Fin de Máster (TFM) relacionados con el entrenamiento, ablación y evaluación del robot SO-101 usando modelos de Visión, Lenguaje y Acción (VLA).

## Estructura de Directorios

La estructura está diseñada para mantener organizado el código finalizado, los scripts de prueba y la documentación.

- `ablation/`: Scripts y utilidades para el estudio de ablación (ej. desactivación de capas en SmolVLA, tests de cortes).
- `camera/`: Jupyter notebooks y datos matriciales (`calib_stereo.npz`) para la calibración y prueba de las cámaras.
- `data/layouts/`: Contiene los archivos CSV que definen los layouts para el robot y evaluación.
- `docs/`: Rúbricas de evaluación, historial de comandos, reportes de hiperparámetros de entrenamiento y checklists de verificación.
- `scripts/`: Scripts utilitarios (ej. cálculo de puntuaciones de evaluación).
- `wip/`: (No rastreada por Git) Carpeta de "Work in Progress" para pruebas rápidas y archivos sin terminar.
- `outputs/` y `results/`: (No rastreadas por Git) Estas carpetas almacenan localmente los pesos, videos y configuraciones generados por LeRobot y wandb durante las ejecuciones, evitando saturar el repositorio.

## Datos y Modelos (HuggingFace)

Dado el tamaño de los datasets y los pesos del modelo, estos se alojan en HuggingFace.

- **Datasets:** [Enlace a tu Dataset en HuggingFace]
- **Modelos:** [Enlace a tu Modelo en HuggingFace]

*(Reemplaza los enlaces anteriores con tus URLs reales)*

## Entrenamiento y Reportes (Weights & Biases)

El seguimiento de los experimentos y el registro de métricas de entrenamiento están disponibles en Weights & Biases:

- **Proyecto WandB:** [Enlace a tu proyecto o reportes en WandB]

## Modificaciones a `lerobot` (Instrucciones)

La librería subyacente `lerobot` ha sido modificada en este entorno para corregir problemas de grabación (códecs de vídeo) y adaptar la ejecución al entorno del TFM.

Dado que actualmente la carpeta `lerobot` tiene modificaciones sin rastrear en un repositorio padre, **debes seguir estos pasos para integrarlo correctamente a GitHub como un submódulo**:

1. **Crear un Fork en GitHub:**
   Entra a [HuggingFace LeRobot en GitHub](https://github.com/huggingface/lerobot) y pulsa en "Fork" para clonarlo a tu cuenta personal (`tu-usuario/lerobot`).

2. **Subir tus Cambios a tu Fork:**
   Abre una terminal, entra en la carpeta `lerobot` local y empuja tus cambios a tu nuevo repositorio:
   ```bash
   cd lerobot
   git remote set-url origin https://github.com/tu-usuario/lerobot.git
   git add .
   git commit -m "Añadidas modificaciones para TFM (códecs, calibración, etc)"
   git push origin main
   ```

3. **Registrar el Submódulo en el Repositorio Principal (TFM):**
   Vuelve a la carpeta raíz del TFM y vincula oficialmente tu fork:
   ```bash
   cd ..
   # Desvincula la referencia actual a lerobot (sin borrar la carpeta)
   git rm --cached lerobot
   
   # Añade el submódulo oficial apuntando a tu URL
   git submodule add https://github.com/tu-usuario/lerobot.git lerobot
   
   git add .gitmodules lerobot
   git commit -m "Configurado submódulo lerobot apuntando a mi fork modificado"
   ```

A partir de este momento, cuando subas tus cambios a GitHub, el código en `lerobot` apuntará directamente a tu versión arreglada.
