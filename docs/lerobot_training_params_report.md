# Informe de Parámetros de Entrenamiento LeRobot — SmolVLA

> Basado en el código fuente de `lerobot/src/lerobot/` (rama actual del submodule).  
> Comando base de referencia: `lerobot-train --policy.path=lerobot/smolvla_base ...`

---

## Índice

1. [Parámetros Globales de Entrenamiento](#1-parámetros-globales-de-entrenamiento)
2. [Parámetros del Dataset](#2-parámetros-del-dataset)
3. [Parámetros de la Política SmolVLA](#3-parámetros-de-la-política-smolvla)
4. [Optimizador](#4-optimizador)
5. [Learning Rate Scheduler](#5-learning-rate-scheduler)
6. [PEFT — Fine-tuning eficiente en parámetros](#6-peft--fine-tuning-eficiente-en-parámetros)
7. [Transformaciones de Imagen (Data Augmentation)](#7-transformaciones-de-imagen-data-augmentation)
8. [RA-BC — Reward-Aligned Behavior Cloning](#8-ra-bc--reward-aligned-behavior-cloning)
9. [WandB Logging](#9-wandb-logging)
10. [RTC — Real-Time Chunking](#10-rtc--real-time-chunking)
11. [Configuraciones Prometedoras](#11-configuraciones-prometedoras)

---

## 1. Parámetros Globales de Entrenamiento

Definidos en `configs/train.py → TrainPipelineConfig`.

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--output_dir` | auto-generado | Directorio donde se guardan checkpoints, logs y configuración del run. Si ya existe, falla a menos que `--resume=true`. |
| `--job_name` | `{policy_type}` | Nombre identificativo del experimento. Usado en logs y W&B. |
| `--resume` | `false` | Retomar un run previo desde su checkpoint. Requiere que `--output_dir` apunte a un run existente. |
| `--seed` | `1000` | Semilla global para inicialización del modelo y shuffling del dataset. `null` = no reproducible. |
| `--cudnn_deterministic` | `false` | Activa algoritmos CuDNN deterministas. Mejora reproducibilidad a costa de ~10-20% de velocidad. |
| `--num_workers` | `4` | Número de workers del DataLoader. Aumentar puede acelerar la carga de datos. |
| `--batch_size` | `8` | Tamaño del batch. Con `4` (el que usas) consumes menos VRAM pero el gradiente es más ruidoso. |
| `--steps` | `100_000` | Número total de pasos de entrenamiento (no epochs). |
| `--eval_freq` | `20_000` | Cada cuántos pasos evaluar en entorno simulado (sólo si `env` está configurado). |
| `--log_freq` | `200` | Cada cuántos pasos loggear métricas. |
| `--save_checkpoint` | `true` | Guardar checkpoints durante entrenamiento. |
| `--save_freq` | `20_000` | Frecuencia de guardado de checkpoints (en pasos). |
| `--use_policy_training_preset` | `true` | Usar los presets de optimizador/scheduler definidos en la política. Si es `false`, debes especificar `--optimizer` y `--scheduler` manualmente. |
| `--rename_map` | `{}` | Mapeo JSON para renombrar claves de observación del dataset. **Crítico** cuando los nombres de cámara del dataset no coinciden con los esperados por la política. |

---

## 2. Parámetros del Dataset

Definidos en `configs/default.py → DatasetConfig`. Se acceden con el prefijo `--dataset.`.

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--dataset.repo_id` | *(requerido)* | ID del dataset en HuggingFace Hub, e.g., `Esk1z0/mi_dataset`. |
| `--dataset.root` | `None` | Ruta local al dataset. Si se especifica, se usa el caché local en lugar de descargar del Hub. |
| `--dataset.episodes` | `None` | Lista de índices de episodios a usar, e.g., `[0,1,2,5]`. Útil para entrenar sólo con un subconjunto. |
| `--dataset.revision` | `None` | Versión/revisión del dataset en el Hub (git commit hash o tag). Fija la versión para reproducibilidad. |
| `--dataset.use_imagenet_stats` | `true` | Normalizar imágenes con media/std de ImageNet. Recomendado cuando el backbone VLM fue preentrenado en ImageNet. |
| `--dataset.video_backend` | auto | Backend de vídeo. Auto-selecciona el codec disponible más eficiente. |
| `--dataset.streaming` | `false` | Streaming directo desde Hub sin descargar el dataset completo. Útil para datasets grandes, pero más lento. |
| `--dataset.image_transforms.enable` | `false` | Activar data augmentation en imágenes durante entrenamiento (ver sección 7). |

---

## 3. Parámetros de la Política SmolVLA

Definidos en `policies/smolvla/configuration_smolvla.py → SmolVLAConfig`. Se acceden con el prefijo `--policy.`.

### 3.1 Estructura de entrada/salida

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--policy.n_obs_steps` | `1` | Número de pasos de observación pasados que se dan al modelo. Con `1`, sólo la observación actual. |
| `--policy.chunk_size` | `50` | Tamaño del chunk de acciones predicho por el modelo en cada forward pass. |
| `--policy.n_action_steps` | `50` | Número de acciones que se ejecutan antes de pedir una nueva predicción. Debe ser `<= chunk_size`. |
| `--policy.max_state_dim` | `32` | Dimensión máxima del vector de estado. Vectores más cortos se padean. |
| `--policy.max_action_dim` | `32` | Dimensión máxima del vector de acción. Vectores más cortos se padean. |

### 3.2 Preprocesado de imagen

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--policy.resize_imgs_with_padding` | `(512, 512)` | Resolución a la que se redimensionan las imágenes (con padding). Mayor resolución = más detalle pero más VRAM y tiempo. |
| `--policy.empty_cameras` | `0` | Añadir N cámaras vacías (negras) como entrada adicional. Útil en setups multi-cámara donde faltan algunas. **En tu comando usas `1`**. |

### 3.3 Tokenizer y decodificación

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--policy.tokenizer_max_length` | `48` | Longitud máxima de la secuencia de texto (instrucción de lenguaje). Instrucciones más largas serán truncadas. |
| `--policy.num_steps` | `10` | Número de pasos de decodificación iterativa del action expert. Más pasos = predicciones más refinadas pero más lento. |
| `--policy.use_cache` | `true` | Usar KV-cache durante la decodificación. Acelera la inferencia. |
| `--policy.pad_language_to` | `"longest"` | Padding del texto: `"longest"` (al más largo del batch) o `"max_length"` (a `tokenizer_max_length`). |

### 3.4 Arquitectura del modelo

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--policy.vlm_model_name` | `HuggingFaceTB/SmolVLM2-500M-Video-Instruct` | Backbone VLM a usar. Determina la arquitectura base. |
| `--policy.num_vlm_layers` | `16` | Número de capas del VLM que se usan (primeras N). **Clave para ablaciones de capas**: reducir este valor elimina capas del backbone. |
| `--policy.num_expert_layers` | `-1` | Capas del action expert. `-1` = igual que el VLM. Reducir permite un expert más ligero que el VLM. |
| `--policy.expert_width_multiplier` | `0.75` | Factor del tamaño oculto del action expert respecto al VLM. `0.75` = 75% del tamaño del VLM. |
| `--policy.self_attn_every_n_layers` | `2` | Cada cuántas capas insertar una capa de self-attention en el expert. |
| `--policy.attention_mode` | `"cross_attn"` | Modo de atención entre VLM y action expert. `"cross_attn"` es el default. |
| `--policy.prefix_length` | `-1` | Longitud del prefijo de atención. `-1` = auto. |
| `--policy.add_image_special_tokens` | `false` | Añadir tokens especiales alrededor de las features de imagen. |

### 3.5 Codificación posicional

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--policy.min_period` | `4e-3` | Periodo mínimo para la codificación posicional seno-coseno del timestep. |
| `--policy.max_period` | `4.0` | Periodo máximo. El rango `[min_period, max_period]` define la sensibilidad temporal. |

### 3.6 Estrategia de fine-tuning

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--policy.freeze_vision_encoder` | `true` | Congelar el encoder visual del VLM durante el entrenamiento. Ahorra VRAM y acelera training; puede reducir adaptación visual. |
| `--policy.train_expert_only` | `true` | Entrenar **sólo** el action expert, congelando el VLM completo. Muy eficiente en parámetros. |
| `--policy.train_state_proj` | `true` | Entrenar la proyección del estado del robot. |
| `--policy.load_vlm_weights` | `false` | Cargar los pesos del VLM desde el checkpoint (cuando `false`, el VLM se inicializa desde pretrained SmolVLM2, no desde pesos de SmolVLA). Al usar `--policy.path=lerobot/smolvla_base` esto se gestiona automáticamente. |

### 3.7 Compilación del modelo

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--policy.compile_model` | `false` | Usar `torch.compile` para optimizar el modelo. Puede dar 10-30% de speedup tras la compilación inicial (~5 min). |
| `--policy.compile_mode` | `"max-autotune"` | Modo de compilación de Torch. `"reduce-overhead"` es más rápido de compilar; `"max-autotune"` optimiza más agresivamente. |

### 3.8 Configuraciones Aloha (no relevantes para SO-101)

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--policy.adapt_to_pi_aloha` | `false` | Conversión de espacio de acciones para el robot Aloha. No necesario para SO-101. |
| `--policy.use_delta_joint_actions_aloha` | `false` | Acciones relativas al estado actual para Aloha. No implementado aún. |

---

## 4. Optimizador

Definido en `optim/optimizers.py`. Se accede con `--optimizer.` (cuando `--use_policy_training_preset=false`).

Por defecto, SmolVLA usa **AdamW** con los presets de `SmolVLAConfig`:

| Parámetro | Defecto SmolVLA | Descripción |
|---|---|---|
| `--policy.optimizer_lr` | `1e-4` | Learning rate base. El scheduler lo modifica durante el entrenamiento. |
| `--policy.optimizer_betas` | `(0.9, 0.95)` | Coeficientes beta de Adam. Beta2=0.95 (vs 0.999 estándar) es más agresivo para VLMs. |
| `--policy.optimizer_eps` | `1e-8` | Término epsilon para estabilidad numérica. |
| `--policy.optimizer_weight_decay` | `1e-10` | Weight decay muy pequeño; prácticamente sin regularización L2. |
| `--policy.optimizer_grad_clip_norm` | `10.0` | Norma máxima del gradiente (gradient clipping). Previene explosión de gradientes. |

### Optimizadores disponibles

| Tipo | Cuándo usarlo |
|---|---|
| `adamw` | Default para SmolVLA. Robusto y estable. |
| `adam` | Sin weight decay. Para datasets pequeños donde la regularización no ayuda. |
| `sgd` | Raramente útil para fine-tuning de VLMs. |
| `xvla-adamw` | AdamW con LR diferencial: VLM al 10% del LR base, útil para fine-tuning full donde no quieres degradar el VLM. |

---

## 5. Learning Rate Scheduler

Definido en `optim/schedulers.py`. SmolVLA usa **CosineDecayWithWarmup**.

| Parámetro | Defecto SmolVLA | Descripción |
|---|---|---|
| `--policy.scheduler_warmup_steps` | `1_000` | Pasos de warmup lineal desde LR≈0 hasta `optimizer_lr`. |
| `--policy.scheduler_decay_steps` | `30_000` | Pasos totales del decay coseno. Si `steps < decay_steps`, se auto-escala proporcionalmente. |
| `--policy.scheduler_decay_lr` | `2.5e-6` | LR mínimo al final del decay coseno. |

### Comportamiento del scheduler

```
LR
 │     /‾‾‾\
 │    /      \__________
 │   /
 │__/
 └─────────────────────── steps
     warmup   cosine decay
```

**Nota clave**: Si `steps < scheduler_decay_steps`, LeRobot auto-escala el warmup y el decay para que el schedule completo quepa en los pasos disponibles. Con `--steps=500` y `scheduler_decay_steps=30_000`, el warmup real será de `500 * (1000/30000) ≈ 16 pasos`.

---

## 6. PEFT — Fine-tuning Eficiente en Parámetros

Definido en `configs/default.py → PeftConfig`. Se activa añadiendo `--peft.method_type=LORA`.

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--peft.method_type` | `"LORA"` | Método PEFT a aplicar. Actualmente soportado: `"LORA"`. |
| `--peft.r` | `16` | Rango de la descomposición LoRA. Mayor rango = más parámetros entrenables = más cercano a full fine-tuning. Valores típicos: 4, 8, 16, 32. |
| `--peft.target_modules` | `None` | Módulos a los que aplicar LoRA. `None` = usa el default de la política. `"all-linear"` = todas las capas lineales. |
| `--peft.full_training_modules` | `None` | Módulos a entrenar completamente (sin LoRA). Por defecto son los módulos nuevos no preentrenados (e.g., proyecciones de estado/acción). |
| `--peft.init_type` | `None` | Método de inicialización del adaptador LoRA. |

### Cuándo usar PEFT/LoRA

- Cuando `train_expert_only=false` y quieres fine-tunear partes del VLM sin explotar VRAM.
- Permite entrenar más parámetros con el mismo budget de memoria.
- **Combinación recomendada**: `train_expert_only=false` + `freeze_vision_encoder=true` + LoRA en capas de atención del LLM.

---

## 7. Transformaciones de Imagen (Data Augmentation)

Definido en `datasets/transforms.py → ImageTransformsConfig`. Se activa con `--dataset.image_transforms.enable=true`.

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--dataset.image_transforms.enable` | `false` | Activar augmentations. **Por defecto desactivado**. |
| `--dataset.image_transforms.max_num_transforms` | `3` | Número máximo de transformaciones aplicadas por frame (de las disponibles). |
| `--dataset.image_transforms.random_order` | `false` | Aplicar transformaciones en orden aleatorio (vs el orden sugerido por Torchvision). |

### Transformaciones disponibles (prefijo `--dataset.image_transforms.tfs.`)

| Transform | Efecto | kwargs por defecto |
|---|---|---|
| `brightness` | ColorJitter en brillo | `brightness: (0.8, 1.2)` |
| `contrast` | ColorJitter en contraste | `contrast: (0.8, 1.2)` |
| `saturation` | ColorJitter en saturación | `saturation: (0.5, 1.5)` |
| `hue` | ColorJitter en tono | `hue: (-0.05, 0.05)` |
| `sharpness` | Nitidez aleatoria | `sharpness: (0.5, 1.5)` |
| `affine` | Rotación + traslación | `degrees: (-5, 5), translate: (0.05, 0.05)` |

**Ejemplo de uso**:
```bash
--dataset.image_transforms.enable=true \
--dataset.image_transforms.max_num_transforms=2
```

---

## 8. RA-BC — Reward-Aligned Behavior Cloning

Definido en `configs/train.py`. Requiere un fichero `sarm_progress.parquet` precomputado.

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--use_rabc` | `false` | Activar entrenamiento ponderado por recompensa. Sólo útil si tienes trayectorias etiquetadas con calidad/recompensa. |
| `--rabc_progress_path` | auto-detectado | Ruta al fichero parquet con el progreso SARM (Self-Assessed Robot Metric). Si `None`, busca en `dataset.root/sarm_progress.parquet`. |
| `--rabc_kappa` | `0.01` | Umbral para muestras de alta calidad. Muestras con progreso < kappa se descartan/pesan menos. |
| `--rabc_epsilon` | `1e-6` | Constante de estabilidad numérica. |
| `--rabc_head_mode` | `"sparse"` | Para modelos dual-head: `"sparse"` o `"dense"`. |

---

## 9. WandB Logging

Definido en `configs/default.py → WandBConfig`. Prefijo `--wandb.`.

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--wandb.enable` | `false` | Activar logging en Weights & Biases. |
| `--wandb.project` | `"lerobot"` | Nombre del proyecto en W&B. |
| `--wandb.entity` | `None` | Organización/usuario de W&B. |
| `--wandb.notes` | `None` | Notas del run (texto libre). |
| `--wandb.mode` | `None` | `"online"`, `"offline"`, o `"disabled"`. |
| `--wandb.add_tags` | `true` | Guardar la configuración como tags del run para facilitar filtrado. |
| `--wandb.disable_artifact` | `false` | Deshabilitar subida de artefactos (checkpoints) aunque `save_checkpoint=true`. |

---

## 10. RTC — Real-Time Chunking

Definido en `policies/rtc/configuration_rtc.py → RTCConfig`. Prefijo `--policy.rtc_config.`.

Sistema experimental para mejorar la inferencia en tiempo real tratando la generación de chunks como un problema de inpainting.

| Parámetro | Defecto | Descripción |
|---|---|---|
| `--policy.rtc_config.enabled` | `false` | Activar RTC. Sólo relevante para **inferencia**, no entrena diferente. |
| `--policy.rtc_config.execution_horizon` | `10` | Número de acciones ejecutadas antes de re-predicción con RTC. |
| `--policy.rtc_config.max_guidance_weight` | `10.0` | Peso de guidance para el inpainting. |
| `--policy.rtc_config.prefix_attention_schedule` | `LINEAR` | Schedule de atención al prefijo: `LINEAR` o `EXP`. |

---

## 11. Configuraciones Prometedoras

A continuación se presentan configuraciones ordenadas por objetivos concretos para tu setup (SO-101, 2 cámaras, dataset de ablación de capas, batch=4, GPU CUDA).

---

### Config A — Baseline rápido (tu configuración actual)

> Objetivo: Verificar que el pipeline funciona. Mínima carga computacional.

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockup_try_resume \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_mockup_try_resume" \
    --batch_size=4 \
    --steps=500 \
    --output_dir=outputs/train/config_A_baseline \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

---

### Config B — Fine-tuning real (5k–20k pasos)

> Objetivo: Entrenamiento serio con configuración estándar SmolVLA. Punto de partida para ablaciones.

**Cambios respecto a baseline**: más pasos, WandB activo, checkpoints frecuentes, seed fija.

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockup_try_resume \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_mockup_try_resume" \
    --batch_size=8 \
    --steps=10000 \
    --seed=42 \
    --save_freq=2000 \
    --log_freq=50 \
    --output_dir=outputs/train/config_B_full_finetune \
    --job_name=smolvla_so101_config_B \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=tfm_smolvla_ablation \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

---

### Config C — Ablación de capas VLM (menos capas = modelo más pequeño)

> Objetivo: Estudiar el impacto del número de capas del VLM backbone en el rendimiento. **Core de tu TFM**.

Varía `--policy.num_vlm_layers` entre corridas. El modelo SO-101 base usa 16 capas.

```bash
# Ejemplo: ablación con 8 capas VLM (50% del modelo)
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockup_try_resume \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_mockup_try_resume" \
    --batch_size=8 \
    --steps=10000 \
    --seed=42 \
    --output_dir=outputs/train/config_C_8layers \
    --job_name=smolvla_so101_8vlm_layers \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.num_vlm_layers=8 \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=tfm_smolvla_ablation \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

**Valores sugeridos para la ablación**: `num_vlm_layers ∈ {4, 6, 8, 10, 12, 14, 16}`

---

### Config D — Fine-tuning del VLM completo (sin congelar)

> Objetivo: Comparar con el fine-tuning sólo del expert (Config B). Mayor costo pero potencialmente mayor adaptación al dominio.

**Cambios**: `train_expert_only=false`, `freeze_vision_encoder=false`, LR más bajo.

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockup_try_resume \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_mockup_try_resume" \
    --batch_size=4 \
    --steps=10000 \
    --seed=42 \
    --output_dir=outputs/train/config_D_full_vlm \
    --job_name=smolvla_so101_full_vlm \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.train_expert_only=false \
    --policy.freeze_vision_encoder=true \
    --policy.optimizer_lr=5e-5 \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=tfm_smolvla_ablation \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

> **Nota**: Con `train_expert_only=false` necesitarás más VRAM. Reducir `batch_size=4` o activar `--policy.compile_model=true` para optimizar memoria.

---

### Config E — LoRA sobre el VLM (eficiencia de parámetros)

> Objetivo: Fine-tunear más partes del modelo sin explotar VRAM. Ideal como alternativa a Config D si tienes limitaciones de memoria.

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockup_try_resume \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_mockup_try_resume" \
    --batch_size=8 \
    --steps=10000 \
    --seed=42 \
    --output_dir=outputs/train/config_E_lora \
    --job_name=smolvla_so101_lora_r16 \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.train_expert_only=false \
    --peft.method_type=LORA \
    --peft.r=16 \
    --peft.target_modules=all-linear \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=tfm_smolvla_ablation \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

---

### Config F — Con Data Augmentation

> Objetivo: Mejorar la generalización cuando el dataset es pequeño (que es el caso de ablaciones). Especialmente útil para variaciones de iluminación y posición.

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockup_try_resume \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_mockup_try_resume" \
    --batch_size=8 \
    --steps=10000 \
    --seed=42 \
    --output_dir=outputs/train/config_F_augmentation \
    --job_name=smolvla_so101_augmented \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --dataset.image_transforms.enable=true \
    --dataset.image_transforms.max_num_transforms=3 \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.project=tfm_smolvla_ablation \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

---

### Config G — LR tuning (learning rate alto vs. bajo)

> Objetivo: Encontrar el LR óptimo para tu dataset. LR demasiado alto → inestabilidad; demasiado bajo → convergencia lenta.

```bash
# LR alto (agresivo)
--policy.optimizer_lr=3e-4 \
--policy.scheduler_warmup_steps=500

# LR estándar SmolVLA
--policy.optimizer_lr=1e-4 \
--policy.scheduler_warmup_steps=1000

# LR conservador
--policy.optimizer_lr=3e-5 \
--policy.scheduler_warmup_steps=200
```

---

### Config H — Torch Compile (optimización de velocidad)

> Objetivo: Acelerar el entrenamiento en GPU. La compilación inicial toma ~3-5 min pero luego hay un 15-30% de speedup.

```bash
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockup_try_resume \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_mockup_try_resume" \
    --batch_size=8 \
    --steps=10000 \
    --seed=42 \
    --output_dir=outputs/train/config_H_compiled \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.compile_model=true \
    --policy.compile_mode=reduce-overhead \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'
```

---

## Resumen de Decisiones Clave

| Decisión | Opción A | Opción B | Recomendación TFM |
|---|---|---|---|
| **¿Qué entrenar?** | Sólo action expert (`train_expert_only=true`) | VLM + expert | Empieza con A, B como ablación |
| **¿Congelar visual?** | Sí (`freeze_vision_encoder=true`) | No | A (menos VRAM, más rápido) |
| **Nº capas VLM** | 16 (completo) | 4-14 (ablación) | Variar para la ablación |
| **LoRA** | No | Rank 16 | Si VRAM limitada al hacer full ft |
| **Data augmentation** | Desactivado | Activado | Activar si el dataset es pequeño |
| **Torch compile** | No | `reduce-overhead` | Activar para runs largos |
| **Batch size** | 4 (actual) | 8-16 | Subir si VRAM lo permite |
| **WandB** | Desactivado | Activado | **Activar siempre para el TFM** |

---

*Informe generado el 2026-06-15. Basado en el código fuente del submodule `lerobot` en `/home/juanes/OneDrive/TFM/pruebas/lerobot/src/`.*
