# SmolVLA-M — Guía de conversión y entrenamiento

> **Última actualización:** 2026-06-29

---

## Por qué el proceso es distinto a vanilla SmolVLA

SmolVLA-M añade **memoria temporal corta** al ViT de SigLIP: en vez de procesar un solo
frame por cámara, procesa K=6 frames consecutivos con atención temporal causal.

Implicaciones para el dataset y entrenamiento:

| Aspecto | Vanilla SmolVLA | SmolVLA-M |
|---------|----------------|-----------|
| Frames por observación | 1 | 6 |
| `n_obs_steps` | 1 | 6 |
| `observation_delta_indices` | `[0]` | `[-5,-4,-3,-2,-1,0]` |
| Nuevos parámetros entrenables | 0 | 3 escalares (`temporal_alpha`) |
| Dataset compatible | Cualquier LeRobot | **Los mismos datasets** ✓ |
| Coste de inferencia (LLM) | — | Idéntico (tokens del ViT = mismos) |

El punto clave es que **no hace falta grabar nuevos episodios**: LeRobot rellena el comienzo
de los episodios repitiendo el frame más antiguo disponible, por lo que los datasets
existentes funcionan directamente con `n_obs_steps=6`.

---

## Paso 1 — Convertir SmolVLA base a SmolVLA-M

```bash
python ./lerobot/scripts/convert_smolvla_to_smolvla_m.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_m_base
```

**Qué hace el script:**
- Descarga (o copia) el checkpoint vanilla de SmolVLA.
- Parchea `config.json`:
  - `"type": "smolvla_m"`
  - `"n_obs_steps": 6`  (era 1)
  - Añade `temporal_num_frames`, `temporal_stride`, `temporal_vit_layers`
- **No toca los pesos.** Cuando SmolVLA-M carga desde el checkpoint convertido,
  `PreTrainedPolicy.from_pretrained` usa `strict=False`:
  - Todos los pesos del ViT, LLM y expert → cargados desde el checkpoint de SmolVLA ✓
  - `temporal_alpha` (3 escalares, init=0) → se quedan en zero-init ✓

**Verificar la conversión:**
```bash
python ./lerobot/scripts/convert_smolvla_to_smolvla_m.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_m_base \
    --validate
```
La validación carga el modelo, verifica que alpha=0, y comprueba que K=1 produce
salida idéntica a vanilla (retrocompatibilidad garantizada).

---

## Paso 2 — Entrenamiento con datasets existentes

Los episodios grabados para los experimentos anteriores (abla de capas, etc.)
son compatibles directamente.

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=outputs/smolvla_m_base \
    --dataset.repo_id=Esk1z0/tfm_smolvla_v1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_smolvla_v1" \
    --policy.n_obs_steps=6 \
    --policy.temporal_stride=1 \
    --batch_size=8 \
    --steps=90000 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=10000 \
    --log_freq=50 \
    --output_dir=outputs/train/smolvla_m_v1 \
    --job_name=tfm_smolvla_m_v1 \
    --policy.device=cuda \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.push_to_hub=false \
    --policy.scheduler_decay_steps=86000 \
    --dataset.image_transforms.enable=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=tfm_smolvla_m
```

**Parámetros que se entrenan:**
- `temporal_alpha` (×3 escalares) — los únicos parámetros nuevos del ViT
- `state_proj`, `action_in/out_proj`, `action_time_mlp_*` — igual que vanilla

**Nota sobre batch_size:** con K=6 el ViT procesa 6× más imágenes por step.
Si hay OOM, reducir `batch_size` a la mitad respecto a vanilla (e.g., 8 → 4) o
activar `gradient_checkpointing`.

### Stride de 1 segundo (~1s entre frames)

Para reproducir el stride del paper MEM (~1 segundo entre frames a 20 Hz):

```bash
    --policy.temporal_stride=20 \
```

Esto requiere que los episodios tengan al menos 101 steps (`(K-1)×stride + 1 = 5×20+1`).
Los episodios más cortos funcionan con el padding de LeRobot.

---

## Paso 3 — Grabar nuevos episodios (opcional, para mejor rendimiento)

SmolVLA-M funciona con los datasets existentes, pero el paper MEM muestra mejoras
cuando se entrena con stride ~1 segundo (frames temporalmente distintos). Para grabar
datos nuevos con cameras estándar:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/robots/so_follower" \
    --robot.cameras='{
        camera1: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"},
        camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_smolvla_m_v1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_smolvla_m_v1" \
    --dataset.num_episodes=60 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.num_image_writer_processes=2 \
    --dataset.push_to_hub=false
```

> Cámaras estándar 640×480 — igual que los experimentos anteriores.
> La única diferencia con vanilla SmolVLA es `--policy.n_obs_steps=6` al entrenar.

---

## Paso 4 — Evaluación (inferencia)

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{
        camera1: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"},
        camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=false \
    --dataset.repo_id=Esk1z0/eval_smolvla_m_v1 \
    --dataset.num_episodes=15 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/smolvla_m_v1/checkpoints/last/pretrained_model
```

> Durante la inferencia, LeRobot gestiona automáticamente el buffer de K=6 frames:
> en los primeros pasos del episodio se repite el frame actual hasta llenar el buffer.
> No hay ningún cambio necesario en el código de inferencia — `n_obs_steps=6` se
> lee del checkpoint y se aplica automáticamente.

---

## Compatibilidad con SmolVLA-D

SmolVLA-M y SmolVLA-D son extensiones independientes de vanilla SmolVLA.
Por ahora **no se combinan** (SmolVLA-MD sería un trabajo futuro).

| Modelo | Temporal | Profundidad | Dataset |
|--------|----------|-------------|---------|
| SmolVLA | ✗ | ✗ | estándar, n_obs=1 |
| SmolVLA-M | ✓ (K=6 frames) | ✗ | estándar, n_obs=6 |
| SmolVLA-D | ✗ | ✓ (estéreo) | camera2 a 2560×800 |

---

## Diagnóstico rápido

### Verificar que temporal_alpha son los únicos parámetros nuevos del VLM

```python
from lerobot.policies.smolvla_m.modeling_smolvla_m import SmolVLAMPolicy

p = SmolVLAMPolicy.from_pretrained("outputs/smolvla_m_base")
vlm_trainable = [(n, param.shape)
                 for n, param in p.model.vlm_with_expert.vlm.named_parameters()
                 if param.requires_grad]
print(vlm_trainable)
# Expected: [('model.vision_model.encoder.temporal_alpha.0', torch.Size([1])),
#            ('model.vision_model.encoder.temporal_alpha.1', torch.Size([1])),
#            ('model.vision_model.encoder.temporal_alpha.2', torch.Size([1]))]
```

### Verificar que K=6 y K=1 producen misma forma de embedding

```python
import torch
vm = p.model.vlm_with_expert
B, K = 2, 6
with torch.no_grad():
    e1 = vm.embed_image(torch.zeros(B, 3, 512, 512), n_frames=1)
    eK = vm.embed_image(torch.zeros(B*K, 3, 512, 512), n_frames=K)
print(e1.shape, eK.shape)  # both → torch.Size([2, 64, 960])
```

---

## Resumen de cambios necesarios vs vanilla SmolVLA

| Qué | Vanilla SmolVLA | SmolVLA-M |
|-----|----------------|-----------|
| Script conversión | — | `convert_smolvla_to_smolvla_m.py` |
| `n_obs_steps` | 1 | 6 |
| `temporal_stride` | — | 1 (ajustable) |
| Dataset | cualquiera | **mismo que antes** ✓ |
| Parámetros nuevos | 0 | 3 escalares |
| Coste GPU (ViT) | 1× | 6× (K frames por step) |
| Coste GPU (LLM) | 1× | **1×** (tokens idénticos) |
| Inferencia | sin cambios | sin cambios |
