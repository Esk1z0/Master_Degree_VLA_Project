# SmolVLA-D — Guía de grabación y entrenamiento

> **Última actualización:** 2026-06-29

---

## Por qué el proceso es distinto a vanilla SmolVLA

SmolVLA-D usa la cámara muñeca como cámara **estéreo**. El frame completo
(2560×800, ambos ojos lado a lado) se almacena en el dataset. El modelo lo
divide automáticamente durante el entrenamiento e inferencia:

| Mitad      | Destino                                              |
|------------|------------------------------------------------------|
| Izquierda  | VLM (pathway RGB normal, igual que vanilla SmolVLA)  |
| Frame completo | SGBM+WLS → nube de puntos → inyección en el expert |

Esto implica **dos diferencias** frente a los datasets de los experimentos
anteriores:

1. La cámara `camera2` debe capturar a **2560×800** (en lugar de 640×480).
2. El entrenamiento necesita `--policy.stereo_camera_keys='["observation.images.camera2"]'`.

---

## Paso 1 — Convertir SmolVLA base a SmolVLA-D

```bash
python ./lerobot/scripts/convert_smolvla_to_smolvla_d.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_d_base
```

**Qué hace el script:**
- Descarga (o copia) el checkpoint vanilla de SmolVLA.
- Parchea `config.json`: `"type": "smolvla_d"`.
- Añade los campos nuevos de SmolVLA-D con sus valores por defecto
  (`stereo_camera_keys`, `depth_injection_layers`, etc.) para que el config
  sea explícito y fácil de editar.
- **No toca los pesos.** Cuando SmolVLA-D carga desde el checkpoint convertido,
  `PreTrainedPolicy.from_pretrained` usa `strict=False`:
  - Pesos de VLM + expert → cargados desde el checkpoint de SmolVLA ✓
  - `PointCloudEncoder` + `DepthInjector` → se quedan con la inicialización
    (zero-init en los adapters → sin efecto al inicio) ✓

---

## Paso 2 — Grabar episodios para SmolVLA-D

> La única diferencia respecto a los experimentos anteriores es `camera2`:
> `width: 2560, height: 800, fps: 15`.

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/robots/so_follower" \
    --robot.cameras='{
        camera1: {type: opencv, index_or_path: "/dev/video0", width: 640,  height: 480, fps: 30, fourcc: "MJPG"},
        camera2: {type: opencv, index_or_path: "/dev/video2", width: 2560, height: 800, fps: 15, fourcc: "MJPG"}
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_smolvla_d_v1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_smolvla_d_v1" \
    --dataset.num_episodes=60 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.num_image_writer_processes=2 \
    --dataset.push_to_hub=false
```

> **Nota sobre FPS:** 2560×800 MJPG es mucho más ancho que 640×480. Se usa
> `fps: 15` para no saturar el bus USB. Si la cámara aguanta 30 fps sin
> perder frames, se puede subir.

---

## Paso 3 — Entrenamiento Fase 1 (sin profundidad)

En esta fase el modelo se entrena **como si fuera vanilla SmolVLA** pero usando
el frame estéreo (la mitad izquierda va al VLM). La inyección de profundidad
está dormida porque `batch["depth_point_cloud"]` no existe en el dataset
→ `depth_emb = None` → `delta = 0` (gracias al zero-init de los adapters).

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=outputs/smolvla_d_base \
    --dataset.repo_id=Esk1z0/tfm_smolvla_d_v1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_smolvla_d_v1" \
    --batch_size=16 \
    --steps=90000 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=10000 \
    --log_freq=50 \
    --output_dir=outputs/train/smolvla_d_phase1 \
    --job_name=tfm_smolvla_d_phase1 \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.push_to_hub=false \
    --policy.scheduler_decay_steps=86000 \
    --dataset.image_transforms.enable=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=tfm_smolvla_d
```

**Parámetros que se entrenan en Fase 1:**
- `state_proj`, `action_in/out_proj`, `action_time_mlp_*` (igual que vanilla)
- `PointCloudEncoder` y `DepthInjector` tienen gradientes pero reciben
  `None` como entrada → sus pesos no se actualizan en la práctica

---

## Paso 4 — Evaluación del modelo de Fase 1

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{
        camera1: {type: opencv, index_or_path: "/dev/video0", width: 640,  height: 480, fps: 30, fourcc: "MJPG"},
        camera2: {type: opencv, index_or_path: "/dev/video2", width: 2560, height: 800, fps: 15, fourcc: "MJPG"}
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=false \
    --dataset.repo_id=Esk1z0/eval_smolvla_d_phase1 \
    --dataset.num_episodes=15 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/smolvla_d_phase1/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.stereo_camera_keys='["observation.images.camera2"]'
```

---

## Paso 5 (futuro) — Entrenamiento Fase 2 con profundidad

Para entrenar la inyección de profundidad hay que pre-computar las nubes de
puntos del dataset. Flujo previsto:

```
Dataset grabado (Fase 1)
    └─ script: precompute_depth_features.py    ← todavía por crear
         └─ lee cada frame de camera2 (2560×800)
         └─ SGBM+WLS → reprojectImageTo3D
         └─ muestrea 1024 puntos → guarda como depth_point_cloud (B,1024,3)
Dataset enriquecido (Fase 2)
    └─ lerobot-train con depth_point_cloud en el batch
```

Durante la Fase 2:
- `batch["depth_point_cloud"]` llega al modelo → inyección activa
- `depth_dropout_prob=0.2` → 20% de los steps zerean la profundidad
  (robustez frente a frames sin depth válido)

---

## Resumen de cambios necesarios

| Qué           | Fase 1 (ya posible)         | Fase 2 (futuro)                    |
|---------------|-----------------------------|------------------------------------|
| Grabación     | camera2 a 2560×800, fps 15  | Misma grabación                    |
| Training flag | `stereo_camera_keys`        | `stereo_camera_keys` + depth dataset |
| Profundidad   | Dormida (delta=0)           | Activa con dropout 20%             |
| Parámetros nuevos entrenados | Ninguno (gradientes cero) | PointCloudEncoder + DepthInjector |
