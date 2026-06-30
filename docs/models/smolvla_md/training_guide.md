# SmolVLA-MD — Guía de conversión y entrenamiento

> **Última actualización:** 2026-06-29

---

## Diferencias clave vs vanilla SmolVLA

SmolVLA-MD combina las dos extensiones anteriores:

| Aspecto | Vanilla | SmolVLA-M | SmolVLA-D | **SmolVLA-MD** |
|---------|---------|-----------|-----------|----------------|
| n_obs_steps | 1 | 6 | 1 | **6** |
| Cámara wrist | estándar | estándar | estéreo 2560×800 | **estéreo 2560×800** |
| Input ViT | frame único | K=6 frames | mitad izda. estéreo | **mitad izda. × K=6** |
| Profundidad | ✗ | ✗ | ✓ PointCloudEncoder | **✓ PointCloudEncoder** |
| Params nuevos | 0 | 3 escalares | ~560 K | **~560 K + 3** |
| Dataset | cualquiera | cualquiera | cam2 2560×800 | **cam2 2560×800** |

---

## Paso 1 — Convertir SmolVLA base a SmolVLA-MD

```bash
python ./lerobot/scripts/convert_smolvla_to_smolvla_md.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_md_base
```

Con validación:

```bash
python ./lerobot/scripts/convert_smolvla_to_smolvla_md.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_md_base \
    --validate
```

La validación comprueba que:
- temporal_alpha = 0 (sin regresión inicial)
- DepthInjector zero_proj = 0 (sin efecto inicial)
- embed_image funciona para K=1 y K=6
- El checkpoint carga sin errores

---

## Paso 2 — Grabar episodios con cámara estéreo

SmolVLA-MD **requiere** que camera2 sea la cámara estéreo SVPRO (2560×800).
El script de grabación es idéntico al de SmolVLA-D:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/robots/so_follower" \
    --robot.cameras='{
        camera1: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"},
        camera2: {type: svpro, index_or_path: "/dev/video2", width: 2560, height: 800, fps: 30}
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_smolvla_md_v1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_smolvla_md_v1" \
    --dataset.num_episodes=60 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.num_image_writer_processes=2 \
    --dataset.push_to_hub=false
```

---

## Paso 3 — Entrenamiento Fase 1 (RGB + temporal, sin depth precomputado)

En la Fase 1 sólo se entrenan:
- temporal_alpha ×3 (del ViT temporal)
- state_proj, action_in/out_proj, action_time_mlp
- PointCloudEncoder y DepthInjector (no ven point clouds reales — depth_dropout=1 implícito por ausencia)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=outputs/smolvla_md_base \
    --dataset.repo_id=Esk1z0/tfm_smolvla_md_v1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_smolvla_md_v1" \
    --policy.n_obs_steps=6 \
    --policy.temporal_stride=1 \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --batch_size=8 \
    --steps=90000 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=10000 \
    --log_freq=50 \
    --output_dir=outputs/train/smolvla_md_phase1 \
    --job_name=tfm_smolvla_md_phase1 \
    --policy.device=cuda \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.push_to_hub=false \
    --policy.scheduler_decay_steps=86000 \
    --dataset.image_transforms.enable=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=tfm_smolvla_md
```

---

## Paso 4 — Entrenamiento Fase 2 (con point clouds precomputadas)

Precomputa las point clouds del dataset de entrenamiento (igual que en SmolVLA-D):

```bash
python lerobot/scripts/precompute_depth_features.py \
    --dataset_path /home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_smolvla_md_v1 \
    --calib_path camera/stereo_calibration_result.npz \
    --camera_key observation.images.camera2 \
    --n_points 1024
```

Luego entrena Fase 2 partiendo del checkpoint de Fase 1:

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=outputs/train/smolvla_md_phase1/checkpoints/last/pretrained_model \
    --dataset.repo_id=Esk1z0/tfm_smolvla_md_v1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_smolvla_md_v1" \
    --policy.n_obs_steps=6 \
    --policy.temporal_stride=1 \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.depth_dropout_prob=0.2 \
    --batch_size=8 \
    --steps=90000 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=10000 \
    --output_dir=outputs/train/smolvla_md_phase2 \
    --job_name=tfm_smolvla_md_phase2 \
    --policy.device=cuda \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=tfm_smolvla_md
```

---

## Paso 5 — Evaluación / Inferencia

```python
import numpy as np
from lerobot.policies.smolvla_md.modeling_smolvla_md import SmolVLAMDPolicy

policy = SmolVLAMDPolicy.from_pretrained("outputs/train/smolvla_md_phase2/checkpoints/last/pretrained_model")

# Inicializar profundidad estéreo (una vez)
policy.setup_stereo_depth(calib_path="camera/stereo_calibration_result.npz")
policy.config.stereo_camera_keys = ["observation.images.camera2"]

while True:
    obs = robot.capture()

    # Actualizar profundidad desde el frame estéreo crudo (BGR, H×2W)
    policy.update_stereo_frame(obs["camera2_bgr"])

    # Inferencia — usa depth cached + K=6 frames temporales
    action = policy.select_action(obs_batch)
    robot.apply(action)
```

La cámara estéreo proporciona ambas modalidades:
- **Lado izquierdo × K frames** → temporal ViT (memoria de contexto)
- **Frame completo actual** → SGBM+WLS → point cloud → DepthInjector

---

## Diagnóstico rápido

```python
from lerobot.policies.smolvla_md.modeling_smolvla_md import SmolVLAMDPolicy

p = SmolVLAMDPolicy.from_pretrained("outputs/smolvla_md_base")

# Verificar temporal_alpha
enc = p.model.vlm_with_expert.temporal_enc
for i, param in enumerate(enc.temporal_alpha.parameters()):
    print(f"temporal_alpha[{i}] = {param.item():.6f}, grad={param.requires_grad}")

# Verificar depth injector
for key, adapter in p.model.depth_injector.adapters.items():
    print(f"adapter[{key}].zero_proj max = {adapter.zero_proj.weight.abs().max().item():.2e}")

# Verificar trainable params
n_train = sum(p.numel() for p in p.parameters() if p.requires_grad)
n_total = sum(p.numel() for p in p.parameters())
print(f"Trainable: {n_train:,} / {n_total:,} ({100*n_train/n_total:.1f}%)")
```
