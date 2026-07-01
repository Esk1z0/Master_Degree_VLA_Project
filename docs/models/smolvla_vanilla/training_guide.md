# SmolVLA-Vanilla — Guía de conversión y entrenamiento

> **Última actualización:** 2026-07-01

---

## Diferencias clave vs. las otras variantes

| Aspecto | **Vanilla** | SmolVLA-M | SmolVLA-D | SmolVLA-MD |
|---------|:-----------:|:---------:|:---------:|:----------:|
| n_obs_steps | 1 | 6 | 1 | 6 |
| Cámara wrist | estándar o estéreo (1 ojo usado) | estándar | estéreo 2560×800 | estéreo 2560×800 |
| Profundidad | ✗ | ✗ | ✓ PointCloudEncoder | ✓ PointCloudEncoder |
| Params nuevos | **0** | 3 escalares | ~560 K | ~560 K + 3 |
| Fases de entrenamiento | 1 | 1 | 2 (RGB, luego depth) | 2 (RGB+temporal, luego depth) |
| Dataset | cualquiera | cualquiera | cam2 2560×800 | cam2 2560×800 |

Vanilla es, con diferencia, la variante más simple de entrenar: es literalmente SmolVLA base con un `if` de
recorte estéreo opcional. No requiere script de precómputo, no tiene fase 2, y el checkpoint resultante es
compatible con las herramientas estándar de SmolVLA.

---

## Paso 1 — Convertir SmolVLA base a SmolVLA-Vanilla

```bash
python lerobot/scripts/convert_smolvla_to_smolvla_vanilla.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_vanilla_base
```

Con validación (comprueba que el checkpoint convertido carga y que `stereo_camera_keys` está presente):

```bash
python lerobot/scripts/convert_smolvla_to_smolvla_vanilla.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_vanilla_base \
    --validate
```

Este script **no toca los pesos** — solo cambia `type` a `smolvla_vanilla` en `config.json` y añade el
campo `stereo_camera_keys: []`. Al no añadirse ni quitarse parámetros, la carga es `strict=True`.

---

## Paso 2 — Grabar episodios (cámara estéreo opcional)

SmolVLA-Vanilla funciona con cualquier cámara estándar. Si se quiere usar como baseline comparable con
SmolVLA-D/MD, se graba con la misma cámara estéreo SVPRO en `camera2` (2560×800), y el modelo simplemente
usará un solo ojo:

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{
        camera1: {type: opencv, index_or_path: "/dev/video2", width: 640,  height: 480, fps: 30, fourcc: "MJPG"},
        camera2: {type: opencv, index_or_path: "/dev/video0", width: 2560, height: 800, fps: 15, fourcc: "MJPG"}
    }' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_final_dataset_120_eps \
    --dataset.num_episodes=120 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.push_to_hub=false
```

---

## Paso 3 — Entrenamiento (fase única)

```bash
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=outputs/smolvla_vanilla_base \
    --dataset.repo_id=Esk1z0/tfm_final_dataset_120_eps \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_final_dataset_120_eps" \
    --batch_size=16 \
    --steps=90000 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --log_freq=50 \
    --output_dir=outputs/train/smolvla_vanilla_v1 \
    --job_name=smolvla_vanilla_v1 \
    --policy.device=cuda \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.push_to_hub=false \
    --policy.scheduler_decay_steps=86000 \
    --dataset.image_transforms.enable=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=tfm_final_models
```

Si no se usa cámara estéreo, simplemente se omite `--policy.stereo_camera_keys` (o se pasa `'[]'`) y el
entrenamiento es idéntico al de SmolVLA base estándar.

---

## Paso 4 — Smoke test / evaluación

```bash
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video2", width: 640,  height: 480, fps: 30, fourcc: "MJPG"},camera2: {type: opencv, index_or_path: "/dev/video0", width: 2560, height: 800, fps: 15, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=false \
    --dataset.repo_id=Esk1z0/eval_smoketest_smolvla_vanilla_v1 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/smolvla_vanilla_v1/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.compile_model=true
```

---

## Diagnóstico rápido

```python
from lerobot.policies.smolvla_vanilla.modeling_smolvla_vanilla import SmolVLAVanillaPolicy

p = SmolVLAVanillaPolicy.from_pretrained("outputs/smolvla_vanilla_base")

print("stereo_camera_keys:", p.config.stereo_camera_keys)

# Confirmar que no hay parámetros nuevos respecto a un SmolVLA base equivalente
n_total = sum(param.numel() for param in p.parameters())
n_train = sum(param.numel() for param in p.parameters() if param.requires_grad)
print(f"Total: {n_total:,} | Trainable: {n_train:,} ({100 * n_train / n_total:.1f}%)")
```

Si `stereo_camera_keys` está vacío o la clave de cámara no aparece en el batch, `prepare_images` no realiza
ningún recorte y el forward es idéntico al de `smolvla_base`.
