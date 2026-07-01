# Entrenamiento
## Estudio Capas
### Primera tanda
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=tfm_so101_follower \
    --robot.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/robots/so_follower" \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_batch_1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_batch_1" \
    --dataset.num_episodes=60 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.num_image_writer_processes=2 \
    --dataset.push_to_hub=false \
    #--resume=true

### Segunda tanda
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/robots/so_follower" \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_batch_2 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_batch_2" \
    --dataset.num_episodes=60 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.num_image_writer_processes=2 \
    --dataset.push_to_hub=false \
    #--resume=true

## Prueba Entrenar
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_batch_1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_batch_1" \
    --batch_size=4 \
    --steps=500 \
    --output_dir=outputs/train/config_A_baseline_mockup \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

## Prueba entrenar mejor, ahora con monitorizacion
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_batch_1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_batch_1" \
    --batch_size=8 \
    --steps=50000 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=2000 \
    --log_freq=50 \
    --output_dir=outputs/train/tfm_layer_ablation_expert_only_1h \
    --job_name=tfm_layer_ablation_expert_only_1h \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.push_to_hub=false \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=tfm_smolvla_ablation \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

- Problemas con el decay que estaba muy mal configurado, probaremos despues

## Prueba entrenar mejor, monitorizacion y decay mejorado, ademas de aumentacion

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_batch_1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_batch_1" \
    --batch_size=16 \
    --steps=50000 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=2000 \
    --log_freq=50 \
    --output_dir=outputs/train/tfm_layer_ablation_expert_only_v2 \
    --job_name=tfm_layer_ablation_expert_only_v2 \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.push_to_hub=false \
    --policy.scheduler_decay_steps=48000 \
    --dataset.image_transforms.enable=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=tfm_smolvla_ablation \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

## Prueba entrenar con 120 episodios en lugar de 60
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_batch_1_and_2 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_batch_1_and_2" \
    --batch_size=16 \
    --steps=90000 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=30000 \
    --log_freq=50 \
    --output_dir=outputs/train/tfm_layer_ablation_expert_only_v3 \
    --job_name=tfm_layer_ablation_expert_only_v3 \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.push_to_hub=false \
    --policy.scheduler_decay_steps=86000 \
    --dataset.image_transforms.enable=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=tfm_smolvla_ablation \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

# Evaluaciones
## Prueba Evaluar modelo tfm_layer_ablation_expert_only_v2 entrenado
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=false \
    --dataset.repo_id=Esk1z0/eval_tfm_layer_ablation_expert_only_v2_try_1 \
    --dataset.num_episodes=20 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/tfm_layer_ablation_expert_only_v2/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.compile_model=true

## Prueba Evaluar modelo tfm_layer_ablation_expert_only_v3 entrenado
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=false \
    --dataset.repo_id=Esk1z0/eval_tfm_layer_ablation_expert_only_v3_try_1 \
    --dataset.num_episodes=15 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/tfm_layer_ablation_expert_only_v3/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.compile_model=true

## Replay de episodios
lerobot-dataset-viz --repo-id /home/juanes/.cache/huggingface/lerobot/Esk1z0/eval_tfm_layer_ablation_expert_only_v3_try_1 --episode 13

## prueba con la ablacion de capas
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=false \
    --dataset.repo_id=Esk1z0/eval_tfm_layer_ablation_expert_only_v3_try_1 \
    --dataset.num_episodes=15 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/tfm_layer_ablation_expert_only_v3/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.compile_model=true

## Reajuste del modelo
python ./ablation/prepare_layercut_checkpoint.py \
    --source outputs/train/tfm_layer_ablation_expert_only_v3/checkpoints/last/pretrained_model \
    --output outputs/eval/layercut_3 \
    --ablate_layer_indices 3 

## Prueba con ablacion de capas
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=false \
    --dataset.repo_id=Esk1z0/eval_tfm_layercut_range_9_10_11 \
    --dataset.num_episodes=15 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/eval/layercut_range_9_10_11 \
    --policy.empty_cameras=1 \
    --policy.compile_model=true


# Datasets
## Juntar dos datasets para formar uno solo para las capas
lerobot-edit-dataset \
    --new_repo_id Esk1z0/tfm_layer_ablation_batch_1_and_2 \
    --operation.type merge \
    --operation.repo_ids "['Esk1z0/tfm_layer_ablation_batch_1', 'Esk1z0/tfm_layer_ablation_batch_2']"

# Evaluaciones
## calcula los resultados de las evaluaciones
python ./ablation/evaluations/calculate_evaluation_scores.py 

# wandb
## programacion de sincronizaciones
veces=96          # Cuántas veces quieres que se ejecute
espera="10m"      # Tiempo entre ejecuciones (puedes usar "30m", "1h", "2h", etc.)

for ((i=1; i<=veces; i++)); do
    echo "=== Ejecución $i de $veces - $(date) ==="
    .lerobot/bin/wandb sync outputs/train/smolvla_vanilla_v1/wandb/latest-run
    
    if [ $i -lt $veces ]; then
        echo "Esperando $espera antes de la siguiente sincronización..."
        sleep $espera
        
    fi
done
echo "=== Tarea completada ==="


# Rutas importantes
## Follower
/home/juanes/.cache/huggingface/lerobot/calibration/robots/so_follower/tfm_so101_follower.json
## Leader
/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader/tfm_so101_leader.json


# Nuevos modelos
## convertir smolvla en smolvla-md
python ./lerobot/scripts/convert_smolvla_to_smolvla_md.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_md_base
## Prueba copia SmolVLA entrenando pocos episodios, ahora como SmolVLA-MD
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=outputs/smolvla_md_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_batch_1_and_2 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_batch_1_and_2" \
    --batch_size=16 \
    --steps=600 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=300 \
    --log_freq=50 \
    --output_dir=outputs/train/smolvla_md \
    --job_name=tfm_smolvla_md_v1 \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.push_to_hub=false \
    --policy.scheduler_decay_steps=86000 \
    --dataset.image_transforms.enable=true \
    --wandb.enable=true \
    --wandb.mode=offline \
    --wandb.project=tfm_smolvla_md_v1 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

## Prueba SmolVLA-MD entrenado, se prueba la inferencia
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=false \
    --dataset.repo_id=Esk1z0/eval_smolvla_md_v1 \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=60 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/smolvla_md/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.compile_model=true

# Dataset final con cámara estéreo
## Grabación dataset final (60 eps, cámara top monocular + muñeca estéreo SVPRO)
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/robots/so_follower" \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video3", width: 2560, height: 800, fps: 25, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_final_dataset_60_eps_v1 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_final_dataset_60_eps_v1" \
    --dataset.fps=25 \
    --dataset.num_episodes=60 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.num_image_writer_processes=4 \
    --dataset.push_to_hub=false \
    #--resume=true

## Grabación segundo dataset final (60 eps, cámara top monocular + muñeca estéreo SVPRO)
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/robots/so_follower" \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 2560, height: 800, fps: 25, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_final_dataset_60_eps_v2 \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_final_dataset_60_eps_v2" \
    --dataset.fps=25 \
    --dataset.num_episodes=60 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.num_image_writer_processes=4 \
    --dataset.push_to_hub=false \
    #--resume=true


# Entrenamiento final de modelos

## Preparación de datasets

### Merge v1 + v2 → dataset completo 120 episodios
lerobot-edit-dataset \
    --new_repo_id Esk1z0/tfm_final_dataset_120_eps \
    --operation.type merge \
    --operation.repo_ids "['Esk1z0/tfm_final_dataset_60_eps_v1', 'Esk1z0/tfm_final_dataset_60_eps_v2']"

### Copia del dataset para añadir depth (modelos D y MD) — no tocar el original
cp -r /home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_final_dataset_120_eps \
      /home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_final_dataset_120_eps_depth

### TODO: precomputar point clouds sobre la copia depth
# (requiere crear lerobot/scripts/precompute_depth_features.py)
# python lerobot/scripts/precompute_depth_features.py \
#     --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_final_dataset_120_eps_depth" \
#     --calib_path="camera/stereo_calibration_result.npz"


## SmolVLA-Vanilla (baseline estéreo)

### Conversión de pesos
python lerobot/scripts/convert_smolvla_to_smolvla_vanilla.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_vanilla_base

### Entrenamiento
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


# SMOKE TESTS
## SMOVLA-vanilla


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

## SMOVLA-M
### Training smoke
python lerobot/scripts/convert_smolvla_to_smolvla_m.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_m_base

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=outputs/smolvla_m_base \
    --dataset.repo_id=Esk1z0/tfm_final_dataset_120_eps \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_final_dataset_120_eps" \
    --batch_size=4 \
    --steps=10 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=5 \
    --output_dir=outputs/train/smoke/smolvla_m \
    --policy.device=cuda \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.n_obs_steps=6 \
    --policy.push_to_hub=false \
    --policy.temporal_stride=1 \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.scheduler_decay_steps=10 \
    --dataset.image_transforms.enable=false \
    --wandb.enable=false
### Inferencia smoke
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
    --dataset.repo_id=Esk1z0/eval_smoketest_smolvla_m_v1 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/smoke/smolvla_m/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.compile_model=true

## SMOLVLA-D
### Training smoke
python lerobot/scripts/convert_smolvla_to_smolvla_d.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_d_base

PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=outputs/smolvla_d_base \
    --dataset.repo_id=Esk1z0/tfm_final_dataset_120_eps_depth \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_final_dataset_120_eps_depth" \
    --batch_size=4 \
    --steps=10 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=5 \
    --output_dir=outputs/train/smoke/smolvla_d \
    --policy.device=cuda \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.push_to_hub=false \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.scheduler_decay_steps=10 \
    --dataset.image_transforms.enable=false \
    --wandb.enable=false
### Inferencia smoke
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
    --dataset.repo_id=Esk1z0/eval_smoketest_smolvla_d_v1 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/smoke/smolvla_d/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.compile_model=true

## SMOLVLA-MD
### Conversión de pesos
python lerobot/scripts/convert_smolvla_to_smolvla_md.py \
    --src lerobot/smolvla_base \
    --dst outputs/smolvla_md_base

### Training smoke
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=outputs/smolvla_md_base \
    --dataset.repo_id=Esk1z0/tfm_final_dataset_120_eps_depth \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_final_dataset_120_eps_depth" \
    --batch_size=4 \
    --steps=10 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=5 \
    --output_dir=outputs/train/smoke/smolvla_md \
    --policy.device=cuda \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.n_obs_steps=6 \
    --policy.push_to_hub=false \
    --policy.temporal_stride=1 \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.scheduler_decay_steps=10 \
    --dataset.image_transforms.enable=false \
    --wandb.enable=false
### Inferencia smoke
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
    --dataset.repo_id=Esk1z0/eval_smoketest_smolvla_md_v1 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/smoke/smolvla_md/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.compile_model=true


## SmolVLA-D (depth + estereo, sin temporal)

### Entrenamiento final
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
lerobot-train \
    --policy.path=outputs/smolvla_d_base \
    --dataset.repo_id=Esk1z0/tfm_final_dataset_120_eps_depth \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_final_dataset_120_eps_depth" \
    --batch_size=16 \
    --steps=90000 \
    --seed=42 \
    --save_checkpoint=true \
    --save_freq=5000 \
    --log_freq=50 \
    --output_dir=outputs/train/smoke/smolvla_d_v1 \
    --job_name=smolvla_d_v1 \
    --policy.device=cuda \
    --policy.train_expert_only=true \
    --policy.freeze_vision_encoder=true \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.push_to_hub=false \
    --policy.scheduler_decay_steps= \
    --dataset.image_transforms.enable=true \
    --wandb.enable=false \


### Evaluación final con robot
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"},camera2: {type: opencv, index_or_path: "/dev/video0", width: 2560, height: 800, fps: 15, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=false \
    --dataset.repo_id=Esk1z0/eval_smolvla_d_v1 \
    --dataset.num_episodes=10 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and put the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=false \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/smolvla_d_v1/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.stereo_camera_keys='["observation.images.camera2"]' \
    --policy.compile_model=true