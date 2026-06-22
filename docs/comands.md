## Puertos
- leader /dev/ttyACM0
- follower /dev/ttyACM1
- camara /dev/video0  (WxH -> 640x480 30fps)
- camara /dev/video2  (WxH -> 640x480 30fps)

## Comandos
### Teleoperar
#### Sin camaras
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm
#### Con camaras
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --robot.cameras="{ images.top: {type: opencv, index_or_path: 0, width: 640, height: 480, fps: 30}, \
    images.wrist.left: {type: opencv, index_or_path: 2, width: 640, height: 240, fps: 25} }" \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm \
    --display_data=true


### Calibrado
#### Brazo Lider
lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \ 
    --teleop.id=my_awesome_leader_arm 
#### Brazo Follower
lerobot-calibrate \
    --robot.type=so101_leader \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_leader_arm

## Encontrar puerto de un brazo
lerobot-find-port

## Linux
#### Detectar camaras
v4l2-ctl --list-devices

#### ver fps y dimensiones
v4l2-ctl --device=/dev/video0 --list-formats-ext



# pruebas del pipeline
## Pruebas de usar el modelo completo
lerobot-record \
  --robot.type=so101_follower \
  --robot.port=/dev/ttyACM0 \
  --robot.id=so101_follower \
  --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video2", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video5", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
  --dataset.single_task="Move the arm slightly to the front and open the claw a little" \
  --dataset.repo_id=local/eval_smolvla_so101_smoke_test8 \
  --dataset.episode_time_s=5 \
  --dataset.num_episodes=1 \
  --policy.path=lerobot/smolvla_base \
  --policy.device=cuda \
  --dataset.push_to_hub=false
## Encontrar camaras
lerobot-find-cameras opencv

## PRuebas de entrenar el modelo y el pipeline y demas
lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video2", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --display_data=true



lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader


lerobot-calibrate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower



lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video2", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --display_data=true \
    --dataset.repo_id=Esk1z0/so101_cubes_stars_mockup_try2_3eps \
    --dataset.num_episodes=3 \
    --dataset.single_task="Stack the cubes and put the stars in the bin." \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2
    --push_to_hub=true

lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/so101_cubes_stars_mockup_try2_3eps \
    --batch_size=4 \
    --steps=500 \
    --output_dir=outputs/train/smolvla_so101_cubes_stars_pilot \
    --job_name=smolvla_so101_cubes_stars_pilot \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'


lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video2", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=true \
    --dataset.repo_id=${HF_USER}/so101_cubes_stars_pilot_eval \
    --dataset.num_episodes=1 \
    --dataset.single_task="Stack the cubes and put the stars in the bin." \
    --policy.path=outputs/train/smolvla_so101_cubes_stars_pilot/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --dataset.rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'


lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video2", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=true \
    --dataset.repo_id=Esk1z0/so101_cubes_stars_pilot_policy_eval \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Stack the cubes and put the stars in the bin." \
    --dataset.push_to_hub=false \
    --policy.path=outputs/train/smolvla_so101_cubes_stars_pilot/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --dataset.rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

last try

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video2", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=true \
    --dataset.repo_id=Esk1z0/eval_so101_cubes_stars_pilot_policy_v2 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Stack the cubes and put the stars in the bin." \
    --dataset.push_to_hub=false \
    --policy.path=outputs/train/smolvla_so101_cubes_stars_pilot/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'


laslast try

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video2", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPso101_cubes_stars_mockup_try2_3epsG"}}' \
    --display_data=true \
    --dataset.repo_id=Esk1z0/eval_so101_cubes_stars_pilot_policy_v3 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=20 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Stack the cubes and put the stars in the bin." \
    --dataset.push_to_hub=false \
    --policy.path=outputs/train/smolvla_so101_cubes_stars_pilot/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1




hf auth whoami

  A new version of huggingface_hub is available: 1.9.0 → 1.16.1








# prueba pick and place

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=tfm_so101_leader \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockuptry \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.push_to_hub=false

## purbea con codec distinto

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=tfm_so101_leader \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockuptry \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=2 \
    --dataset.vcodec=h264_nvenc \
    --dataset.push_to_hub=false

## prueba tres, ahora probando la codificacione a posteriori

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM1 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \
    --teleop.id=tfm_so101_leader \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockuptry \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.num_image_writer_processes=2 \
    --dataset.push_to_hub=false


## prueba a apagar y luego a reanudar

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockup_try_resume \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.num_image_writer_processes=2 \
    --dataset.push_to_hub=false

lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockup_try_resume \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_mockup_try_resume" \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.num_image_writer_processes=2 \
    --dataset.push_to_hub=false \
    --resume=true




# este ha ido bien
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/robots/so_follower" \
    --robot.cameras='{top: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, wrist: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=tfm_so101_leader \
    --teleop.calibration_dir="/home/juanes/.cache/huggingface/lerobot/calibration/teleoperators/so_leader" \
    --display_data=true \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockup_try_resume \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_mockup_try_resume" \
    --dataset.num_episodes=3 \
    --dataset.episode_time_s=120 \
    --dataset.reset_time_s=30 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.streaming_encoding=false \
    --dataset.num_image_writer_processes=2 \
    --dataset.push_to_hub=false \
    --resume=true



## prueba a entrenar co los datos parados y resumidos
# funciona
lerobot-train \
    --policy.path=lerobot/smolvla_base \
    --dataset.repo_id=Esk1z0/tfm_layer_ablation_mockup_try_resume \
    --dataset.root="/home/juanes/.cache/huggingface/lerobot/Esk1z0/tfm_layer_ablation_mockup_try_resume" \
    --batch_size=4 \
    --steps=500 \
    --output_dir=outputs/train/smolvla_so101_ablation_mockup_try_resume \
    --job_name=smolvla_so101_cubes_stars_pilot \
    --policy.device=cuda \
    --policy.empty_cameras=1 \
    --policy.push_to_hub=false \
    --wandb.enable=false \
    --rename_map='{"observation.images.top": "observation.images.camera1", "observation.images.wrist": "observation.images.camera2"}'

## prueba a evaluacion con el modelo entrenado

## ha ido meh por problemas
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=true \
    --dataset.repo_id=Esk1z0/eval_smolvla_so101_ablation_mockup_try_resume_v1 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --policy.path=outputs/train/smolvla_so101_ablation_mockup_try_resume/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1



nope
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=true \
    --dataset.repo_id=Esk1z0/eval_smolvla_so101_ablation_mockup_try_resume_v3 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.num_image_writer_processes=2 \
    --dataset.push_to_hub=false \
    --policy.path=outputs/train/smolvla_so101_ablation_mockup_try_resume/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.compile_model=true


meh pero a 18 hz, el mejor de hecho, veremos que pasa
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=true \
    --dataset.repo_id=Esk1z0/eval_smolvla_so101_ablation_mockup_try_resume_v1 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.num_image_writer_processes=2 \
    --dataset.push_to_hub=false \
    --policy.path=outputs/train/smolvla_so101_ablation_mockup_try_resume/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.compile_model=true

# ha bajado a 9 hz otra vez
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=true \
    --dataset.repo_id=Esk1z0/eval_smolvla_so101_ablation_mockup_try_resume_v3 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.num_image_writer_processes=4 \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=hevc_nvenc \
    --dataset.push_to_hub=false \
    --policy.path=outputs/train/smolvla_so101_ablation_mockup_try_resume/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.compile_model=false




lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 1280, height: 720, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=true \
    --dataset.repo_id=Esk1z0/eval_smolvla_so101_ablation_mockup_try_resume_v4 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.num_image_writer_processes=4 \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=auto \
    --dataset.push_to_hub=false \
    --policy.path=outputs/train/smolvla_so101_ablation_mockup_try_resume/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.compile_model=false





OJO QUE HA SACADO PRACTICAMENTE 20 HZ
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=false \
    --dataset.repo_id=Esk1z0/eval_smolvla_so101_ablation_mockup_try_resume_v6 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/smolvla_so101_ablation_mockup_try_resume/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.compile_model=true


## ultima prueba con precision variable

ha sacado unos 18 hz mas o menos
lerobot-record \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=tfm_so101_follower \
    --robot.cameras='{camera1: {type: opencv, index_or_path: "/dev/video0", width: 640, height: 480, fps: 30, fourcc: "MJPG"}, camera2: {type: opencv, index_or_path: "/dev/video2", width: 640, height: 480, fps: 30, fourcc: "MJPG"}}' \
    --display_data=false \
    --dataset.repo_id=Esk1z0/eval_smolvla_so101_ablation_mockup_try_resume_v8 \
    --dataset.num_episodes=1 \
    --dataset.episode_time_s=40 \
    --dataset.reset_time_s=10 \
    --dataset.single_task="Put the stars in the bin and the cubes on the marked area." \
    --dataset.push_to_hub=false \
    --dataset.streaming_encoding=true \
    --dataset.encoder_threads=4 \
    --dataset.vcodec=h264 \
    --policy.path=outputs/train/smolvla_so101_ablation_mockup_try_resume/checkpoints/last/pretrained_model \
    --policy.empty_cameras=1 \
    --policy.use_amp=true \
    --policy.compile_model=true