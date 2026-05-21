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


