ffmpeg -version

lerobot-calibrate \
    --robot.type=so101_leader \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_leader_arm


lerobot-calibrate \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM0 \ 
    --teleop.id=my_awesome_leader_arm 

leader /dev/ttyACM0

follower /dev/ttyACM1

lerobot-teleoperate \
    --robot.type=so101_follower \
    --robot.port=/dev/ttyACM0 \
    --robot.id=my_awesome_follower_arm \
    --teleop.type=so101_leader \
    --teleop.port=/dev/ttyACM1 \
    --teleop.id=my_awesome_leader_arm