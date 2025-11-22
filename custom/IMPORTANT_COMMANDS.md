# Important commands for running bimanual SO101 experiments


## Configuración de los motores

```sh
lerobot-setup-motors \
  --teleop.type=so101_leader \
  --teleop.port=/dev/ttyACM3
```


## Calibración de los robots

```sh
export FOLLOWER_LEFT_PORT="/dev/ttyACM0"
export FOLLOWER_RIGHT_PORT="/dev/ttyACM3"
export LEADER_LEFT_PORT="/dev/ttyACM1"
export LEADER_RIGHT_PORT="/dev/ttyACM2"
```

```sh
lerobot-calibrate \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=$FOLLOWER_LEFT_PORT \
  --robot.right_arm_port=$FOLLOWER_RIGHT_PORT \
  --robot.id=bimanual_follower
```

```sh
lerobot-calibrate \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=$LEADER_LEFT_PORT \
  --teleop.right_arm_port=$LEADER_RIGHT_PORT \
  --teleop.id=bimanual_leader
```


## Grabación de dataset

```sh
lerobot-record \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=$FOLLOWER_LEFT_PORT \
  --robot.right_arm_port=$FOLLOWER_RIGHT_PORT \
  --robot.id=bimanual_follower \
  --robot.cameras="{
      left_wrist:  {type: opencv, index_or_path: /dev/video2, width: 640, height: 480, fps: 20},
      right_wrist: {type: opencv, index_or_path: /dev/video4, width: 640, height: 480, fps: 20},
      overhead:    {type: opencv, index_or_path: /dev/video6, width: 640, height: 480, fps: 30}
  }" \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=$LEADER_LEFT_PORT \
  --teleop.right_arm_port=$LEADER_RIGHT_PORT \
  --teleop.id=bimanual_leader \
  --display_data=true \
  --dataset.repo_id=local/3d_printed_pieces \
  --dataset.num_episodes=10 \
  --dataset.episode_time_s=60 \
  --dataset.reset_time_s=5 \
  --dataset.single_task="Pick up the orange 3D printed pieces and put the into their respective silhouttes" \
  --dataset.push_to_hub=False
```


## Entrenamiento supervisado

```sh
lerobot-train \
  --dataset.repo_id=local/bimanual-so101-demo-30 \
  --policy.type=act \
  --output_dir=outputs/train/act_bimanual-so101-demo-30 \
  --job_name=act_bimanual-so101-demo-30 \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --dataset.video_backend=pyav \
  --batch_size=4 \
  --num_workers=2 \
  --steps=20000 \
  --eval_freq=5000 \
  --save_freq=5000
```


## Entrenamiento por RL en entorno custom con Genesis

```sh
python -m lerobot.rl_custom.train_genesis \
    --device cuda \
    --batch_size 16 \
    --max_steps 300 \
    --steps 200 \
    --policy_path outputs/train/act_bimanual-so101-demo-30/checkpoints/002000/pretrained_model/
```

(Si omites la policy_path, se entrena desde cero)

