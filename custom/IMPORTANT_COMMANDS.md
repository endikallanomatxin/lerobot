# Important commands for running bimanual SO101 experiments

```bash
export FOLLOWER_LEFT_PORT="/dev/ttyACM0"
export FOLLOWER_RIGHT_PORT="/dev/ttyACM3"
export LEADER_LEFT_PORT="/dev/ttyACM1"
export LEADER_RIGHT_PORT="/dev/ttyACM2"
```

```bash
lerobot-calibrate \
  --robot.type=bi_so101_follower \
  --robot.left_arm_port=$FOLLOWER_LEFT_PORT \
  --robot.right_arm_port=$FOLLOWER_RIGHT_PORT \
  --robot.id=bimanual_follower
```

```bash
lerobot-calibrate \
  --teleop.type=bi_so101_leader \
  --teleop.left_arm_port=$LEADER_LEFT_PORT \
  --teleop.right_arm_port=$LEADER_RIGHT_PORT \
  --teleop.id=bimanual_leader
```

```bash
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
  --dataset.repo_id=local/bimanual-so101-demo-27 \
  --dataset.num_episodes=2 \
  --dataset.episode_time_s=25 \
  --dataset.reset_time_s=10 \
  --dataset.single_task="Tarea bimanual X" \
  --dataset.push_to_hub=False
```

```bash
python -m lerobot.rl_custom.train_genesis --device cuda --batch_size 16 --max_steps 300 --steps 400
```

