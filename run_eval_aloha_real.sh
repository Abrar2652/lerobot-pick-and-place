export MUJOCO_GL=egl

python lerobot/scripts/control_robot.py record \
  --robot-path lerobot/configs/robot/aloha.yaml \
  --robot-overrides max_relative_target=null \
  --fps 30 \
  --root data \
  --repo-id outputs/eval_act_aloha_test \
  --tags aloha tutorial eval \
  --warmup-time-s 5 \
  --episode-time-s 40 \
  --reset-time-s 10 \
  --num-episodes 10 \
  --num-image-writer-processes 1 \
  -p outputs/train/2024-11-12/23-41-49_aloha_act_default/checkpoints/100000/pretrained_model
  # -p outputs/train/act_aloha_test/checkpoints/last/pretrained_model
