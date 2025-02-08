export MUJOCO_GL=egl

## The environment is created in make_env
#python lerobot/scripts/eval.py \
#    -p outputs/train/2024-11-12/23-41-49_aloha_act_default/checkpoints/025000/pretrained_model \
#    eval.n_episodes=10
#    # -p outputs/train/2024-11-12/23-41-49_aloha_act_default/checkpoints/100000/pretrained_model \
#    # env=aloha \
#    # policy=act \


MODEL=outputs/train/2025-01-04/08-10-32_aloha_act_default/checkpoints/last/pretrained_model
# -p outputs/train/2024-11-28/10-18-30_aloha_act_default/checkpoints/last/pretrained_model \
python lerobot/scripts/eval.py \
    -p ${MODEL} \
    eval.n_episodes=4
