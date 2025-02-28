export MUJOCO_GL=egl
#conda activate lerobot

POLICY=diffusion
POLICY=act
NUM_SAMPLES=20
OBS_TYPE=pixels_agent_pos_obj_doubleview

# specify DATA_DIR to avoid downloading the dataset
# policy=act \
# policy=vqbet \
# policy=act_pick_and_place \
DATA_DIR=data python lerobot/scripts/train.py \
    device=cuda \
    env=aloha_gym_local \
    policy=${POLICY}_pick_and_place \
    training.num_workers=4 \
    training.eval_freq=15000 \
    training.log_freq=2500 \
    training.offline_steps=400000 \
    training.save_freq=10000 \
    training.batch_size=8 \
    eval.n_episodes=2000 \
    eval.batch_size=4 \
    wandb.enable=false \
    dataset_repo_id=pick_and_place/sim_${POLICY}_${OBS_TYPE}_${NUM_SAMPLES} \
    # dataset_repo_id=user/sim_dataset
    # training.save_model=true \
    # hydra.run.dir=outputs/train/act_aloha_sim_transfer_cube_human \
    # env.task=AlohaTransferCube-v0 \
    # dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
    # policy.num_workers=2


# To train with an existing setup
# python lerobot/scripts/train.py \
#     device=cuda \
#     env=aloha \
#     policy=act \
#     training.num_workers=2 \
#     training.eval_freq=10000 \
#     training.log_freq=2500 \
#     training.offline_steps=100000 \
#     training.save_freq=25000 \
#     eval.n_episodes=500 \
#     eval.batch_size=4 \
#     wandb.enable=false
#    # training.save_model=true \
#    # hydra.run.dir=outputs/train/act_aloha_sim_transfer_cube_human \
#    # env.task=AlohaTransferCube-v0 \
#    # dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
#    # policy.num_workers=2

# resume
# DATA_DIR=data python3 lerobot/scripts/train.py hydra.run.dir=outputs/train/2025-01-19/22-07-20_aloha_AlohaPickAndPlace-v0_act_default/ resume=true
