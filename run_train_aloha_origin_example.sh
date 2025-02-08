export MUJOCO_GL=egl
#conda activate lerobot

# act, vqbet, diffusion
# To train with an existing setup
 python lerobot/scripts/train.py \
     device=cuda \
     env=aloha \
     policy=act \
     training.num_workers=2 \
     training.eval_freq=10000 \
     training.log_freq=2500 \
     training.offline_steps=100000 \
     training.save_freq=25000 \
     eval.n_episodes=500 \
     eval.batch_size=4 \
     wandb.enable=false
#    # training.save_model=true \
#    # hydra.run.dir=outputs/train/act_aloha_sim_transfer_cube_human \
#    # env.task=AlohaTransferCube-v0 \
#    # dataset_repo_id=lerobot/aloha_sim_transfer_cube_human \
#    # policy.num_workers=2

# TO resume:
# python3 lerobot/scripts/train.py hydra.run.dir=outputs/train/2025-01-16/20-30-07_aloha_AlohaInsertion-v0_act_default/ resume=true
