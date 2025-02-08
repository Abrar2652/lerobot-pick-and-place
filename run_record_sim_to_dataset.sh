export MUJOCO_GL=egl
POLICY=diffusion
POLICY=act
# You may need to run this for a better encoder:
# conda install -c conda-forge ffmpeg
python lerobot/scripts/record_sim.py \
    fps=50 \
    data_file=combined_episodes.yaml \
    repo_id=pick_and_place/sim_${POLICY} \
    env.task=AlohaPickAndPlace-v0 \
    target_policy=${POLICY}
    # env.task=AlohaInsertion-v0

