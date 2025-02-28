export MUJOCO_GL=egl
POLICY=diffusion
POLICY=act
NUM_SAMPLES=20
OBS_TYPE=pixels_agent_pos_obj_doubleview
# You may need to run this for a better encoder:
# conda install -c conda-forge ffmpeg
python lerobot/scripts/record_sim.py \
    fps=50 \
    num_samples=${NUM_SAMPLES} \
    data_file=combined_episodes.yaml \
    repo_id=pick_and_place/sim_${POLICY}_${OBS_TYPE}_${NUM_SAMPLES} \
    env.task=AlohaPickAndPlace-v0 \
    target_policy=${POLICY}
    # env.task=AlohaInsertion-v0

