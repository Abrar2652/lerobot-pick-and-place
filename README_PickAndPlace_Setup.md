# Train
- Make sure dataset is located in data/pick_and_place/sim_act

- Install conda environment
conda env create -f environment.yml

- Use this script. Set the policy to act.
run_train_aloha_pick_and_place.sh

You will see ouputs in outputs/train/<date>/, in which you will be able to find checkpoints and evaluation results (in the format of videos)
