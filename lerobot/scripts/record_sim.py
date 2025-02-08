"""
Script to create a dataset by replaying sequences of actions in simulation.
Each episode initializes the env with a given state and replays a sequence of actions.
"""

import numpy as np
import yaml
import time
from pathlib import Path
from typing import List, Dict
import hydra
from omegaconf import DictConfig
import torch
from tqdm import tqdm
from lerobot.common.datasets.populate_dataset import (
    init_dataset,
    add_frame,
    save_current_episode,
    create_lerobot_dataset,
)
from lerobot.common.envs.factory import make_env
from lerobot.common.utils.utils import init_logging, log_say
import matplotlib.pyplot as plt

def load_replay_data(yaml_path: str) -> List[Dict]:
    """
    Load replay data from YAML file.
    
    Expected YAML structure:
    config:
      ...
    episodes:
      - init_state:
          ...
        actions:
          ...
    """
    with open(yaml_path, 'r') as f:
        data = yaml.safe_load(f)
    
    # Convert action lists to numpy arrays
    episodes = data['episodes']
    for episode in episodes:
        episode['actions'] = np.array(episode['actions'])
    
    return episodes

def create_replay_dataset(
    env,
    target_policy: str,
    episodes: List[Dict],
    fps: int,
    root: str,
    repo_id: str,
    play_sounds: bool = True,
    num_image_writer_processes: int = 1,
    num_image_writer_threads: int = 1,
    do_render: bool = False,

):
    """
    Creates a dataset by replaying sequences of actions in simulation.
    
    Args:
        env: Gymnasium environment instance
        episodes: List of dictionaries containing init_state and actions for each episode
        fps: Frames per second for replay
        root: Root directory for dataset
        repo_id: Dataset repo ID
        play_sounds: Whether to play sounds during recording
        num_image_writer_processes: Number of processes for writing images
        num_image_writer_threads: Number of threads per process for writing images
    """
    # Initialize empty dataset
    dataset = init_dataset(
        repo_id=repo_id,
        root=root,
        force_override=True,
        fps=fps,
        video=True,
        write_images=True,
        num_image_writer_processes=num_image_writer_processes,
        num_image_writer_threads=num_image_writer_threads,
    )

    # Initialize a plot for displaying images
    plt.ion()  # Turn on interactive mode
    fig, ax = plt.subplots()

    # Record each episode
    for episode_idx, episode in enumerate(episodes):
        log_say(f"Recording episode {episode_idx}: {episode['name']} with {len(env.envs)} environments", play_sounds)
            
        # Reset environment and set initial state
        env.reset()
        for single_env in env.envs:
            single_env.set_state(episode['init_state'])
            
            # Initialize step counter
            step = 0

            # Record each step
            for action in tqdm(episode['actions']):
                # Step environment
                obs, reward, terminated, truncated, info = single_env.step(action)

                # Create observation dictionary with proper keys
                camera_name = 'top'
                # TODO: add other cameras and invoke env.render() with it.
                # TODO: use obs['observation.image.top']
                # TODO: set next.* properly
                if target_policy == 'act':
                    observation = {
                        'observation.state': torch.tensor(obs['agent_pos']) if isinstance(obs, dict) else torch.tensor(obs),
                        f'observation.images.{camera_name}': torch.tensor(single_env.render().copy()) if hasattr(single_env, 'render') else None,
                    }
                elif target_policy == 'diffusion':
                    image = torch.tensor(single_env.render().copy()) if hasattr(single_env, 'render') else None
                    # permute dimensions to (C, H, W)
                    image = image.permute(2, 0, 1)
                    # resize image to 96 x 96
                    image = torch.nn.functional.interpolate(image.unsqueeze(0), size=(96, 96), mode='bilinear').squeeze(0)
                    image = image.permute(1, 2, 0)
                    observation = {
                        #'observation.environment_state': torch.tensor(obs['env_state']) if isinstance(obs, dict) else torch.tensor(obs),
                        'observation.state': torch.tensor(obs['agent_pos']) if isinstance(obs, dict) else torch.tensor(obs),
                        'observation.image': image,
                        'next.reward': reward,
                        'next.success': info['is_success'],
                        #'donedone': terminated or truncated,
                    }
                else:
                    raise ValueError(f"Invalid target policy: {target_policy}")
                    
                # Create action dictionary
                action_dict = {'action': torch.tensor(action)}
                    
                # Add frame to dataset
                add_frame(dataset, observation, action_dict)
                # Render the image
                if do_render and hasattr(single_env, 'render'):
                    image = single_env.render().copy()
                    observation[f'observation.images.{camera_name}'] = torch.tensor(image)

                    # Display the image using matplotlib
                    ax.clear()
                    ax.imshow(image)
                    ax.set_title(f"Episode {episode_idx}, Step {step}")
                    plt.pause(0.02)  # Pause to update the plot
                    
                # Increment step counter
                step += 1

            save_current_episode(dataset)

    run_compute_stats = True
    push_to_hub = False
    tags = None
    play_sounds = True
    lerobot_dataset = create_lerobot_dataset(
        dataset, run_compute_stats, push_to_hub, tags, play_sounds,
        target_policy)
            
    return lerobot_dataset

@hydra.main(version_base=None, config_path="../configs", config_name="record_sim")
def main(cfg: DictConfig):
    """Main function for recording simulation data."""
    init_logging()
    
    # Load episodes from YAML
    episodes = load_replay_data(cfg.data_file)
    
    # Create environment with n_envs=1 for replay
    cfg.eval.n_envs = 1  # Force single environment
    cfg.eval.batch_size = 1  # Force single environment
    cfg.eval.use_async_envs = False  # Use sync environment
    env = make_env(cfg)
    
    # Create dataset
    dataset = create_replay_dataset(
        env=env,
        target_policy=cfg.target_policy,
        episodes=episodes,
        fps=cfg.fps,
        root=cfg.root,
        repo_id=cfg.repo_id,
        play_sounds=cfg.get('play_sounds', True),
        num_image_writer_processes=cfg.num_image_writer_processes,
        num_image_writer_threads=cfg.num_image_writer_threads
    )

if __name__ == "__main__":
    main()
