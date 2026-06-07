import os
import glob
import numpy as np
import imageio
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack
from src.env_wrapper import VizDoomGym
from src.config import Config

def get_latest_checkpoint(model_dir: str) -> str:
    """Finds the most recent checkpoint in the models directory to act as 'half-trained'."""
    checkpoints = glob.glob(os.path.join(model_dir, "ppo_vizdoom_*_steps.zip"))
    if not checkpoints:
        return None
    # Sort by step count
    checkpoints.sort(key=lambda x: int(x.split('_')[-2]))
    # Pick one from the middle of the training process
    mid_index = len(checkpoints) // 2
    return checkpoints[mid_index]

def record_agent_video(model_path: str, env: VecFrameStack, video_path: str, num_steps: int = 300):
    """Runs the agent in the environment and captures RGB frames for the video."""
    print(f"Recording video: {video_path}...")
    
    if model_path is None:
        model = None  # Untrained / Random Agent
        print("Using random actions (Untrained).")
    else:
        model = PPO.load(model_path, env=env)
        print(f"Loaded model from {model_path}.")
        
    obs = env.reset()
    frames = []
    
    for _ in range(num_steps):
        # Access the raw ViZDoom environment to get the high-res colored screen buffer
        raw_env = env.envs[0].unwrapped.game
        state = raw_env.get_state()
        
        if state is not None and state.screen_buffer is not None:
            # ViZDoom outputs channels first (C, H, W). Convert to (H, W, C) for video encoding.
            frame = np.transpose(state.screen_buffer, (1, 2, 0))
            frames.append(frame)
            
        if model is None:
            # Sample random action
            action = [env.action_space.sample()]
        else:
            # Predict best action
            action, _ = model.predict(obs, deterministic=True)
            
        obs, _, done, _ = env.step(action)
        
        if done[0]:
            obs = env.reset()
            
    # Save the frames as an MP4 video
    imageio.mimsave(video_path, frames, fps=35, macro_block_size=None)
    print(f"Successfully saved {video_path}\n")

def main():
    # We only need 1 environment for evaluation and recording
    env = make_vec_env(lambda: VizDoomGym(render=False, config_path=Config.CONFIG_PATH), n_envs=1)
    env = VecFrameStack(env, n_stack=Config.FRAME_STACK)
    
    # Define paths for the three required stages
    untrained_video = os.path.join(Config.VIDEO_DIR, "1_untrained.mp4")
    half_trained_video = os.path.join(Config.VIDEO_DIR, "2_half_trained.mp4")
    fully_trained_video = os.path.join(Config.VIDEO_DIR, "3_fully_trained.mp4")
    
    final_model_path = os.path.join(Config.MODEL_DIR, "ppo_vizdoom_final.zip")
    half_trained_model_path = get_latest_checkpoint(Config.MODEL_DIR)
    
    # Generate exactly three stages as required by the assignment
    record_agent_video(None, env, untrained_video)
    
    if half_trained_model_path:
        record_agent_video(half_trained_model_path, env, half_trained_video)
    else:
        print("Warning: No checkpoints found for the half-trained video.")
        
    if os.path.exists(final_model_path):
        record_agent_video(final_model_path, env, fully_trained_video)
    else:
        print("Error: Final model not found. Did training complete?")

if __name__ == "__main__":
    main()
