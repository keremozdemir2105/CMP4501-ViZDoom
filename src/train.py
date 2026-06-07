import os
import urllib.request
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.vec_env import VecFrameStack, SubprocVecEnv
from stable_baselines3.common.callbacks import CheckpointCallback
from src.env_wrapper import VizDoomGym
from src.model import CustomCNN
from src.config import Config

def download_scenario():
    base_url = f"https://raw.githubusercontent.com/Farama-Foundation/ViZDoom/master/scenarios/{Config.SCENARIO}"
    for ext in ["cfg", "wad"]:
        if not os.path.exists(f"{Config.SCENARIO}.{ext}"):
            print(f"Downloading {Config.SCENARIO}.{ext}...")
            urllib.request.urlretrieve(f"{base_url}.{ext}", f"{Config.SCENARIO}.{ext}")

def main():
    download_scenario()
    
    print(f"Initializing {Config.N_ENVS} parallel environments using Multiprocessing...")
    
    # ADDED SubprocVecEnv HERE to use multiple CPU cores!
    env = make_vec_env(lambda: VizDoomGym(render=False), n_envs=Config.N_ENVS, vec_env_cls=SubprocVecEnv)
    env = VecFrameStack(env, n_stack=Config.FRAME_STACK)
    
    policy_kwargs = dict(
        features_extractor_class=CustomCNN,
        features_extractor_kwargs=dict(features_dim=512)
    )
    
    model = PPO(
        "CnnPolicy", 
        env, 
        policy_kwargs=policy_kwargs,
        learning_rate=Config.LEARNING_RATE,
        n_steps=Config.N_STEPS,
        batch_size=Config.BATCH_SIZE,
        n_epochs=Config.N_EPOCHS,
        gamma=Config.GAMMA,
        tensorboard_log=Config.LOG_DIR,
        device=Config.DEVICE,
        verbose=1
    )
    
    checkpoint_callback = CheckpointCallback(
        save_freq=max(500_000 // Config.N_ENVS, 1),
        save_path=Config.MODEL_DIR,
        name_prefix="ppo_vizdoom"
    )
    
    print(f"Starting training on {Config.DEVICE}...")
    model.learn(total_timesteps=Config.TOTAL_TIMESTEPS, callback=checkpoint_callback, tb_log_name="PPO_Run")
    
    model.save(os.path.join(Config.MODEL_DIR, "ppo_vizdoom_final"))
    print("Training complete! Final model saved.")

if __name__ == "__main__":
    main()
