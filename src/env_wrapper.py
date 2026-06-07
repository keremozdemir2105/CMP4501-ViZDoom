import os
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import cv2
from vizdoom import DoomGame

# Set a dummy video driver to prevent crashes on cloud servers without monitors
os.environ["SDL_VIDEODRIVER"] = "dummy"

class VizDoomGym(gym.Env):
    def __init__(self, render: bool = False, config_path: str = "deadly_corridor.cfg"):
        super().__init__()
        self.game = DoomGame()
        self.game.load_config(config_path)
        self.game.set_window_visible(render)
        self.game.set_sound_enabled(False)
        self.game.init()
        
        # Define action and observation spaces
        self.action_space = spaces.Discrete(self.game.get_available_buttons_size())
        self.observation_space = spaces.Box(
            low=0, high=255, shape=(1, 84, 84), dtype=np.uint8
        )
        
    def step(self, action: int):
        actions = np.identity(self.action_space.n, dtype=int)
        # Apply frame skipping: repeat the selected action for 4 frames
        reward = self.game.make_action(list(actions[action]), 4)
        done = self.game.is_episode_finished()
        
        if done:
            return np.zeros((1, 84, 84), dtype=np.uint8), reward, done, False, {}
        
        state = self.game.get_state()
        frame = self.process_frame(state.screen_buffer)
        
        # Extract health variable for evaluation metrics
        info = {"health": state.game_variables[0] if state.game_variables else 0}
        
        return frame, reward, done, False, info
        
    def reset(self, seed: int = None, options: dict = None):
        super().reset(seed=seed)
        self.game.new_episode()
        state = self.game.get_state()
        frame = self.process_frame(state.screen_buffer)
        return frame, {}
        
    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        # Convert from PyTorch format to OpenCV format
        frame = np.transpose(frame, (1, 2, 0))
        # Convert RGB to Grayscale if necessary
        if frame.shape[2] == 3:
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        # Resize to 84x84 and expand dimensions for the CNN
        frame = cv2.resize(frame, (84, 84), interpolation=cv2.INTER_AREA)
        return np.expand_dims(frame, axis=0)
        
    def close(self):
        self.game.close()
