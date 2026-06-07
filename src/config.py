import torch

class Config:
    # Environment Settings
    SCENARIO = "deadly_corridor"
    CONFIG_PATH = f"{SCENARIO}.cfg"
    IMAGE_SHAPE = (1, 84, 84)
    FRAME_STACK = 4
    
    # PPO Hyperparameters optimized for RTX 4090
    ALGORITHM = "PPO"
    LEARNING_RATE = 1e-4
    N_STEPS = 2048
    BATCH_SIZE = 256  
    N_EPOCHS = 10
    GAMMA = 0.99
    N_ENVS = 16       # Parallel environments to maximize CPU usage
    TOTAL_TIMESTEPS = 1_000_000 
    
    # Compute Device Setup
    DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    
    # File Paths
    LOG_DIR = "./logs/"
    MODEL_DIR = "./models/"
    VIDEO_DIR = "./videos/"
