import gym
import torch
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv

# Check and print CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Create the environment
env = gym.make('LunarLanderContinuous-v2', render_mode='human')
env = DummyVecEnv([lambda: env])

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create SAC model with comprehensive hyperparameters
model = SAC(
    'MlpPolicy',
    env,
    verbose=1,
    device=device,

    # Key Hyperparameters
    learning_rate=3e-4,  # Learning rate for all networks
    buffer_size=300000,  # Replay buffer size
    learning_starts=1000,  # Number of steps before starting learning
    batch_size=256,  # Minibatch size for training
    tau=0.005,  # Soft update coefficient for target networks
    gamma=0.99,  # Discount factor
    train_freq=1,  # Training frequency
    gradient_steps=1,  # Gradient steps per training

    # SAC-Specific Hyperparameters
    ent_coef='auto',  # Automatic entropy coefficient
    target_entropy='auto',  # Automatic target entropy

    # Network architecture parameters
    policy_kwargs=dict(
        net_arch=dict(
            pi=[256, 256],  # Policy network architecture
            qf=[256, 256]   # Q-function network architecture
        )
    )
)

# Train the model
model.learn(total_timesteps=50000)

# Save the trained model
model.save('lunar_lander_modelSAC_cuda')

# Load the model
trained_model = SAC.load('lunar_lander_modelSAC_cuda', device=device)

# Evaluation function


def trained_SAC():
    return trained_model
