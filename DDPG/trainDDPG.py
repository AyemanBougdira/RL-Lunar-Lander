import gym
import torch
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
import numpy as np

# Check and print CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Create the environment
env = gym.make('LunarLanderContinuous-v2', render_mode='human')
env = DummyVecEnv([lambda: env])

# Define the device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create action noise (important for exploration in DDPG)
n_actions = env.action_space.shape[-1]
action_noise = NormalActionNoise(mean=np.zeros(
    n_actions), sigma=0.1 * np.ones(n_actions))

# Create DDPG model with default hyperparameters
model = DDPG(
    'MlpPolicy',
    env,
    action_noise=action_noise,
    verbose=1,
    device=device,

    # Key Hyperparameters
    learning_rate=1e-3,  # Learning rate for both actor and critic
    buffer_size=100000,  # Replay buffer size
    learning_starts=1000,  # Steps before starting learning
    batch_size=64,  # Minibatch size for training
    tau=0.005,  # Soft update coefficient for target networks
    gamma=0.99,  # Discount factor
    train_freq=1,  # Training frequency
    gradient_steps=1,  # Gradient steps per training

    # Network architecture parameters
    policy_kwargs=dict(
        # Neural network architecture
        net_arch=dict(pi=[400, 300], qf=[400, 300])
    )
)

# Train the model
model.learn(total_timesteps=50000)

# Save the trained model
model.save('lunar_lander_modelDDPG_cuda')

# Load the model
trained_model = DDPG.load('lunar_lander_modelDDPG_cuda', device=device)

# Evaluation function


def trained_DDPG():
    return trained_model



