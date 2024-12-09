import gym
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

# Check and print CUDA availability
print(f"CUDA available: {torch.cuda.is_available()}")

# Create the environment
env = gym.make('LunarLander-v2', render_mode='human')
env = DummyVecEnv([lambda: env])

# Create PPO model with CUDA
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = PPO('MlpPolicy', env, verbose=1, device=device)

# Train the model
model.learn(total_timesteps=50000)



# Save the trained model
model.save('lunar_lander_modelPPO_cuda')

trained_model = PPO.load('lunar_lander_modelPPO_cuda', device=device)


def trained_PPO():
    return trained_model




env.close()

