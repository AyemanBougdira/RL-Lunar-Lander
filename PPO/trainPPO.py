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
model = PPO('MlpPolicy', env,
                verbose=1,  # Not a hyperparameter, just logging level
                device=device,
                # Default hyperparameters:
                learning_rate=3e-4,  # Default learning rate
                n_steps=2048,  # Number of steps to collect per update
                batch_size=64,  # Minibatch size for each gradient step
                n_epochs=10,  # Number of epoch when optimizing the surrogate loss
                gamma=0.99,  # Discount factor
                gae_lambda=0.95,  # Generalized Advantage Estimation lambda
                clip_range=0.2,  # Clipping parameter for PPO
                ent_coef=0.0,  # Entropy coefficient for exploration
                vf_coef=0.5,  # Value function coefficient
                max_grad_norm=0.5,  # Maximum gradient norm
                target_kl=None  # Target KL divergence threshold
                )


# Train the model
model.learn(total_timesteps=50000)



# Save the trained model
model.save('lunar_lander_modelPPO_cuda')

trained_model = PPO.load('lunar_lander_modelPPO_cuda', device=device)


def trained_PPO():
    return trained_model




env.close()

