import gym
import torch
import numpy as np
import imageio
import os
from stable_baselines3 import DDPG
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


def record_video(model, env, num_episodes=3):
    os.makedirs('./video', exist_ok=True)

    for episode in range(num_episodes):
        obs = env.reset()[0]  # Compatible avec dernière version Gym
        done = False
        frames = []
        total_reward = 0

        while not done:
            action, _ = model.predict(obs)
            obs, reward, terminated, truncated, info = env.step(action)

            done = terminated or truncated
            frame = env.render()

            if frame is not None:
                frames.append(frame)
            total_reward += reward
        imageio.mimsave(f'./video/episode_{episode}.mp4', frames, fps=30)
        print(f"Episode {episode} - Total Reward: {total_reward}")


def train_ddpg_model():
    # Configuration
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA available: {torch.cuda.is_available()}")

    # Environnement
    env = gym.make('LunarLanderContinuous-v2')
    env = DummyVecEnv([lambda: env])

    # Bruit d'action
    n_actions = env.action_space.shape[-1]
    action_noise = NormalActionNoise(
        mean=np.zeros(n_actions),
        sigma=0.1 * np.ones(n_actions)
    )

    # Modèle DDPG
    model = DDPG(
        'MlpPolicy',
        env,
        action_noise=action_noise,
        verbose=1,
        device=device,
        learning_rate=1e-3,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        policy_kwargs=dict(net_arch=dict(pi=[400, 300], qf=[400, 300]))
    )

    # Entraînement
    model.learn(total_timesteps=10000)  # Augmenté

    # Évaluation
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Mean reward: {mean_reward:.2f} +/- {std_reward:.2f}")

    # Sauvegarde
    model.save('lunar_lander_modelDDPG_cuda')

    return model


def main():
    trained_model = train_ddpg_model()

    # Nouvel environnement pour l'évaluation
    eval_env = gym.make('LunarLanderContinuous-v2', render_mode='human')
    record_video(trained_model, eval_env)
    eval_env.close()


if __name__ == "__main__":
    main()
