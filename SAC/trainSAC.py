import os
import gym
import torch
import numpy as np
import imageio
from stable_baselines3 import SAC
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.evaluation import evaluate_policy


def record_video(model, env, num_episodes=3, video_dir='./video', fps=30):
    """
    Enregistre les vidéos de simulations pour un modèle entraîné.
    
    Args:
        model: Le modèle RL entraîné.
        env: L'environnement Gym.
        num_episodes: Nombre d'épisodes à enregistrer.
        video_dir: Répertoire pour enregistrer les vidéos.
        fps: Frames par seconde pour les vidéos.
    """
    os.makedirs(video_dir, exist_ok=True)

    for episode in range(num_episodes):
        obs = env.reset()[0]  # Compatible avec gym version récente
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

        video_path = os.path.join(video_dir, f'episode_{episode}.mp4')
        imageio.mimsave(video_path, frames, fps=fps)
        print(
            f"Episode {episode} enregistré : {video_path} - Récompense totale : {total_reward}")


def train_sac_model(env_id='LunarLanderContinuous-v2', total_timesteps=10000):
    """
    Entraîne un modèle SAC sur un environnement donné.

    Args:
        env_id: Identifiant de l'environnement Gym.
        total_timesteps: Nombre total de pas pour l'entraînement.

    Returns:
        Le modèle entraîné.
    """
    # Configuration du dispositif
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"CUDA disponible: {torch.cuda.is_available()}")

    # Création de l'environnement vectorisé
    env = DummyVecEnv([lambda: gym.make(env_id)])

    # Initialisation du modèle SAC
    model = SAC(
        policy='MlpPolicy',
        env=env,
        verbose=1,
        device=device,
        learning_rate=3e-4,
        buffer_size=1000000,
        learning_starts=10000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=1,
        gradient_steps=1,
        ent_coef='auto'
    )

    # Entraînement du modèle
    print("Début de l'entraînement...")
    model.learn(total_timesteps=total_timesteps)
    print("Entraînement terminé.")

    # Évaluation du modèle
    mean_reward, std_reward = evaluate_policy(model, env, n_eval_episodes=10)
    print(f"Récompense moyenne : {mean_reward:.2f} +/- {std_reward:.2f}")

    # Sauvegarde du modèle
    model_path = f"{env_id}_sac_model"
    model.save(model_path)
    print(f"Modèle sauvegardé à : {model_path}")

    return model


def main():
    """
    Point d'entrée principal pour entraîner, évaluer et enregistrer des vidéos du modèle.
    """
    # Entraînement du modèle
    trained_model = train_sac_model()

    # Création d'un nouvel environnement pour l'évaluation
    eval_env = gym.make('LunarLanderContinuous-v2', render_mode='rgb_array')

    # Enregistrement des vidéos
    print("Enregistrement des vidéos...")
    record_video(trained_model, eval_env)

    eval_env.close()
    print("Programme terminé.")


if __name__ == "__main__":
    main()
