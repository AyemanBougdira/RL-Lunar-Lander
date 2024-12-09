import gym
from stable_baselines3 import PPO
from trainPPO import trained_model, env
from gym.wrappers import Monitor  # Importer Monitor pour l'enregistrement vidéo
# Enregistrer chaque épisode
env = Monitor(env, './video', force=True,
              video_callable=lambda episode_id: True)


# Load and evaluate the model
obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = trained_model.predict(obs)  # Prédire l'action
    obs, reward, done, _ = env.step(action)  # Appliquer l'action
    total_reward += reward  # Accumuler la récompense

print(f"Total reward: {total_reward}")

env.close()