import gym
from stable_baselines3 import PPO
from trainDDPG import trained_model, env
from gym.wrappers import RecordVideo

env = RecordVideo(env, './video',
                  episode_trigger=lambda episode_id: True)

obs = env.reset()
done = False
total_reward = 0

while not done:
    action, _ = trained_model.predict(obs)
    obs, reward, done, _ = env.step(action)
    total_reward += reward

print(f"Total reward: {total_reward}")

env.close()
