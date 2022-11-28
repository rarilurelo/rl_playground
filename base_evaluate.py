import argparse
from pathlib import Path

import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='base_log')
args = parser.parse_args()

env = RecordVideo(gym.make("Pendulum-v1"), Path(args.log_dir), lambda x: True)

model = PPO.load(Path(args.log_dir) / "model")

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()

env.close()
