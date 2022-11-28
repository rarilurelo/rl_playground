import argparse
from pathlib import Path

import gym
from gym.envs import register
from gym.wrappers import RecordVideo
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='discrete_log')
parser.add_argument('--n', type=int, default=10)
args = parser.parse_args()

# 作成した環境をGymに登録する
register(
    id="DiscretePendulum-v1",
    entry_point="discrete_pendulum:DiscretePendulumEnv",
    max_episode_steps=200,
    kwargs=dict(n=args.n),
)

# DiscretePendulumEnvを使う
env = RecordVideo(gym.make("DiscretePendulum-v1"), Path(args.log_dir), lambda x: True)

model = PPO.load(Path(args.log_dir) / 'model')

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()

env.close()

