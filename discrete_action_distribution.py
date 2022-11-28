import argparse
import os
from pathlib import Path

import gym
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from stable_baselines3 import PPO

def get_action_list(env, model):
    action_list = []
    obs = env.reset()
    for i in range(1000):
        action, _states = model.predict(obs, deterministic=True)
        action_list.append(action[0])
        obs, reward, done, info = env.step(action)
        if done:
          obs = env.reset()

    env.close()
    return action_list

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='action_distribution_log')
args = parser.parse_args()

env = gym.make("Pendulum-v1")

log_dir = Path(args.log_dir)
os.makedirs(str(log_dir), exist_ok=True)

model = PPO("MlpPolicy", env, verbose=1)
# 50000ステップを10回
# 学習ごとにactionの分布を保存
action_dist = {}
for ind in range(10):
    model.learn(total_timesteps=50000)
    action_list = get_action_list(env, model)
    action_dist[ind] = action_list

import pickle
with open(log_dir / 'action_dist.pkl', 'wb') as f:
    pickle.dump(action_dist, f)

