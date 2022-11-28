import argparse
from pathlib import Path

import gym
from gym.envs import register
from gym.wrappers import RecordVideo
from stable_baselines3 import PPO

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='reward_log')
parser.add_argument('--thdot_coef', type=float, default=0.1)
parser.add_argument('--u_coef', type=float, default=0.001)
parser.add_argument('--angle_goal_diff', type=float, default=0)
args = parser.parse_args()

# 作成した環境をGymに登録する
register(
    id="RewardPendulum-v1",
    entry_point="reward_pendulum:RewardPendulumEnv",
    max_episode_steps=200,
    kwargs=dict(thdot_coef=args.thdot_coef, u_coef=args.u_coef, angle_goal_diff=args.angle_goal_diff),
)

# RewardPendulumEnvを使う
env = RecordVideo(gym.make("RewardPendulum-v1"), Path(args.log_dir), lambda x: True)

model = PPO.load(Path(args.log_dir) / 'model')

obs = env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = env.step(action)
    if done:
      obs = env.reset()

env.close()

