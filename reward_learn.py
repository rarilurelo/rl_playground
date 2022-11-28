import argparse
import os
from pathlib import Path

import gym
from gym.envs import register
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
env = gym.make('RewardPendulum-v1')

# envをMonitorでWrappして、log_dirに記録できるようにする
from stable_baselines3.common.monitor import Monitor
log_dir = Path(args.log_dir)
os.makedirs(str(log_dir), exist_ok=True)
env = Monitor(env, str(log_dir))

model = PPO("MlpPolicy", env, verbose=1)
# 学習ステップが足りないので増やす 10000 -> 500000
model.learn(total_timesteps=500000)

# 学習したモデルを保存する
model.save(log_dir / 'model')

# modelの評価を別スクリプト(base_evaluate.py)に移行する


