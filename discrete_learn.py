import argparse
import os
from pathlib import Path

import gym
from gym.envs import register
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
env = gym.make('DiscretePendulum-v1')

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

