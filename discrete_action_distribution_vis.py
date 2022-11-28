import argparse
import os
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

parser = argparse.ArgumentParser()
parser.add_argument('--log_dir', type=str, default='action_distribution_log')
args = parser.parse_args()

log_dir = Path(args.log_dir)
os.makedirs(str(log_dir), exist_ok=True)

import pickle
with open(log_dir / 'action_dist.pkl', 'rb') as f:
    action_dist = pickle.load(f)

# 横軸を学習の進み具合、縦軸をactionの分布にして描画する
num = 200
fig = make_subplots(rows=1, cols=10)
for ind, action_list in action_dist.items():
    action_list_init = action_list[0:0+num] + action_list[200:200+num] + action_list[400:400+num] + action_list[600:600+num] + action_list[800:800+num]
    fig.add_trace(
            go.Histogram(x=action_list_init,
                     xbins=dict(start=-2, end=2, size=0.5)),
        row=1, col=ind+1,
        )
fig.show()

