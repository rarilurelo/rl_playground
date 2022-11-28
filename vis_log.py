import argparse

import numpy as np
import plotly.graph_objects as go
from stable_baselines3.common.results_plotter import load_results, ts2xy

def moving_average(values, window):
    """
    Smooth values by doing a moving average
    :param values: (numpy array)
    :param window: (int)
    :return: (numpy array)
    """
    weights = np.repeat(1.0, window) / window
    return np.convolve(values, weights, 'valid')


def get_go_results(log_folder, title='Learning Curve'):
    """
    plot the results

    :param log_folder: (str) the save location of the results to plot
    :param title: (str) the title of the task to plot
    """
    x, y = ts2xy(load_results(log_folder), 'timesteps')
    y = moving_average(y, window=50)
    # Truncate x
    x = x[len(x) - len(y):]

    data = go.Scatter(x=x, y=y, name=log_folder)
    return data

def show(go_data_list):
    layout = go.Layout(
            xaxis=dict(title='Number of Timesteps'),
            yaxis=dict(title='Rewards'),
            title='Learning Curve Smoothed',
            )
    fig = go.Figure(data=go_data_list, layout=layout)
    fig.show()


parser = argparse.ArgumentParser()
parser.add_argument('--log_dirs', type=str, nargs='*', default='base_log')
args = parser.parse_args()

go_data_list = []
for log_dir in args.log_dirs:
    go_data = get_go_results(log_dir)
    go_data_list.append(go_data)
show(go_data_list)
