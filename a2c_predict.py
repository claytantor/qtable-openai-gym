
import numpy as np
import pandas as pd
import sys, os
import json
import gym
import gym_anytrading
import quantstats as qs
import optuna
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from datetime import datetime
from utils import load_dataset

from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

from dotenv import load_dotenv, dotenv_values
load_dotenv()  # take environment variables from .env.



def predict(config):

    #df = gym_anytrading.datasets.STOCKS_GOOGL.copy()
    COINS_USD_BTC = load_dataset('COINS_USD_BTC', 'Date')
    df = COINS_USD_BTC.copy()

    window_size = int(config['n_steps'])
    start_index = window_size
    end_index = len(df)

    print("=========PREDICT=============")

    env_maker = lambda: gym.make(
        'stocks-v0',
        df = df,
        window_size = window_size,
        frame_bound = (start_index, end_index)
    )

    env = env_maker()
    observation = env.reset()

    model = A2C("MlpPolicy", env).load('/workspace/models/a2c_btc_20210902_043521_0_12')

    reward_total = 0.0
    step_index = 1
    while True:
        observation = observation[np.newaxis, ...]

        # action = env.action_space.sample()
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        print(action, reward, info)
        
        reward_total += reward
        avg_reward = reward_total/step_index

        step_index += 1

        # env.render()
        if done:
            print("info:", info)
            print(f'total reward:{reward_total} average reward:{avg_reward}')
            break

    qs.extend_pandas()

    net_worth = pd.Series(env.history['total_profit'], index=df.index[start_index+1:end_index])
    returns = net_worth.pct_change().iloc[1:]
    qs.reports.metrics(returns,mode='full')


if __name__ == "__main__":
    t_id='20210902_043521_0_12'
    config = {}
    with open(f'/workspace/params/{t_id}.json', 'r') as params:
        # Read & print the entire file
        config =json.loads(params.read())

    predict(config)