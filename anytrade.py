import numpy as np
import pandas as pd
import sys, os
import gym
import gym_anytrading
import quantstats as qs
import optuna
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import matplotlib.pyplot as plt
from datetime import datetime


from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

def load_dataset(name, index_name):
    path = os.path.join('/workspace/data', name + '.csv')
    df = pd.read_csv(path, parse_dates=True, index_col=index_name)
    return df

def linear_schedule(initial_value: float) -> Callable[[float], float]:
    """
    Linear learning rate schedule.

    :param initial_value: Initial learning rate.
    :return: schedule that computes
      current learning rate depending on remaining progress
    """
    def func(progress_remaining: float) -> float:
        """
        Progress will decrease from 1 (beginning) to 0.

        :param progress_remaining:
        :return: current learning rate
        """
        return progress_remaining * initial_value

    return func




def main(argv):

    log_path = "./log/sb3_log/"
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    

    #df = gym_anytrading.datasets.STOCKS_GOOGL.copy()
    COINS_USD_BTC = load_dataset('COINS_USD_BTC', 'Date')
    df = COINS_USD_BTC.copy()

    window_size = 20
    start_index = window_size
    end_index = len(df)
    num_cpu = 4

    env_maker = lambda: gym.make(
        'stocks-v0',
        df = df,
        window_size = window_size,
        frame_bound = (start_index, end_index)
    )

    env = DummyVecEnv([env_maker])
    env = VecNormalize(env, norm_obs=True, norm_reward=True,
                   clip_obs=10.)
    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    #model = A2C('MlpPolicy', env, verbose=1, ent_coef=0.1, learning_rate=1e-7)
    model = A2C('MlpPolicy', env, verbose=1, use_rms_prop=True, learning_rate=linear_schedule(0.001))

    # callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_path)
    # study = optuna.create_study()  # Create a new study.
    # study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
    # model = A2C('MlpPolicy', env, params=sample_a2c_params(study))
    model.set_logger(new_logger)
    # model.learn(total_timesteps=100000, callback=callback)
    # model.learn(int(2e5))

    #model.learn(int(2e6), eval_freq=1000, n_eval_episodes=5, eval_log_path=log_path, callback=callback)
    model.learn(int(2e5), eval_freq=1000, n_eval_episodes=5, eval_log_path=log_path)

    # save the model
    model.save(f"/workspace/models/a2c_stock_rms_google_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

    env = env_maker()
    observation = env.reset()

    reward_total = 0.0
    step_index = 1
    while True:
        observation = observation[np.newaxis, ...]

        # action = env.action_space.sample()
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        avg_reward = reward_total/step_index
        #tf.summary.scalar('learning rate', data=avg_reward, step=step_index)
        reward_total += reward
        step_index += 1

        # env.render()
        if done:
            print("info:", info)
            print(avg_reward)
            break

    qs.extend_pandas()

    net_worth = pd.Series(env.history['total_profit'], index=df.index[start_index+1:end_index])
    returns = net_worth.pct_change().iloc[1:]

    qs.reports.full(returns)
    qs.reports.html(returns, output='/workspace/a2c_quantstats.html')



if __name__ == "__main__":
    main(sys.argv[1:])


