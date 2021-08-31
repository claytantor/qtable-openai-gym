import numpy as np
import pandas as pd
import sys, os
import gym
import gym_anytrading
import quantstats as qs
import tensorflow as tf
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
    base_dir = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(base_dir, 'data', name + '.csv')
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


# class SaveOnBestTrainingRewardCallback(BaseCallback):
#     """
#     Callback for saving a model (the check is done every ``check_freq`` steps)
#     based on the training reward (in practice, we recommend using ``EvalCallback``).

#     :param check_freq: (int)
#     :param log_dir: (str) Path to the folder where the model will be saved.
#       It must contains the file created by the ``Monitor`` wrapper.
#     :param verbose: (int)
#     """
#     def __init__(self, check_freq: int, log_dir: str, verbose=1):
#         super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
#         self.check_freq = check_freq
#         self.log_dir = log_dir
#         self.save_path = os.path.join(log_dir, 'best_model')
#         self.best_mean_reward = -np.inf

#     def _init_callback(self) -> None:
#         # Create folder if needed
#         if self.save_path is not None:
#             os.makedirs(self.save_path, exist_ok=True)

#     def _on_step(self) -> bool:
        
#         if self.n_calls % self.check_freq == 0:
          
#           # Retrieve training reward
#           x, y = ts2xy(load_results(self.log_dir), 'timesteps')
#           print("check best",x)  
#           if len(x) > 0:
#               # Mean training reward over the last 100 episodes
#               mean_reward = np.mean(y[-100:])
#               if self.verbose > 0:
#                 print("Num timesteps: {}".format(self.num_timesteps))
#                 print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

#               # New best model, you could save the agent here
#               if mean_reward > self.best_mean_reward:
#                   self.best_mean_reward = mean_reward
#                   # Example for saving best model
#                   if self.verbose > 0:
#                     print("Saving new best model to {}".format(self.save_path))
#                   self.model.save(self.save_path)

#         return True


def main(argv):

    log_path = "./log/sb3_log/"
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    

    df = gym_anytrading.datasets.STOCKS_GOOGL.copy()

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
    model.save(f"./models/a2c_stock_rms_google_{datetime.now().strftime('%Y%m%d_%H%M%S')}")

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
        tf.summary.scalar('learning rate', data=avg_reward, step=step_index)
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
    qs.reports.html(returns, output='a2c_quantstats.html')



if __name__ == "__main__":
    main(sys.argv[1:])


