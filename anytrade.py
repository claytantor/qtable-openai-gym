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



from stable_baselines3 import A2C
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure

from stable_baselines3.common import results_plotter
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.results_plotter import load_results, ts2xy, plot_results
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import BaseCallback

# def linear_schedule(initial_value: Union[float, str]) -> Callable[[float], float]:
#     """
#     Linear learning rate schedule.
#     :param initial_value: (float or str)
#     :return: (function)
#     """
#     if isinstance(initial_value, str):
#         initial_value = float(initial_value)

#     def func(progress_remaining: float) -> float:
#         """
#         Progress will decrease from 1 (beginning) to 0
#         :param progress_remaining: (float)
#         :return: (float)
#         """
#         return progress_remaining * initial_value

#     return func


# def sample_a2c_params(trial: optuna.Trial) -> Dict[str, Any]:
#     """
#     Sampler for A2C hyperparams.
#     :param trial:
#     :return:
#     """
#     gamma = trial.suggest_categorical("gamma", [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9999])
#     normalize_advantage = trial.suggest_categorical("normalize_advantage", [False, True])
#     max_grad_norm = trial.suggest_categorical("max_grad_norm", [0.3, 0.5, 0.6, 0.7, 0.8, 0.9, 1, 2, 5])
#     # Toggle PyTorch RMS Prop (different from TF one, cf doc)
#     use_rms_prop = trial.suggest_categorical("use_rms_prop", [False, True])
#     gae_lambda = trial.suggest_categorical("gae_lambda", [0.8, 0.9, 0.92, 0.95, 0.98, 0.99, 1.0])
#     n_steps = trial.suggest_categorical("n_steps", [8, 16, 32, 64, 128, 256, 512, 1024, 2048])
#     lr_schedule = trial.suggest_categorical("lr_schedule", ["linear", "constant"])
#     learning_rate = trial.suggest_loguniform("learning_rate", 1e-5, 1)
#     ent_coef = trial.suggest_loguniform("ent_coef", 0.00000001, 0.1)
#     vf_coef = trial.suggest_uniform("vf_coef", 0, 1)
#     # Uncomment for gSDE (continuous actions)
#     # log_std_init = trial.suggest_uniform("log_std_init", -4, 1)
#     ortho_init = trial.suggest_categorical("ortho_init", [False, True])
#     net_arch = trial.suggest_categorical("net_arch", ["small", "medium"])
#     # sde_net_arch = trial.suggest_categorical("sde_net_arch", [None, "tiny", "small"])
#     # full_std = trial.suggest_categorical("full_std", [False, True])
#     # activation_fn = trial.suggest_categorical('activation_fn', ['tanh', 'relu', 'elu', 'leaky_relu'])
#     activation_fn = trial.suggest_categorical("activation_fn", ["tanh", "relu"])

#     if lr_schedule == "linear":
#         learning_rate = linear_schedule(learning_rate)

#     net_arch = {
#         "small": [dict(pi=[64, 64], vf=[64, 64])],
#         "medium": [dict(pi=[256, 256], vf=[256, 256])],
#     }[net_arch]

#     # sde_net_arch = {
#     #     None: None,
#     #     "tiny": [64],
#     #     "small": [64, 64],
#     # }[sde_net_arch]

#     activation_fn = {"tanh": nn.Tanh, "relu": nn.ReLU, "elu": nn.ELU, "leaky_relu": nn.LeakyReLU}[activation_fn]

#     return {
#         "n_steps": n_steps,
#         "gamma": gamma,
#         "gae_lambda": gae_lambda,
#         "learning_rate": learning_rate,
#         "ent_coef": ent_coef,
#         "normalize_advantage": normalize_advantage,
#         "max_grad_norm": max_grad_norm,
#         "use_rms_prop": use_rms_prop,
#         "vf_coef": vf_coef,
#         "policy_kwargs": dict(
#             # log_std_init=log_std_init,
#             net_arch=net_arch,
#             # full_std=full_std,
#             activation_fn=activation_fn,
#             # sde_net_arch=sde_net_arch,
#             ortho_init=ortho_init,
#         ),
#     }

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


class SaveOnBestTrainingRewardCallback(BaseCallback):
    """
    Callback for saving a model (the check is done every ``check_freq`` steps)
    based on the training reward (in practice, we recommend using ``EvalCallback``).

    :param check_freq: (int)
    :param log_dir: (str) Path to the folder where the model will be saved.
      It must contains the file created by the ``Monitor`` wrapper.
    :param verbose: (int)
    """
    def __init__(self, check_freq: int, log_dir: str, verbose=1):
        super(SaveOnBestTrainingRewardCallback, self).__init__(verbose)
        self.check_freq = check_freq
        self.log_dir = log_dir
        self.save_path = os.path.join(log_dir, 'best_model')
        self.best_mean_reward = -np.inf

    def _init_callback(self) -> None:
        # Create folder if needed
        if self.save_path is not None:
            os.makedirs(self.save_path, exist_ok=True)

    def _on_step(self) -> bool:
        
        if self.n_calls % self.check_freq == 0:
          
          # Retrieve training reward
          x, y = ts2xy(load_results(self.log_dir), 'timesteps')
          print("check best",x)  
          if len(x) > 0:
              # Mean training reward over the last 100 episodes
              mean_reward = np.mean(y[-100:])
              if self.verbose > 0:
                print("Num timesteps: {}".format(self.num_timesteps))
                print("Best mean reward: {:.2f} - Last mean reward per episode: {:.2f}".format(self.best_mean_reward, mean_reward))

              # New best model, you could save the agent here
              if mean_reward > self.best_mean_reward:
                  self.best_mean_reward = mean_reward
                  # Example for saving best model
                  if self.verbose > 0:
                    print("Saving new best model to {}".format(self.save_path))
                  self.model.save(self.save_path)

        return True


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
    model = A2C('MlpPolicy', env, verbose=1, learning_rate=linear_schedule(0.001))

    
    callback = SaveOnBestTrainingRewardCallback(check_freq=1000, log_dir=log_path)
    # study = optuna.create_study()  # Create a new study.
    # study.optimize(objective, n_trials=100)  # Invoke optimization of the objective function.
    # model = A2C('MlpPolicy', env, params=sample_a2c_params(study))
    model.set_logger(new_logger)
    # model.learn(total_timesteps=100000, callback=callback)
    # model.learn(int(2e5))

    model.learn(int(2e5), eval_freq=1000, n_eval_episodes=5, eval_log_path=log_path)

    # save the model
    model.save("sac_pendulum")

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


