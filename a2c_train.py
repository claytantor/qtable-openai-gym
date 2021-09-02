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


t_id = datetime.now().strftime('%Y%m%d_%H%M%S')


def get_untuned_config():
    return {
        **dotenv_values(f"/workspace/config/.env"),  # load shared development variables
        **os.environ,  # override loaded values with environment variables
    }

def update_config_values_untuned(config):

    """
    set basic param values, tuned params will overwrite
    learning_rate=0.0007|0.0008
    n_steps=8|10
    gae_lambda=0.9|1.0
    vf_coef=0.44|0.46
    total_timesteps=908511|1508511
    gamma=0.9|1.0
    """

    config['learning_rate'] = float(config['learning_rate'])
    config['gamma'] = float(config['gamma'])
    config['n_steps'] = int(config['n_steps']) 

    config['vf_coef'] = float(config['vf_coef'])
    config['gae_lambda'] = float(config['gae_lambda'])
    config['total_timesteps'] = int(float(config['total_timesteps']))



def update_config_values_shared(config):

    """
    set basic param values, tuned params will overwrite
    """

    config['use_rms_prop'] = int(config['use_rms_prop'])
    config['sde_sample_freq'] = int(config['sde_sample_freq'])
    config['verbose'] = int(config['verbose'])
    config['training_id'] = f'{t_id}_0'
    config['use_rms_prop'] = (config['use_rms_prop'] == 1)
    config['use_schedule'] = (config['use_schedule']=='True')
    config['learn_eval_freq'] = int(config['learn_eval_freq'])
    config['n_eval_episodes'] = int(config['n_eval_episodes'])



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

def get_tuned_config():
    return {
        **dotenv_values(f"/workspace/config/.env"),  # load shared development variables
        **dotenv_values(f"/workspace/config/.env.tuned"),
        **os.environ,  # override loaded values with environment variables
    }


def update_config_values_tuned(trial, config):

    # Override tuned values
    # linear_schedule_parts=config['learning_rate'].split("|")
    # config['learning_rate'] = trial.suggest_uniform('learning_rate',
    #     float(linear_schedule_parts[0]),float(linear_schedule_parts[1]))
    config_uniform_field('learning_rate', config, trial)

    # n_steps_parts = config['n_steps'].split("|")
    # config['n_steps'] = trial.suggest_int('n_steps', int(n_steps_parts[0]), int(n_steps_parts[1]))
    config_int_field("n_steps", config, trial)

    # total_timesteps_parts = config['total_timesteps'].split("|")
    # config['total_timesteps'] = trial.suggest_int('total_timesteps', 
    #     int(float(total_timesteps_parts[0])), int(float(total_timesteps_parts[1])))
    config_int_field("total_timesteps", config, trial)

    # gae_lambda_parts=config['gae_lambda'].split("|")
    # config['gae_lambda'] = trial.suggest_uniform('gae_lambda',
    #     float(gae_lambda_parts[0]),float(gae_lambda_parts[1]))  
    config_uniform_field('gae_lambda', config, trial)

    # vf_coef_parts=config['vf_coef'].split("|")
    # config['vf_coef'] = trial.suggest_uniform('vf_coef',
    #     float(vf_coef_parts[0]),float(vf_coef_parts[1]))  
    config_uniform_field('vf_coef', config, trial)
    
    config_uniform_field('gamma', config, trial)

    
    # config['sde_sample_freq'] = int(config['sde_sample_freq'])
    # config['use_rms_prop'] = int(config['use_rms_prop'])
    # config['verbose'] = int(config['verbose'])

    # return config

def config_uniform_field(field_name, config, trial):
    f_parts=config[field_name].split("|")
    config[field_name] = trial.suggest_uniform(field_name,
        float(f_parts[0]),float(f_parts[1]))


def config_int_field(field_name, config, trial):
    f_parts = config[field_name].split("|")
    config[field_name] = trial.suggest_int(field_name, 
        int(float(f_parts[0])), int(float(f_parts[1])))

# def update_config_values_untuned(config):

#     """
#     set basic param values, tuned params will overwrite
#     learning_rate=0.0007|0.0008
#     n_steps=8|10
#     gae_lambda=0.9|1.0
#     vf_coef=0.44|0.46
#     total_timesteps=908511|1508511
#     gamma=0.9|1.0
#     """

#     config['learning_rate'] = float(config['learning_rate'])
#     config['gamma'] = float(config['gamma'])
#     config['n_steps'] = int(config['n_steps']) 

#     config['vf_coef'] = float(config['vf_coef'])
#     config['gae_lambda'] = float(config['gae_lambda'])
#     config['total_timesteps'] = int(float(config['total_timesteps']))



# def update_config_values_shared(config):

#     """
#     set basic param values, tuned params will overwrite
#     """

#     config['use_rms_prop'] = int(config['use_rms_prop'])
#     config['sde_sample_freq'] = int(config['sde_sample_freq'])
#     config['verbose'] = int(config['verbose'])
#     config['training_id'] = f'{t_id}_0'
#     config['use_rms_prop'] = (config['use_rms_prop'] == 1)
#     config['use_schedule'] = (config['use_schedule']=='True')
#     config['learn_eval_freq'] = int(config['learn_eval_freq'])
#     config['n_eval_episodes'] = int(config['n_eval_episodes'])

def train_optimized(trial):
    
    config = get_tuned_config()
    # overrite tuning params
    update_config_values_tuned(trial, config)

    # write basic vals
    update_config_values_shared(config)


    print("study_id=",trial._study_id)
    print("trial_id=",trial._trial_id)
    config['training_id'] = f'{t_id}_{trial._study_id}_{trial._trial_id}'
    return train(config)

def save_config(key_id, config):
    f = open(f"/workspace/params/{key_id}.json", "a")
    f.write(json.dumps(config, indent=3))
    f.close()

def train(config):
    
    print(json.dumps(config, indent=3))

    log_path = "/workspace/log"
    new_logger = configure(log_path, ["stdout", "csv", "tensorboard"])
    
    #df = gym_anytrading.datasets.STOCKS_GOOGL.copy()
    COINS_USD_BTC = load_dataset('COINS_USD_BTC', 'Date')
    df = COINS_USD_BTC.copy()

    window_size = int(config['n_steps'])
    start_index = window_size
    end_index = len(df)

    print("=========TRAIN=============")

    env_maker = lambda: gym.make(
        'stocks-v0',
        df = df,
        window_size = window_size,
        frame_bound = (start_index, end_index)
    )

    env = DummyVecEnv([env_maker])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.)

    # n_actions = env.action_space.shape[-1]
    # action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # learning_s = float(config['learning_rate'])
    # use_rms = (config['use_rms_prop'] == 1)

    # classstable_baselines3.a2c.A2C(policy, env, learning_rate=0.0007, 
        # n_steps=5, gamma=0.99, gae_lambda=1.0, 
        # ent_coef=0.0, vf_coef=0.5, max_grad_norm=0.5, 
        # rms_prop_eps=1e-05, use_rms_prop=True, 
        # use_sde=False, sde_sample_freq=- 1, 
        # normalize_advantage=False, 
        # tensorboard_log=None, 
        # create_eval_env=False, 
        # policy_kwargs=None, verbose=0, 
        # seed=None, device='auto', 
        # _init_setup_model=True)[source]    
    
    # l_r_v = config['learning_rate']
    # if config['use_schedule']=='True':
    #     l_r_v = linear_schedule(config['learning_rate'])

    if config['use_schedule']:
        config['learning_rate'] = linear_schedule(config['learning_rate'])

    model = A2C('MlpPolicy', env, 
        verbose=config['verbose'], 
        n_steps=config['n_steps'],
        vf_coef=config['vf_coef'],
        gae_lambda=config['gae_lambda'],
        use_rms_prop=config['use_rms_prop'],
        learning_rate=config['learning_rate'])

    model.set_logger(new_logger)
    
    model.learn(total_timesteps=config['total_timesteps'], 
        eval_freq=config['learn_eval_freq'], 
        n_eval_episodes=config['n_eval_episodes'], 
        eval_log_path=log_path)

    env = env_maker()
    observation = env.reset()

    reward_total = 0.0
    step_index = 1
    while True:
        observation = observation[np.newaxis, ...]

        # action = env.action_space.sample()
        action, _states = model.predict(observation)
        observation, reward, done, info = env.step(action)
        
        reward_total += reward
        avg_reward = reward_total/step_index

        step_index += 1

        if done:
            print("info:", info)
            print(f'total reward:{reward_total} average reward:{avg_reward}')
            break


    qs.extend_pandas()

    net_worth = pd.Series(env.history['total_profit'], index=df.index[start_index+1:end_index])
    returns = net_worth.pct_change().iloc[1:]

    if qs.stats.sharpe(returns) > 0.0:
        save_config(config['training_id'], config)
        qs.reports.html(returns, "SPY", output=f"/workspace/reports/{config['training_id']}.html")
        model.save(f"/workspace/models/a2c_btc_{config['training_id']}")

        plt.figure(figsize=(16, 6))
        env.render_all()
        plt.savefig(f"/workspace/images/plot_{config['training_id']}.png", dpi=100.0)

    qs.reports.full(returns)

    if reward_total>0.0:
        return qs.stats.sharpe(returns)
    else:
        return -5.00

def main():
    if os.getenv("OPTUNA") == "True":
        config = get_tuned_config()
        num_trials=int(os.getenv("N_TRIALS"))
        study = optuna.create_study(sampler=optuna.samplers.TPESampler(), direction='maximize')
        study.optimize(train_optimized, n_trials=num_trials)
        print(study.best_params)
        params_list=config['param_fields']
        plot_fig = optuna.visualization.plot_parallel_coordinate(study, 
            params=[params_list])
        plot_fig.write_image(f"/workspace/images/{t_id}.png")

    else:
        config = get_untuned_config()
        # overrite tuning params
        update_config_values_untuned(config)

        # write basic vals
        update_config_values_shared(config)

        train(config)



if __name__ == "__main__":
    main()



