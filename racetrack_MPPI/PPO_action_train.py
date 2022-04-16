import numpy as np
import gym
from gym.wrappers import RecordVideo
from stable_baselines3 import DQN, DDPG, PPO
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.vec_env import SubprocVecEnv
import highway_env
import sys
import os
current = os.path.dirname(os.path.realpath(__file__))
parent = os.path.dirname(os.path.dirname(current))
sys.path.append(parent)
from racetrack_MPPI.env_config import config

if __name__ == '__main__':

    n_cpu = 6
    batch_size = 64
    env = gym.make("racetrack-v0")
    env.configure((config))
    env.reset()
    model = PPO("MlpPolicy",
                env,
                policy_kwargs=dict(net_arch=[dict(pi=[256, 256], vf=[256, 256])]),
                n_steps=batch_size * 12 // n_cpu,
                batch_size=batch_size,
                n_epochs=10,
                learning_rate=1e-3,
                gamma=0.9,
                verbose=1,
                )
    tmp_path = "PPO_action_model/"
    eval_callback = EvalCallback(env, best_model_save_path=tmp_path,
                                 log_path=tmp_path + 'log/', eval_freq=10000,
                                 deterministic=True, render=True)
    new_logger = configure(tmp_path + 'log/', ["stdout", "tensorboard"])
    model.set_logger(new_logger)
    # Train the model
    model.learn(total_timesteps=int(1e4), callback=eval_callback)
    model.save("PPO_action_model/latest_model")
