import gym
from stable_baselines3 import DQN, PPO
from env_config import config
import highway_env
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy

net_arch = [dict(pi=[64,64,32,32], vf=[64,64,32,32])]

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=net_arch)

env = gym.make('intersection-v0')
env.configure(config)
env.reset()

tmp_path = "intersection_DQN/model/"
eval_callback = EvalCallback(env, best_model_save_path=tmp_path,
                             log_path=tmp_path+'log/', eval_freq=2000,
                             deterministic=True, render=True)
new_logger = configure(tmp_path+'log/', ["stdout", "tensorboard"])
model = PPO(CustomPolicy, env,
              learning_rate=5e-4,
              batch_size=32,
              gamma=0.8,
              verbose=1,
              )

model.set_logger(new_logger)
model.learn(int(2e4), callback=eval_callback)
model.save("intersection_DQN/model/latest_model")


