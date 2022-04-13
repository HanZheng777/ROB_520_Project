import gym
from sb3_contrib import TRPO
from stable_baselines3.common.logger import configure
import highway_env
from stable_baselines3.common.callbacks import EvalCallback
from env_config import CustomPolicy, config

if __name__ == '__main__':
    env = gym.make('myintersection-env-v0')
    env.configure((config))
    env.reset()
    # #
    tmp_path = "model/"
    # #
    # # # set call back
    eval_callback = EvalCallback(env, best_model_save_path=tmp_path,
                                 log_path=tmp_path+'log/', eval_freq=10000,
                                 deterministic=True, render=True)
    new_logger = configure(tmp_path+'log/', ["stdout", "tensorboard"])

    model = TRPO(CustomPolicy,
                 env,
                 learning_rate=1e-3,
                 gamma=1,
                 verbose=1,
                 )

    model.set_logger(new_logger)
    model.learn(int(200000), callback=eval_callback)
    model.save("intersection_TRPO/model/latest_model")

    # keep training the model
    # model.set_env(env)
    # model.learn(int(1000000), callback=eval_callback, reset_num_timesteps=False)
    # model.save("intersection_TRPO/best_model/last_model")



