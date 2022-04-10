import numpy as np
import gym
import copy
import time
import pprint
import highway_env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN

observe_vehicles = 15
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": observe_vehicles,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]},
    "simulation_frequency": 15,
    "policy_frequency": 5,
    "collision_reward": -15,
    "spawn_probability": 0.1,

}
env = gym.make('intersection-v0')
env.configure(config)
obs = env.reset()


# model = DQN('MlpPolicy', env,
#               policy_kwargs=dict(net_arch=[256, 256]),
#               learning_rate=5e-4,
#               buffer_size=15000,
#               learning_starts=200,
#               batch_size=32,
#               gamma=0.8,
#               train_freq=1,
#               gradient_steps=1,
#               target_update_interval=50,
#               verbose=1,
#               tensorboard_log="intersection_DQN/")
#
# model.learn(int(5e3))
# model.save("intersection_DQN/model")

# Load and test saved model
model = DQN.load("intersection_DQN/model")

for i in range(1):
    done = False
    obs = env.reset()
    av_action_trej = []
    while not done:
      action, _states = model.predict(obs, deterministic=True)
      av_action = [env.controlled_vehicles[0].action["acceleration"],env.controlled_vehicles[0].action["steering"] ]
      av_action_trej.append(av_action)
      obs, reward, done, info = env.step(action)
      env.render()

    print(av_action_trej)