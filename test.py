import numpy as np
import gym
import copy
import time
import pprint
import highway_env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN, PPO
from sb3_contrib import TRPO

observe_vehicles = 7
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": observe_vehicles,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"]},
    "simulation_frequency": 15,
    "policy_frequency": 15,
    "collision_reward": -15,
    "spawn_probability": 0.1,

    "action": {
        "type": "DiscreteMetaAction",
        # "longitudinal": True,
        # "lateral": False,
        # "target_speeds": [0, 4.5, 9]
        # "actions_per_axis": 10,
    },
    "show_trajectories": False,
    # "render_agent": False,

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
# # model = DQN("MlpPolicy", env, verbose=1)
# model.learn(int(2e4))
# model.save("intersection_DQN/model")

# Load and test saved model
# model = DQN.load("intersection_DQN/model")

time_step = 10
av_current_position = env.controlled_vehicles[0].position
av_control_sequence = None
rollout_env = copy.deepcopy(env)

def get_other_future_trajectory(env, av_control_sequence, av_current_position, time_step):
    "check done"
    other_future_trajectoies = np.zeros(shape=[observe_vehicles-1, time_step, 2])
    for i in range(time_step):

        rollout_action = np.random.uniform(low=-1,high=1, size=(2,))
        rollout_obs, rollout_reward, rollout_done, rollout_info = env.step(rollout_action)
        other_future_trajectoies[:,i,:] = rollout_obs[1::,1:3]
        if rollout_done:
            break
    return other_future_trajectoies

future_trajectory = get_other_future_trajectory(rollout_env, av_control_sequence, av_current_position, time_step=10)
print("break_point")
# trajectory_sample = np.random.uniform(low=-1, high=1, size=(20, 10, 2))
#
# for step in range(10):
#     ti = time.time()
#
#     rollout_values = []
#     for rollout in range(20):
#         rollout_env = copy.deepcopy(env)
#         rollout_done = False
#         rollout_value = 0
#         for t in range(10):
#             rollout_action = trajectory_sample[rollout][t]
#
#             rollout_obs, rollout_reward, rollout_done, rollout_info = rollout_env.step(rollout_action)
#             rollout_value += rollout_reward
#             if rollout_done:
#                 break
#         rollout_values.append(rollout_value)
#
#     optimal_trajectory = trajectory_sample[np.argmax(rollout_values), :]
#     action = optimal_trajectory[0]
#     tf = time.time()
#     print(tf - ti)
#
#         # action, _states = model.predict(obs, deterministic=True)
#     obs, reward, done, info = env.step(action)
#     env.render()


# rollout_env = copy.deepcopy(env)
# other_poisition = {}
# for t in range(10):
#     rollout_action = trajectory_sample[1][t]

# plt.imshow(env.render(mode="rgb_array"))
# plt.show()
