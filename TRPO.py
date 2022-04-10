import numpy as np
import gym
import logging
import copy
import time
import pprint
import highway_env
from matplotlib import pyplot as plt
from stable_baselines3 import DQN, PPO, A2C
from sb3_contrib import TRPO
from stable_baselines3.common.logger import configure
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.policies import ActorCriticPolicy



net_arch = [dict(pi=[64,64,32,32], vf=[64,64,32,32])]

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=net_arch)

observe_vehicles = 6
config = {
    "observation": {
        "type": "Kinematics",
        "vehicles_count": observe_vehicles,
        "features": ["presence", "x", "y", "vx", "vy", "cos_h", "sin_h"],
        "features_range": {
            "x": [-100, 100],
            "y": [-100, 100],
            "vx": [-20, 20],
            "vy": [-20, 20],
        },
    },
    "duration": 30,
    # "observation": {
    #             "type": "AttributesObservation",
    #             "attributes": ["state", "derivative", "reference_state"]
    #         },
    "simulation_frequency": 15,
    "policy_frequency": 15,
    "spawn_probability": 0,

    "action": {
        "type": "ContinuousAction",
        "acceleration_range": [-5, 5],
        "steering_range": (-np.pi / 3, np.pi / 3),
        "longitudinal": True,
        "lateral": True,
    },
    "show_trajectories": False,
    "initial_vehicle_count": 5,
    # "render_agent": False,
    # "state_noise": 0,
    # "derivative_noise": 0,
    "collision_reward": -500,
    "lane_keep_reward": 10,
    "high_speed_reward": 1,
    "arrived_reward": 10,
    "action_reward": 1,
    "offroad_terminal": True,

}

#
env = gym.make('myintersection-env-v0')
env.configure((config))
obs = env.reset()
# #
tmp_path = "intersection_TRPO/best_model/log/"
# #
# # # set call back
eval_callback = EvalCallback(env, best_model_save_path="intersection_TRPO/best_model",
                             log_path=tmp_path, eval_freq=10000,
                             deterministic=True, render=True)
new_logger = configure(tmp_path, ["stdout", "tensorboard"])

model = TRPO(CustomPolicy,
             env,
             learning_rate=1e-3,
             gamma=1,
             verbose=1,
             )

model.set_logger(new_logger)
model.learn(int(2000000), callback=eval_callback)
model.save("intersection_TRPO/best_model/last_model")


# pre_set_av_actions_tejct = [[0, 0], [-1.316872427983539, 0.0], [-0.924881595511072, 0.0], [-0.6495739052149101, 0.0],
#                             [-0.45621651504805943, 0.0], [-0.32041543992401483, 0.0], [-0.22503800444594843, 0.0],
#                             [-0.1580513830950962, 0.0], [-0.11100453792138273, 0.0], [-0.0779620348638499, 0.0],
#                             [-0.05475522887556927, 0.0], [-0.038456347303551894, 0.0], [-0.027009121837335098, 0.0],
#                             [-0.018969369520872213, 0.0], [-0.013322794505740514, 0.0], [-0.009357024399093325, 0.0],
#                             [-0.006571737300872608, 0.0], [-0.004615541149582588, 0.0], [-0.003241642069392962, 0.0],
#                             [-0.0022767088333737937, 0.0], [-0.0015990053809152491, 0.0],
#                             [-0.0011230325857732502, 0.1771302837342867], [-0.0007887416788943113, -0.1664212420718221],
#                             [-0.0005539584905249011, -0.26588426013984856],
#                             [-0.0003890627532913508, -0.30630885707814237],
#                             [-0.00027325120670163017, -0.3292503859121649],
#                             [-0.00019191305600981443, -0.3448051723185103],
#                             [-0.00013478667308201392, -0.356075898423417],
#                             [-9.466498850289422e-05, -0.36442156153777927],
#                             [-6.64862470678429e-05, -0.37064555356707063],
#                             [-4.669541632177736e-05, -0.3752993987416928],
#                             [-3.279568333939646e-05, -0.37878321915081914],
#                             [-2.3033456611661528e-05, -0.558447413858122],
#                             [-1.617713276521234e-05, -0.21959840220008878],
#                             [-1.1361717389135606e-05, -0.11584566873181561],
#                             [-7.979697261480586e-06, -0.07536227612232796],
#                             [-5.60439643138011e-06, -0.053592670636127104],
#                             [-3.936146738917993e-06, -0.03928720021655574],
#                             [-2.7644816598855946e-06, -0.029078187502037323],
#                             [-1.941583827047566e-06, -0.021586400567841703],
#                             [-1.3636363771496463e-06, -0.016039844227981485],
#                             [-9.577254100889832e-07, -0.011922063094262275],
#                             [-6.726411664696268e-07, -0.008862313892422794],
#                             [-4.724173911322775e-07, -0.0065880812512411225],
#                             [-3.3179383388907507e-07, -0.004897530261505548],
#                             [-2.330294150236038e-07, -0.0036408104078503364],
#                             [-1.636640067914641e-07, -0.002706576415799849]]

#

# test the best model

model = TRPO.load("intersection_TRPO/best_model_0407/best_model")

#
# keep training the model
# model.set_env(env)
# model.set_logger(new_logger)
# model.learn(int(1000000), callback=eval_callback, reset_num_timesteps=False)
# model.save("intersection_TRPO/best_model/last_model")
# # #
test_value = []
for i in range(20):
    done = False
    obs = env.reset()
    value = 0
    j = 0
    while not done:
        action, _states = model.predict(obs, deterministic=True)
        # action = [pre_set_av_actions_tejct[j][0], 0]
        j += 1
        obs, reward, done, info = env.step(action)
        value += reward
        env.render()
    test_value.append(value)

# logging.basicConfig(filename="intersection_TRPO/best_model/config.log", encoding="utf-8", level=logging.INFO)
# logging.info(net_arch)
# logging.info(config)
# logging.info(test_value)


