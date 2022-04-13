import numpy as np
from stable_baselines3.common.policies import ActorCriticPolicy


net_arch = [dict(pi=[64,64,32,32], vf=[64,64,32,32])]

class CustomPolicy(ActorCriticPolicy):
    def __init__(self, *args, **kwargs):
        super(CustomPolicy, self).__init__(*args, **kwargs,
                                           net_arch=net_arch)

observe_vehicles = 6
config = {
    "observation": {
                "type": "OccupancyGrid",
                "features": ['presence', 'vx','vy','on_road'],
                "grid_size": [[-18, 18], [-18, 18]],
                "grid_step": [3, 3],
                "as_image": False,
                "align_to_vehicle_axes": True
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