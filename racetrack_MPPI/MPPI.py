import numpy as np
import copy
from stable_baselines3 import DQN, DDPG, PPO
from highway_env.envs.racetrack_env import RacetrackEnv

from env_config import config


class MPPI(object):

    def __init__(self, look_ahead, num_rollout, prior_std, lamda):
        """

        :param look_ahead:
        :param num_rollout:
        :param prior_std:
        :param lamda:
        """
        self.num_rollout = num_rollout
        self.look_ahead = look_ahead
        self.env = RacetrackEnv()
        self.env.configure(config)
        self.env.reset()
        self.std = prior_std
        self.lamda = lamda
        self.RL_model = PPO.load("model/best_model")


    def export_env_vehicle_state(self):
        """

        :return:
        """

        self.other_vehicle_copy = [copy.deepcopy(vehicle) for vehicle in self.env.road.vehicles]
        self.env_step_copy = self.env.steps
        self.time_copy = self.env.time


    def return_initial_obs(self):
        """

        :return:
        """
        self.env.road.vehicles = copy.deepcopy(self.other_vehicle_copy)
        self.env.controlled_vehicles[0] = self.env.road.vehicles[0]
        self.env.vehicles = self.env.road.vehicles[0]

        self.env.steps = self.env_step_copy
        self.env.time = self.time_copy


    def rollout(self, action_traj):
        """

        :param action_traj:
        :return:
        """
        rollout_perturbations = []
        rollout_values = []
        for i in range(self.num_rollout):
            self.return_initial_obs()
            perturbation = np.random.normal(loc=0, scale=self.std, size=self.look_ahead)
            rollout_action = action_traj + perturbation
            rollout_value = 0

            for t in range(self.look_ahead):
                action = rollout_action[t]
                obs, reward, done, info = self.env.step([action])
                rollout_value += reward + self.lamda*action/self.std*perturbation[t]
                if done:
                    break

            rollout_perturbations.append(perturbation)
            rollout_values.append(rollout_value)

        return np.array(rollout_perturbations), np.array(rollout_values)

    def get_weighted_action_traj(self, action_traj, rollout_perturbations, rollout_values):
        """

        :param action_traj:
        :param rollout_perturbations:
        :param rollout_values:
        :return:
        """
        beta = np.min(rollout_values)
        yita = np.sum(np.exp(-1/self.lamda*(rollout_values - beta)))
        weights = 1/yita * np.exp(-1/self.lamda*(rollout_values - beta))

        weighted_action_traj = action_traj + np.dot(weights, rollout_perturbations)

        return weighted_action_traj


    def get_initial_trajectory(self):
        """

        :return:
        """
        obs = self.env.reset()
        action_traj = []
        i = 0
        while i < self.look_ahead:
            action, _states = self.RL_model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, info = self.env.step(action)
            action_traj.append(action[0])
            i += 1
            if done:
                i = 0
                self.env.reset()
                break

        return np.array(action_traj)


    def run(self):
        """

        :return:
        """
        self.export_env_vehicle_state()
        action_traj = self.get_initial_trajectory()
        done = False
        self.return_initial_obs()

        while not done:

            self.export_env_vehicle_state()
            self.return_initial_obs()
            perturbations, values = self.rollout(action_traj)
            action_traj = self.get_weighted_action_traj(action_traj, perturbations, values)
            self.return_initial_obs()

            for t in range(self.look_ahead):
                c_action = action_traj[t]
                c_obs, c_reward, c_done, c_info = self.env.step([c_action])
                if c_done:
                    break
            new_action, _new_state = self.RL_model.predict(c_obs, deterministic=True)
            self.return_initial_obs()

            action = action_traj[0]
            obs, reward, done, info = self.env.step([action])

            action_traj = np.delete(action_traj, 0)
            action_traj = np.append(action_traj, new_action)

            self.env.render()


if __name__ == "__main__":
    controller = MPPI(look_ahead=5, num_rollout=10, prior_std=0.02, lamda=1)
    for j in range(1):
        controller.run()




