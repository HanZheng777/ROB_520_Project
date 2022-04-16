import tensorflow as tf
from tensorflow.keras.layers import Input, Dense, Lambda

import gym
import highway_env
from highway_env.envs import RacetrackEnv
import argparse
import numpy as np
from stable_baselines3 import PPO
from env_config import config
import copy
import sys
sys.path.append('/')
# import os
# current = os.path.dirname(os.path.realpath(__file__))
# parent = os.path.dirname(os.path.dirname(current))
# sys.path.append(parent)

dtype_all = 'float32'
tf.keras.backend.set_floatx(dtype_all)

parser = argparse.ArgumentParser()
parser.add_argument('--gamma', type=float, default=0.99)
parser.add_argument('--update_interval', type=int, default=5)
parser.add_argument('--actor_lr', type=float, default=0.0005)
parser.add_argument('--critic_lr', type=float, default=0.001)
parser.add_argument('--clip_ratio', type=float, default=0.1)
parser.add_argument('--lmbda', type=float, default=0.95)
parser.add_argument('--epochs', type=int, default=3)

args = parser.parse_args()


class Actor:
    def __init__(self, obs_dim, action_dim, action_bound, std_bound):
        self.obs_dim = obs_dim
        self.action_dim = action_dim
        self.action_bound = action_bound
        self.std_bound = std_bound
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.actor_lr)

    def get_action(self, obs):
        obs = np.reshape(obs, [1, self.obs_dim])
        mu, std = self.model.predict(obs)
        action = np.random.normal(mu[0], std[0], size=self.action_dim)
        action = np.clip(action, 0, self.action_bound)
        log_policy = self.log_pdf(mu, std, action)

        return log_policy, action

    def log_pdf(self, mu, std, action):
        std = tf.clip_by_value(std, self.std_bound[0], self.std_bound[1])
        var = std ** 2
        log_policy_pdf = -0.5 * (action - mu) ** 2 / \
                         var - 0.5 * tf.math.log(var * 2 * np.pi)
        return tf.reduce_sum(log_policy_pdf, 1, keepdims=True)

    def create_model(self):
        obs_input = Input((self.obs_dim,))
        dense_1 = Dense(32, activation='relu')(obs_input)
        dense_2 = Dense(32, activation='relu')(dense_1)
        out_mu = Dense(self.action_dim, activation='tanh')(dense_2)
        mu_output = Lambda(lambda x: x * self.action_bound)(out_mu)
        std_output = Dense(self.action_dim, activation='softplus')(dense_2)
        return tf.keras.models.Model(obs_input, [mu_output, std_output])

    def compute_loss(self, log_old_policy, log_new_policy, actions, gaes):
        ratio = tf.exp(log_new_policy - tf.stop_gradient(log_old_policy))
        gaes = tf.stop_gradient(gaes)
        clipped_ratio = tf.clip_by_value(
            ratio, 1.0 - args.clip_ratio, 1.0 + args.clip_ratio)
        surrogate = -tf.minimum(ratio * gaes, clipped_ratio * gaes)
        return tf.reduce_mean(surrogate)

    def train(self, log_old_policy, obss, actions, gaes):
        with tf.GradientTape() as tape:
            mu, std = self.model(obss, training=True)
            log_new_policy = self.log_pdf(mu, std, actions)
            loss = self.compute_loss(
                log_old_policy, log_new_policy, actions, gaes)
        grads = tape.gradient(loss, self.model.trainable_variables)

        self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Critic:
    def __init__(self, obs_dim):
        self.obs_dim = obs_dim
        self.model = self.create_model()
        self.opt = tf.keras.optimizers.Adam(args.critic_lr)

    def create_model(self):
        return tf.keras.Sequential([
            Input((self.obs_dim,)),
            Dense(32, activation='relu'),
            Dense(32, activation='relu'),
            Dense(16, activation='relu'),
            Dense(1, activation='linear')
        ])

    def compute_loss(self, v_pred, td_targets):
        mse = tf.keras.losses.MeanSquaredError()
        return mse(td_targets, v_pred)

    def train(self, obss, td_targets):
        with tf.GradientTape() as tape:
            v_pred = self.model(obss, training=True)
            assert v_pred.shape == td_targets.shape
            loss = self.compute_loss(v_pred, tf.stop_gradient(td_targets))
            grads = tape.gradient(loss, self.model.trainable_variables)

            self.opt.apply_gradients(zip(grads, self.model.trainable_variables))
        return loss


class Agent:
    def __init__(self, env, look_ahead, max_perturbation_std, RL_action_agent, multimodal=False):

        self.env = env
        self.obs_dim = np.product(self.env.observation_space.shape)
        if multimodal:
            self.perturbation_dim = look_ahead
        else:
            self.perturbation_dim = 1
        self.perturbation_bound = max_perturbation_std
        self.std_bound = [1e-2, 1.0]

        self.actor_opt = tf.keras.optimizers.Adam(args.actor_lr)
        self.critic_opt = tf.keras.optimizers.Adam(args.critic_lr)
        self.actor = Actor(self.obs_dim, self.perturbation_dim,
                           self.perturbation_bound, self.std_bound)
        self.critic = Critic(self.obs_dim)

        self.look_ahead = look_ahead
        self.RL_action_agent = PPO.load(RL_action_agent)



        print('done')

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

    def gae_target(self, rewards, v_values, next_v_value, done):
        n_step_targets = np.zeros_like(rewards, dtype=dtype_all)
        gae = np.zeros_like(rewards, dtype=dtype_all)
        gae_cumulative = 0
        forward_val = 0

        if not done:
            forward_val = next_v_value

        for k in reversed(range(0, len(rewards))):
            delta = rewards[k] + args.gamma * forward_val - v_values[k]
            gae_cumulative = args.gamma * args.lmbda * gae_cumulative + delta
            gae[k] = gae_cumulative
            forward_val = v_values[k]
            n_step_targets[k] = gae[k] + v_values[k]
        return gae, n_step_targets

    def list_to_batch(self, list):
        batch = list[0]
        for elem in list[1:]:
            batch = np.append(batch, elem, axis=0)
        return batch

    def train(self, save_path, max_episodes=15, saving_freq=5, train_from_scratch=False):

        step = tf.Variable(1)
        actor_ckpt = tf.train.Checkpoint(step=step, optimizer=self.actor.model, model=self.actor_opt)
        critic_ckpt = tf.train.Checkpoint(step=step, optimizer=self.critic_opt, model=self.critic.model)
        actor_save_path = save_path + "actor/"
        critic_save_path = save_path + "critic/"

        if train_from_scratch:
            actor_ckpt.restore(tf.train.latest_checkpoint(actor_save_path))
            critic_ckpt.restore(tf.train.latest_checkpoint(critic_save_path))

        for ep in range(max_episodes):
            obs_batch = []
            perturbation_batch = []
            look_ahead_reward_batch = []
            old_policy_batch = []

            episode_reward, done = 0, False

            obs = self.env.reset()

            while not done:
                # self.env.render()
                log_old_policy, perturbation = self.actor.get_action(obs)
                variations = np.random.normal(loc=np.zeros(len(perturbation), dtype=dtype_all),
                                              scale=perturbation, size=[self.look_ahead, ])

                self.export_env_vehicle_state()
                look_ahead_reward = 0
                obs_r = obs.copy()
                for t in range(self.look_ahead):
                    RL_action = self.RL_action_agent.predict(obs_r, deterministic=True)[0]
                    rollout_action = RL_action + variations[t]
                    next_obs_r, reward_r, done_r, _ = self.env.step(rollout_action)
                    look_ahead_reward += reward_r
                    obs_r = next_obs_r
                self.return_initial_obs()

                action = self.RL_action_agent.predict(obs, deterministic=True)[0] + variations[0]
                next_obs, reward, done, _ = self.env.step(action)

                obs = np.reshape(obs, [1, self.obs_dim])
                perturbation = np.reshape(perturbation, [1, 1])
                next_obs = np.reshape(next_obs, [1, self.obs_dim])
                look_ahead_reward = np.reshape(look_ahead_reward, [1, 1])
                log_old_policy = np.reshape(log_old_policy, [1, 1])

                obs_batch.append(obs)
                perturbation_batch.append(perturbation)
                look_ahead_reward_batch.append(look_ahead_reward)
                old_policy_batch.append(log_old_policy)

                if len(obs_batch) >= args.update_interval or done:
                    obss = self.list_to_batch(obs_batch)
                    perturbations = self.list_to_batch(perturbation_batch)
                    look_ahead_rewards = self.list_to_batch(look_ahead_reward_batch)
                    old_policys = self.list_to_batch(old_policy_batch)

                    v_values = self.critic.model.predict(obss)
                    next_v_value = self.critic.model.predict(next_obs)

                    gaes, td_targets = self.gae_target(
                        look_ahead_rewards, v_values, next_v_value, done)

                    for epoch in range(args.epochs):
                        step.assign_add(1)
                        actor_loss = self.actor.train(
                            old_policys, obss, perturbations, gaes)
                        critic_loss = self.critic.train(obss, td_targets)


                    obs_batch = []
                    perturbation_batch = []
                    look_ahead_reward_batch = []
                    old_policy_batch = []

                episode_reward += look_ahead_reward[0][0]
                obs = next_obs.reshape(self.env.observation_space.shape)

            if ep % saving_freq == 0:
                actor_ckpt.save(actor_save_path)
                critic_ckpt.save(critic_save_path)

            print('EP{} EpisodeReward={}'.format(ep, episode_reward))


def main():
    env = RacetrackEnv()
    env.configure(config)
    env.reset()

    agent = Agent(env, look_ahead=5, max_perturbation_std=0.02,
                  RL_action_agent="PPO_action_model/best_model", multimodal=False)
    agent.train("PPO_perturbation_model/")


if __name__ == "__main__":
    main()
