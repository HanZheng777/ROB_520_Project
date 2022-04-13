import gym
import highway_env
from env_config import config
from stable_baselines3 import DQN, PPO

env = gym.make('intersection-v0')
env.configure(config)
env.reset()

model = PPO.load("model/latest_model")
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