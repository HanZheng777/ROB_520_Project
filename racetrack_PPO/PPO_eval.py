from PPO_train import *

if __name__ == '__main__':
    env = gym.make("racetrack-v0")
    env.configure((config))
    env.reset()
    model = PPO.load("model/best_model", env)
    # env = RecordVideo(env, video_folder="racetrack_ppo/videos", episode_trigger=lambda e: True)
    # env.unwrapped.set_record_video_wrapper(env)

    for video in range(10):
        done = False
        obs = env.reset()
        while not done:
            # Predict
            action, _states = model.predict(obs, deterministic=True)
            # Get reward
            obs, reward, done, info = env.step(action)
            # Render
            env.render()
    env.close()