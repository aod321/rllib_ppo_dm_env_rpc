import gym
from stable_baselines3 import PPO
import _load_environment as dm_tasks
from gym_wrapper import GymFromDMEnv

_TASK_OBSERVATIONS = ['Camera', 'reward', 'Collided']
      
PORT = 30051

dm_env = dm_tasks._DemoTasksProcessEnv(
        dm_tasks._connect_to_environment(PORT, settings={}), _TASK_OBSERVATIONS,
        num_action_repeats=0)
gym_env = GymFromDMEnv(dm_env)

model = PPO("MlpPolicy", gym_env, verbose=1)
model.learn(total_timesteps=10000)

obs = gym_env.reset()
for i in range(1000):
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = gym_env.step(action)
    # gym_env.render()
    if done:
      obs = gym_env.reset()

gym_env.close()