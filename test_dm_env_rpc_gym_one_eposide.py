import gym
from stable_baselines3 import PPO
import _load_environment as dm_tasks
from gym_wrapper import GymFromDMEnv

_TASK_OBSERVATIONS = ['Camera', 'reward', 'Collided']
      
PORT = 30051

dm_env = dm_tasks._DemoTasksProcessEnv(
        dm_tasks._connect_to_environment(PORT, settings={}), _TASK_OBSERVATIONS,
        num_action_repeats=1)
gym_env = GymFromDMEnv(dm_env)


obs = gym_env.reset()
done = False
while not done:
    obs, reward, done, info = gym_env.step(3)
    if done:
        break
#       obs = gym_env.reset()

gym_env.close()