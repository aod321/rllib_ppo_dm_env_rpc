
from dm_env import specs
import _load_environment as dm_tasks
import numpy as np

PORT = 30015


class RandomAgent(object):
  """Basic random agent for DeepMind Memory Tasks."""

  def __init__(self, action_spec):
    self.action_spec = action_spec

  def act(self):
    action = {}

    for name, spec in self.action_spec.items():
      # Uniformly sample BoundedArray actions.
      if isinstance(spec, specs.BoundedArray):
        action[name] = np.random.uniform(spec.minimum, spec.maximum, spec.shape)
      else:
        action[name] = spec.generate_value()
    return action


def main(_):
  env_settings = dm_tasks.EnvironmentSettings(
      seed=FLAGS.seed, level_name=FLAGS.level_name)
  with dm_tasks._connect_to_environment(port) as env:
    agent = RandomAgent(env.action_spec())

    timestep = env.reset()
    score = 0
    while not timestep.last():
      action = agent.act()
      timestep = env.step(action)

      if timestep.reward:
        score += timestep.reward
        logging.info('Total score: %1.1f, reward: %1.1f', score,
                     timestep.reward)


if __name__ == '__main__':
  app.run(main)
