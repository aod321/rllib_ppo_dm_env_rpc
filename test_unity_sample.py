import ray
from ray import tune
from ray.rllib import env
from dm_env import specs
import _load_environment as dm_tasks
import numpy as np
from ray.rllib.agents.ppo import PPOTrainer

_TASK_OBSERVATIONS = ['Camera', 'reward', 'Collided']
PORT = 30051


def my_train_fn(config, reporter):
    iterations = config.pop("train-iterations", 10)

    # Train for n iterations with high LR
    agent1 = PPOTrainer(env=env_creator,    config=config)
    for _ in range(iterations):
        result = agent1.train()
        result["phase"] = 1
        reporter(**result)
        phase1_time = result["timesteps_total"]
    state = agent1.save()
    agent1.stop()


def env_creator(config):
    return dm_tasks._DemoTasksProcessEnv(
        dm_tasks._connect_to_environment(PORT, settings={}), _TASK_OBSERVATIONS,
        num_action_repeats=0)


if __name__ == '__main__':
    ray.init()
    tune.register_env(
        "unity3d",
        lambda config: env_creator(config), 
    )

    config = {
        "env": "unity3d",
        # For running in editor, force to use just one Worker (we only have
        # one Unity running)!
        "num_workers": 0,
        # Other settings.
        "lr": 0.0003,
        "lambda": 0.95,
        "gamma": 0.99,
        "sgd_minibatch_size": 256,
        "train_batch_size": 4000,
        # Use GPUs iff `RLLIB_NUM_GPUS` env var set to > 0.
        "num_gpus": 0,
        "num_sgd_iter": 20,
        "rollout_fragment_length": 200,
        "clip_param": 0.2,
        "model": {
            "fcnet_hiddens": [512, 512],
        },
        "framework": "tf",
        "no_done_at_end": True,
    }

    # Run the experiment.
    results = tune.run(
        run=my_train_fn,
        config=config,
        verbose=1,
        checkpoint_freq=5,
        checkpoint_at_end=True,
    )

    ray.shutdown()