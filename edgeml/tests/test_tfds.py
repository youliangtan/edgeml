#!/usr/bin/env python3

from edgeml.data.tfds import read_tfds, export_tfds, make_replay_buffer
from edgeml.data.jaxrl_data_store import ReplayBufferDataStore
import gym
from gym import spaces
import numpy as np


class CustomEnv(gym.Env):
    """
    A custom environment that uses a dictionary for the observation space.
    """

    def __init__(self):
        super(CustomEnv, self).__init__()
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        self.observation_space = spaces.Dict({
            'position': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            'velocity': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })

    def step(self, action):
        observation = {
            'position': np.random.randint(0, 10, size=(1,)),
            'velocity': np.random.uniform(-1, 1, size=(1,))
        }
        # Example reward, done, and info
        reward = 1.0
        done = False
        info = {}
        return observation, reward, done, info

    def reset(self):
        # Example initial observation
        observation = {
            'position': np.random.randint(0, 10, size=(1,)),
            'velocity': np.random.uniform(-1, 1, size=(1,))
        }
        return observation, {}


def run_logger(env, capacity=20):
    file_name = "logs/test.tfrecord"
    
    replay_buffer = ReplayBufferDataStore(
        env.observation_space,
        env.action_space,
        capacity=capacity,
    )

    # create some fake data
    sample_obs = env.reset()[0]
    action_shape = env.action_space.shape
    sample_action = np.random.randn(*action_shape)
    print("inserting data")
    for i in range(15): # arbitrary number of 15 samples
        replay_buffer.insert(
            dict(
                observations=sample_obs,
                next_observations=sample_obs,
                actions=sample_action,
                rewards=np.random.randn(),
                masks=1,
            )
        )

    print("inserted data", replay_buffer._insert_index)
    export_tfds(replay_buffer, file_name)

    # read the tfrecord file
    print("reading tfrecord file")
    dataset = read_tfds(file_name,
                        observation_space=env.observation_space,
                        action_space=env.action_space,)

    print(" dataset size", len(list(dataset)))
    dataset_size = len(list(dataset))
    assert dataset_size == min(capacity, 15)

    for data in dataset.take(1):
        print(data)

    print("\nnow read the tfrecord file into replay buffer")
    replay_buffer: ReplayBufferDataStore = make_replay_buffer(
        file_name,
        capacity=200,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    print("inserted data", replay_buffer._insert_index)
    assert replay_buffer._insert_seq_id == min(capacity, 15)


if __name__ == "__main__":
    env = CustomEnv()
    print(" testing custom env")
    run_logger(env)
    print("testing pendulum env")
    env = gym.make("Pendulum-v1")
    run_logger(env, capacity=9)
    print("all tests passed")
