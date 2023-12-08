
from edgeml.data.tfds import read_tfds, export_tfds, ReplayBufferDataStore
import gym
from gym import spaces
import numpy as np

class CustomEnv(gym.Env):
    """
    A custom environment that uses a dictionary for the observation space.
    """

    def __init__(self):
        super(CustomEnv, self).__init__()

        # Define action space (for example, two discrete actions)
        self.action_space = spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)

        # Define observation space using a dictionary
        self.observation_space = spaces.Dict({
            # 'position': spaces.Discrete(10),     # Example: position as a discrete value
            # Example: position as a continuous value
            'position': spaces.Box(low=0, high=10, shape=(1,), dtype=np.int32),
            # Example: velocity as a continuous value
            'velocity': spaces.Box(low=-1, high=1, shape=(1,), dtype=np.float32)
        })

    def step(self, action):
        # Implement the logic for one step in the environment
        # For example, update position and velocity based on the action
        # ...

        # Example observation
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
        # Reset the state of the environment to an initial state
        # ...

        # Example initial observation
        observation = {
            'position': np.random.randint(0, 10, size=(1,)),
            'velocity': np.random.uniform(-1, 1, size=(1,))
        }
        return observation, {}


def run_logger(env, capacity=20):
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
    for i in range(15):
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
    export_tfds(replay_buffer, "logs/test2.tfrecord")

    # read the tfrecord file
    print("reading tfrecord file")
    dataset = read_tfds("logs/test2.tfrecord",
                        observation_space=env.observation_space,
                        action_space=env.action_space,)
    
    print( " dataset size", len(list(dataset)))
    for data in dataset.take(1):
        print(data)

    print("\nnow read the tfrecord file into replay buffer")
    replay_buffer = ReplayBufferDataStore.make_from_tfds(
        "logs/test2.tfrecord",
        capacity=200,
        observation_space=env.observation_space,
        action_space=env.action_space,
    )
    print("inserted data", replay_buffer._insert_index)
    assert replay_buffer._insert_index == min(capacity, 15)


if __name__ == "__main__":
    env = CustomEnv()
    print(" testing custom env")
    run_logger(env)
    print("testing pendulum env")
    env = gym.make("Pendulum-v1", 9)
    run_logger(env)
    print("all tests passed")
