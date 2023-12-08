from __future__ import annotations
from gym import spaces
import os
from typing import List

import tensorflow as tf
import numpy as np
import gym
from jaxrl_m.data.replay_buffer import ReplayBuffer
from edgeml.data.data_store import DataStoreBase
from threading import Lock


##############################################################################


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._insert_seq_id = 0 # keeps increasing
        self._lock = Lock()

    @staticmethod
    def make_from_tfds(input_path: str,
                       capacity: int,
                       observation_space: gym.Space,
                       action_space: gym.Space,
                       ) -> ReplayBufferDataStore:
        """
        make a replay buffer datastore object from a tfrecord file
        """
        # read the tfrecord file
        dataset = tf.data.TFRecordDataset(input_path)
        # parse the tfrecord file
        replay_buffer = ReplayBufferDataStore(
            observation_space=observation_space,
            action_space=action_space,
            capacity=capacity,
        )

        dataset = read_tfds(input_path, observation_space, action_space)
        replay_buffer.batch_insert(list(dataset))
        return replay_buffer

    # ensure thread safety

    def insert(self, *args, **kwargs):
        with self._lock:
            self._insert_seq_id += 1
            super(ReplayBufferDataStore, self).insert(*args, **kwargs)

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(ReplayBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_seq_id

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO

    def end_trajectory(self):
        raise NotImplementedError  # TODO
        # TODO: compute RB indices for latest trajectory...
        indices = ...
        trajectory = jax.tree_map(lambda x: x[indices], self._buffer)
        output_path = ...
        self.save_tfds(trajectory, output_path)
        # End the trajectory
        ReplayBuffer.end_trajectory(self)

##############################################################################


def tensor_feature(value):
    """Serialize a tensor to tf.Example."""
    return tf.train.Feature(
        bytes_list=tf.train.BytesList(value=[tf.io.serialize_tensor(value).numpy()])
    )

def export_tfds(replay_buffer: ReplayBufferDataStore, output_path: str):
    """
    Export the replay buffer to a *.tfrecord file
    : args:
        replay_buffer: a ReplayBufferDataStore object
        output_path: the path to the output file

    The current data dict structure is:
        dataset_dict = dict(
            observations=observation_data,
            next_observations=next_observation_data,
            actions=np.empty((capacity, *action_space.shape), dtype=action_space.dtype),
            rewards=np.empty((capacity,), dtype=np.float32),
            masks=np.empty((capacity,), dtype=bool),
        )
    """
    data = replay_buffer.dataset_dict
    capacity = replay_buffer._capacity
    curr_insert_seq_id = replay_buffer._insert_seq_id

    # get the directory of the output path
    output_dir = os.path.dirname(output_path)
    os.makedirs(output_dir, exist_ok=True)

    def get_obs_tensor(data, idx):
        obs = data["observations"]
        next_obs = data["next_observations"]
        obs_dict = {}
        if isinstance(obs, np.ndarray):
            obs_dict["observations"] = tensor_feature(obs[idx])
            obs_dict["next_observations"] = tensor_feature(next_obs[idx])
        elif isinstance(obs, dict):
            # create a key which is observation/key, e.g. observation/position
            for k, v in obs.items():
                obs_dict[f"observations/{k}"] = tensor_feature(v[idx])
            for k, v in next_obs.items():
                obs_dict[f"next_observations/{k}"] = tensor_feature(v[idx])
        else:
            raise TypeError(f"Unsupported observation space type: {type(obs)}")
        return obs_dict

    with tf.io.TFRecordWriter(output_path) as writer:
        for i in range(capacity):
            # Handle ring buffer, i.e. the buffer is not full and
            # when the list is full and points to the beginning
            if i >= curr_insert_seq_id:
                break
            if curr_insert_seq_id >= capacity:
                i = (curr_insert_seq_id + i) % capacity

            _obs_tensor = get_obs_tensor(data, i)

            feature_dict = {
                "actions": tensor_feature(data["actions"][i]),
                "rewards": tensor_feature(data["rewards"][i]),
                "masks": tensor_feature(data["masks"][i]),
            }
            feature_dict.update(_obs_tensor)  # append the obs tensor
            example = tf.train.Example(
                features=tf.train.Features(
                    feature=feature_dict
                )
            )
            writer.write(example.SerializeToString())
    print("Done writing tfrecord file to ", output_path)

##############################################################################


def read_tfds(input_path: str,
              observation_space: gym.Space,
              action_space: gym.Space,):
    dataset = tf.data.TFRecordDataset(input_path)
    """
    Utility function to get the dataset from the tfrecord file
    """
    # print("size of dataset", len(list(dataset)))
    def _parse_function(proto):
        """This function parses a single tf.Example proto."""

        # Handle different observation space types
        obs_dict = {}
        if isinstance(observation_space, gym.spaces.Dict):
            observation_keys = observation_space.spaces.keys()
            # create dictionary of "observations/key" and "next_observations/key
            for k in observation_keys:
                obs_dict[f"observations/{k}"] = tf.io.FixedLenFeature([], tf.string)
                obs_dict[f"next_observations/{k}"] = tf.io.FixedLenFeature([], tf.string)
        else:
            obs_dict["observations"] = tf.io.FixedLenFeature([], tf.string)
            obs_dict["next_observations"] = tf.io.FixedLenFeature([], tf.string)

        feature_description = {
            'actions': tf.io.FixedLenFeature([], tf.string),
            'rewards': tf.io.FixedLenFeature([], tf.string),
            'masks': tf.io.FixedLenFeature([], tf.string),
        }
        feature_description.update(obs_dict)
        parsed_features = tf.io.parse_single_example(proto, feature_description)

        # Deserialize the tensors to original data types
        new_features = {}

        # Handle different observation space types
        if isinstance(observation_space, gym.spaces.Dict):
            observation_keys = observation_space.spaces.keys()
            new_features['observations'] = {}
            new_features['next_observations'] = {}
            for k in observation_keys:
                obs_dtype = observation_space.spaces[k].dtype
                new_features[f"observations"][k] = tf.io.parse_tensor(
                    parsed_features[f"observations/{k}"], out_type=obs_dtype)
                new_features[f"next_observations"][k] = tf.io.parse_tensor(
                    parsed_features[f"next_observations/{k}"], out_type=obs_dtype)
        else:
            obs_dtype = observation_space.dtype
            new_features['observations'] = tf.io.parse_tensor(parsed_features['observations'], out_type=obs_dtype)
            new_features['next_observations'] = tf.io.parse_tensor(
                parsed_features['next_observations'], out_type=obs_dtype)

        act_dtype = action_space.dtype
        new_features['actions'] = tf.io.parse_tensor(parsed_features['actions'], out_type=act_dtype)
        new_features['rewards'] = tf.io.parse_tensor(parsed_features['rewards'], out_type=tf.float32)
        new_features['masks'] = tf.io.parse_tensor(parsed_features['masks'], out_type=tf.bool)
        return new_features

    # Map the parsing function over the dataset
    return dataset.map(_parse_function)
