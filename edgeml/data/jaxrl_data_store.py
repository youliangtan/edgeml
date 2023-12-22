# !/usr/bin/env python3
# NOTE: this requires jaxrl_m to be installed:
#       https://github.com/rail-berkeley/jaxrl_minimal

from __future__ import annotations

from threading import Lock
from typing import List, Optional
from edgeml.data.data_store import DataStoreBase
from edgeml.data.trajectory_buffer import TrajectoryBuffer, DataShape

import gym
import jax
from jaxrl_m.data.replay_buffer import ReplayBuffer
import tensorflow as tf

# import oxe_envlogger if it is installed
try:
    from oxe_envlogger.rlds_logger import RLDSLogger, RLDSStepType
except ImportError:
    print("rlds logger is not installed, install it if required: "
          "https://github.com/rail-berkeley/oxe_envlogger ")


##############################################################################

class TrajectoryBufferDataStore(TrajectoryBuffer, DataStoreBase):
    def __init__(
        self,
        capacity: int,
        data_shapes: List[DataShape],
        device: Optional[jax.Device] = None,
        seed: int = 0,
        min_trajectory_length: int = 2,
        rlds_logger: Optional[RLDSLogger] = None,
    ):
        TrajectoryBuffer.__init__(
            self,
            capacity=capacity,
            data_shapes=data_shapes,
            min_trajectory_length=min_trajectory_length,
            device=device,
            seed=seed,
        )
        DataStoreBase.__init__(self, capacity)
        self._lock = Lock()

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION # to init the state for restart
            self._logger = rlds_logger

    # ensure thread safety
    def insert(self, data):
        with self._lock:
            super(TrajectoryBufferDataStore, self).insert(data)

            if self._logger:
                # handle restart when it was done before
                if self.step_type == RLDSStepType.TERMINATION:
                    self.step_type = RLDSStepType.RESTART
                elif self.step_type == RLDSStepType.TRUNCATION:
                    self.step_type = RLDSStepType.RESTART
                elif not data["masks"]: # 0 is done, 1 is not done
                    self.step_type = RLDSStepType.TERMINATION
                elif data["end_of_trajectory"]:
                    self.step_type = RLDSStepType.TRUNCATION
                else:
                    self.step_type = RLDSStepType.TRANSITION

                self._logger(
                    action=data["actions"],
                    obs=data["observations"],
                    reward=data["rewards"],
                    step_type=self.step_type,
                )

    # ensure thread safety
    def sample(self, *args, **kwargs):
        with self._lock:
            return super(TrajectoryBufferDataStore, self).sample(*args, **kwargs)

    # NOTE: method for DataStoreBase
    def latest_data_id(self):
        return self._insert_index

    # NOTE: method for DataStoreBase
    def get_latest_data(self, from_id: int):
        raise NotImplementedError  # TODO


class ReplayBufferDataStore(ReplayBuffer, DataStoreBase):
    def __init__(
        self,
        observation_space: gym.Space,
        action_space: gym.Space,
        capacity: int,
        rlds_logger: Optional[RLDSLogger] = None,
    ):
        ReplayBuffer.__init__(self, observation_space, action_space, capacity)
        DataStoreBase.__init__(self, capacity)
        self._insert_seq_id = 0 # keeps increasing
        self._lock = Lock()
        self._logger = None

        if rlds_logger:
            self.step_type = RLDSStepType.TERMINATION # to init the state for restart
            self._logger = rlds_logger

    # ensure thread safety
    def insert(self, data):
        with self._lock:
            super(ReplayBufferDataStore, self).insert(data)
            self._insert_seq_id += 1

            # add data to the rlds logger
            # TODO: the current impl of ReplayBuffer doesn't support
            # proper truncation of the trajectory
            if self._logger:
                if self.step_type == RLDSStepType.TERMINATION:
                    self.step_type = RLDSStepType.RESTART
                elif not data["masks"]: # 0 is done, 1 is not done
                    self.step_type = RLDSStepType.TERMINATION
                else:
                    self.step_type = RLDSStepType.TRANSITION

                self._logger(
                    action=data["actions"],
                    obs=data["observations"],
                    reward=data["rewards"],
                    step_type=self.step_type,
                )

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

    def __del__(self):
        if self._logger:
            self._logger.close()
