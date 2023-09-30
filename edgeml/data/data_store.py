#!/usr/bin/env python3

from __future__ import annotations
from typing import Any, Dict, List, Tuple, Any
from collections import deque

from abc import abstractmethod

##############################################################################


class DataStoreBase:
    @abstractmethod
    def latest_data_id() -> Any:
        """Return the id """
        pass

    @abstractmethod
    def get_latest_data(self, from_id: Any) -> Tuple[list, dict]:
        """return the indices and data from id"""
        pass

    @abstractmethod
    def update_data(self, indices: list, data: dict):
        """with the indices and data, update the latest data"""
        pass
    
    @abstractmethod
    def __len__(self):
        pass

##############################################################################

class QueuedDataStore(DataStoreBase):
    """A simple queue-based data store."""

    def __init__(self, capacity: int):
        self.queue = deque(maxlen=capacity)
        self.latest_seq_id = -1

    def latest_data_id(self) -> int:
        return self.latest_seq_id

    def insert(self, data: Any):
        self.latest_seq_id += 1
        self.queue.append((self.latest_seq_id, data))

    def get_latest_data(self, from_id: int) -> Tuple[List[int], Dict]:
        indices = []
        output_data = {"seq_id": [], "data": []}

        for idx, (seq_id, data) in enumerate(self.queue):
            if seq_id > from_id:
                indices.append(idx)
                output_data["seq_id"].append(seq_id)
                output_data["data"].append(data)

        return indices, output_data

    def update_data(self, indices: List[int], data: Dict):
        for idx in indices: # assume indices are sorted
            if idx < len(self.queue):
                self.queue[idx] = (data["seq_id"], data["data"])
            else:
                self.queue.append((data["seq_id"], data["data"]))
        # assume the last one is the latest
        if len(self.queue) > 0:
            self.latest_seq_id = self.queue[-1][0]

    def __len__(self):
        return len(self.queue)
