#!/usr/bin/env python3

from typing import Any, Optional, Callable, Set, Dict, List
from typing_extensions import Protocol
from collections import deque

from edgeml.zmq_wrapper.req_rep import ReqRepServer, ReqRepClient
from edgeml.zmq_wrapper.broadcast import BroadcastServer, BroadcastClient
from edgeml.internal.utils import compute_hash

import threading
import logging
from pydantic import BaseModel

##############################################################################

class EdgeConfig(BaseModel):
    # Client and server should have the same config
    port_number: int
    action_keys: List[str]
    observation_keys: List[str]
    broadcast_port: Optional[int] = None

class ObsCallbackProtocol(Protocol):
    """
    Define the data type of the callback function
    :param keys: Set of keys of the observation, defaults to all keys
    :return: Response to send to the client
    """
    def __call__(self, keys: Set) -> Dict:
        ...

class ActCallbackProtocol(Protocol):
    """
    Define the data type of the callback function
    :param key: Key of the action
    :param payload: Payload related to the action in dict format
    :return: Response to send to the client
    """
    def __call__(self, key: str, payload: Dict) -> Dict:
        ...

##############################################################################

class EdgeServer:
    def __init__(self,
                 config: EdgeConfig,
                 obs_callback: Optional[ObsCallbackProtocol],
                 act_callback: Optional[ActCallbackProtocol],
                 log_level=logging.DEBUG):
        """
        :param config: Config object
        :param obs_callback: Callback function when client requests observation
        :param act_callback: Callback function when client requests action
        """
        def obs_parser_cb(payload: dict) -> dict:
            if payload["type"] == "obs" and obs_callback is not None:
                return obs_callback(payload["keys"])
            elif payload["type"] == "act" and act_callback is not None:
                return act_callback(payload["key"], payload["payload"])
            elif payload["type"] == "hash":
                return {"payload": compute_hash(config.json())}
            return {"status": "error", "message": "Invalid payload"}

        self.config = config
        self.server = ReqRepServer(config.port_number, obs_parser_cb)
        logging.basicConfig(level=log_level)    
        logging.debug(f"Edge server is listening on port {config.port_number}")

        if config.broadcast_port is not None:
            self.broadcast = BroadcastServer(config.broadcast_port)
            logging.debug(f"Broadcast server is on port {config.broadcast_port}")

    def start(self, threaded: bool = False):
        """
        Starts the server, defaulting to blocking mode
        :param threaded: Whether to start the server in a separate thread
        """
        if threaded:
            self.thread = threading.Thread(target=self.server.run)
            self.thread.start()
        else:
            self.server.run()

    def publish_obs(self, payload: dict) -> bool:
        """
        Publishes the observation to the broadcast server,
        This only works if the broadcast server is initialized 
        via `Config.broadcast_port`
        """
        if self.config.broadcast_port is None:
            logging.warning("Broadcast server not initialized")    
            return False
        self.broadcast.broadcast(payload)
        return True

##############################################################################

class EdgeClient:
    def __init__(self, server_ip: str, config: EdgeConfig):
        self.client = ReqRepClient(server_ip, config.port_number)
        # Check hash of server config to ensure compatibility
        res = self.client.send_msg({"type": "hash"})
        if res is None:
            raise Exception("Failed to connect to server")
        local_hash = compute_hash(config.json())
        if local_hash != res["payload"]:
            raise Exception(
                f"Incompatible config with hash with server. "
                "Please check the config of the server and client")

        # use hash for faster lookup. config uses list because it is
        # used for hash md4 comparison
        config.observation_keys = set(config.observation_keys)
        config.action_keys = set(config.action_keys)
        self.config = config
        self.server_ip = server_ip
        print("Done init client")

    def obs(self, keys: Optional[Set[str]]=None) -> dict:
        """
        Returns the observation from the server
        :param keys: Set of keys to request from the server, defaults to all keys
        :return: Observation from the server
        """
        if keys is not None:
            for key in keys:
                assert key in self.config.observation_keys,\
                    f"Invalid observation key: {key}"
        messsage = {"type": "obs", "keys": keys}
        return self.client.send_msg(messsage)

    def act(self, key:str, payload: dict={}) -> dict:
        """
        Sends the action to the server
        :param key: Key of the action
        :param payload: Payload to send to the server
        :return: Response from the server
        """
        if key not in self.config.action_keys:
            raise Exception(f"Invalid action key: {key}")
        message = {"type": "act", "key": key, "payload": payload}
        return self.client.send_msg(message)

    def register_obs_callback(self, callback: Callable):
        """
        Registers the callback function for observation streaming
        :param callback: Callback function for observation streaming
        """
        if self.config.broadcast_port is None:
            raise Exception("Broadcast server not initialized")
        self.broadcast_client = BroadcastClient(
            self.server_ip, self.config.broadcast_port)
        self.broadcast_client.async_start(callback)

##############################################################################

class InterferenceServer:
    def __init__(self, port_num: int):
        """
        Create a server with the given port number and interface names
        """
        def parser_cb(payload: dict) -> dict:
            interface_name = payload.get('interface')
            if interface_name and interface_name in self.interfaces:
                return self.interfaces[interface_name](payload)
            return {"status": "error", "message": "Invalid interface or payload"}

        self.server = ReqRepServer(port_num, parser_cb)
        self.interfaces = {}

    def start(self, threaded: bool = False):
        """
        Starts the server, defaulting to blocking mode
        :param threaded: Whether to start the server in a separate thread
        """
        if threaded:
            self.thread = threading.Thread(target=self.server.run)
            self.thread.start()
        else:
            self.server.run()
    
    def register_interface(self, name: str, callback: Callable):
        """
        Registers the callback function for the interface
        :param name: Name of the interface
        :param callback: Callback function for the interface
        """
        self.interfaces[name] = callback

##############################################################################

class InterferenceClient:
    def __init__(self, server_ip: str, port_num: int):
        self.client = ReqRepClient(server_ip, port_num)
    
    def interfaces(self) -> Set[str]:
        """
        Returns the set of interfaces available on the server
        :return: Set of interfaces available on the server
        """
        response = self.client.send_msg({"type": "list_interfaces"})
        if response:
            return set(response.get('interfaces', []))
        return set()

    def call(self, name: str, payload: dict) -> Optional[dict]:
        """
        Calls the interface on the server with the given payload
        :param name: Name of the interface
        :param payload: Payload to send to the interface
        :return: Response from the interface
        """
        return self.client.send_msg(
            {"type": "call_interface", "interface": name, "payload": payload})

##############################################################################

class TrainerConfig:
    port_number: int
    broadcast_port: int
    queue_size: int = 1

class TrainerCallbackProtocol(Protocol):
    """
    :param payload: Payload to send to the trainer, e.g. training data
    :return: Response to send to the client, can return updated weights.
    """
    def __call__(self, payload: dict) -> Optional[dict]:
        ...

class TrainerServer:
    def __init__(self,
                 config: TrainerConfig,
                 train_callback: TrainerCallbackProtocol,
                 log_level=logging.DEBUG):
        """
        :param config: Config object
        :param train_callback: Callback function when client requests training
        :param act_callback: Callback function when client requests action
        """
        self.queue = deque() # FIFO queue

        def __callback_impl(payload: dict) -> dict:
            if len(self.queue) >= config.queue_size:
                self.queue.popleft()
            self.queue.append(payload)
            return train_callback(payload)

        self.req_rep_server = ReqRepServer(config.port_number, __callback_impl)

    def get_data(self) -> List[dict]:
        """
        Returns the list of training data, max size is `config.queue_size`
        """
        return list(self.queue)

    def publish_weights(self, payload: dict):
        """
        Publishes the weights to the broadcast server,
        This only works if the broadcast server is initialized 
        via `Config.broadcast_port`
        """
        self.broadcast_server.broadcast(payload)

    def start(self, threaded: bool = False):
        """
        Starts the server, defaulting to blocking mode
        :param threaded: Whether to start the server in a separate thread
        """
        self.req_rep_server.start(threaded=threaded)

##############################################################################

class TrainerClient:
    def __init__(self, server_ip: str, config: TrainerConfig):
        self.req_rep_client = ReqRepClient(server_ip, config.port_number)
        self.server_ip = server_ip
        self.config = config
        print(f"Initiated trainer client at {server_ip}:{config.port_number}")

    def train_step(self, payload: dict) -> Optional[dict]:
        """
        Performs a training step with the given payload.
        NOTE: This is a blocking call. If the trainer needs more time to process,
            use the publish_weights() method in the server instead.
        :param payload: Payload to send to the trainer
        :return: Response from the trainer
        """
        return self.req_rep_client.send_msg(payload)
    
    def recv_weights_callback(self, callback: Callable[[dict], None]):
        """Registers the callback function for receiving weights"""
        self.broadcast_client = BroadcastClient(
            self.server_ip, self.config.broadcast_port)
        self.broadcast_client.async_start(callback)