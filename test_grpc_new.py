import numpy as np

from google.rpc import code_pb2
from google.rpc import status_pb2
from dm_env_rpc.v1 import connection as dm_env_rpc_connection
from dm_env_rpc.v1 import dm_env_adaptor
from dm_env_rpc.v1 import dm_env_rpc_pb2
from dm_env_rpc.v1 import dm_env_rpc_pb2_grpc
from dm_env_rpc.v1 import spec_manager
from dm_env_rpc.v1 import tensor_spec_utils
from dm_env_rpc.v1 import tensor_utils
import grpc
from time import sleep


_FRAMES_PER_SEC = 50
_FRAME_DELAY_MS = int(1000.0 // _FRAMES_PER_SEC)

_ACTION_NOTHING = 0
_ACTION_LEFT = 1
_ACTION_RIGHT = 2
_ACTION_FORWARD = 3
_ACTION_BACKWARD = 4

_ACTION_PADDLE = 'paddle'
_ACTION_JUMP = 'jump'

_OBSERVATION_CAMERA = 'Camera'
_OBSERVATION_DONE = 'Collided'


def main():

    port = 30051
    timeout = 10
    
    with grpc.insecure_channel(f'localhost:{port}') as channel:
        grpc.channel_ready_future(channel).result(timeout)
        connection = dm_env_rpc_connection.Connection(channel)
        print(connection)

        print("create and join world")
        env, world_name = dm_env_adaptor.create_and_join_world(
        connection, create_world_settings={}, join_world_settings={})
        print("joined world:", world_name)

        with env:
            keep_running = True
            while keep_running:
                requested_action = _ACTION_NOTHING
                is_jumping = 0
                requested_action = _ACTION_BACKWARD
                actions = {_ACTION_PADDLE: [requested_action],
                           _ACTION_JUMP: [is_jumping]}
                timestep = env.step(actions)
                obs = timestep.observation[_OBSERVATION_CAMERA]
                reward = timestep.reward
                done = timestep.observation[_OBSERVATION_DONE]
                if done:
                    obs = env.reset()
                    keep_running = False
        connection.send(dm_env_rpc_pb2.DestroyWorldRequest(world_name=world_name))

if __name__ == "__main__":
    main()
