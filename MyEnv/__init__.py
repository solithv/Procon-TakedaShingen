from gymnasium.envs.registration import register
from .game import *
from .worker import *

register(id="TaniJoh-v0", entry_point="MyEnv.game:Game")
__all__ = ["Game", "Worker"]
