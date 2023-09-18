from gymnasium.envs.registration import register

from .Field_API import *
from .game import *
from .worker import *

register(id="TaniJoh-v0", entry_point=f"{__name__}.game:Game")
__all__ = ["Game", "Worker", "API"]
