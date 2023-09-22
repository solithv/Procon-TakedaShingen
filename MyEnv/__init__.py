<<<<<<< HEAD
from gymnasium.envs.registration import register

from .game import *
from .worker import *

register(id="TaniJoh-v0", entry_point=f"{__name__}.game:Game")
=======
from .game import *
from .worker import *

>>>>>>> origin/3x3
__all__ = ["Game", "Worker"]
