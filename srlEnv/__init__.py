from srl.base.env import registration

from .game import *
from .worker import *

registration.register(
    id="TaniJoh-v1",
    entry_point=f"{__name__}.game:Game",
    kwargs={},
)
__all__ = ["Game", "Worker"]
