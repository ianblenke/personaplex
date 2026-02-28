from livekit.agents import Plugin

from .log import logger
from .realtime_model import MoshiRealtimeModel, MoshiRealtimeSession
from .version import __version__

__all__ = [
    "MoshiRealtimeModel",
    "MoshiRealtimeSession",
    "__version__",
]


class MoshiPlugin(Plugin):
    def __init__(self) -> None:
        super().__init__(
            title="Moshi",
            version=__version__,
            package="livekit-plugins-moshi",
            logger=logger,
        )

    def download_files(self) -> None:
        pass


Plugin.register_plugin(MoshiPlugin())
