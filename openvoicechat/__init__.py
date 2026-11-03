"""OpenVoiceChat package. Submodules are imported lazily to keep startup cheap."""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("openvoicechat")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.4.1"

__all__ = ["VAD", "Config", "Conversation", "__version__"]


def __getattr__(name: str):
    if name == "Config":
        from .config import Config

        return Config
    if name == "Conversation":
        from .conversation import Conversation

        return Conversation
    if name == "VAD":
        from .vad import VAD

        return VAD
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
