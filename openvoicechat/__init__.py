"""OpenVoiceChat: the local voice runtime for Python.

Nothing is re-exported here, so importing the package never loads numpy,
onnxruntime, or torch. Import the piece you need:

    from openvoicechat.config import Config
    from openvoicechat.conversation import Conversation
    from openvoicechat.local_speech import LocalSpeech
"""

from importlib.metadata import PackageNotFoundError, version

try:
    __version__ = version("openvoicechat")
except PackageNotFoundError:  # running from a source checkout
    __version__ = "0.4.1"

__all__ = ["__version__"]
