"""A library for parsing SNES assembly code.

Attributes:
    __version__: The installed distribution version.
"""

import importlib.metadata

from .segment import Segment, code, data, note
from .source import Block, Line, Pool, Source, data_size

__version__ = importlib.metadata.version(__package__)

__all__ = [
    "Block",
    "Line",
    "Pool",
    "Segment",
    "Source",
    "__version__",
    "code",
    "data",
    "data_size",
    "note",
]
