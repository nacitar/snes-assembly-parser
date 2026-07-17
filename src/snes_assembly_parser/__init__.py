"""A library for parsing SNES assembly code.

Attributes:
    __version__: The installed distribution version.
"""

import importlib.metadata

from .segment import Segment, code, code_lines, data, datas, note, notes
from .source import Block, Line, Pool, Source, data_size, instruction_shape

try:
    __version__ = importlib.metadata.version(__package__)
except importlib.metadata.PackageNotFoundError:
    # Imported from a bare checkout on PYTHONPATH (not installed) -- e.g. a
    # consumer that clones this repo and uses the library directly.
    __version__ = "0.0.0+local"

__all__ = [
    "Block",
    "Line",
    "Pool",
    "Segment",
    "Source",
    "__version__",
    "code",
    "code_lines",
    "data",
    "data_size",
    "datas",
    "instruction_shape",
    "note",
    "notes",
]
