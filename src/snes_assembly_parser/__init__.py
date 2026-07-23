"""A library for parsing and rewriting asar-compatible SNES assembly.

The unit everything is built around is :class:`Assembly` (one editable, sized,
addressable run of :class:`Line`); :class:`Rom` composes many of them into a
whole program. ``Source``/``Segment`` remain as the previous split and are kept
for compatibility.

Attributes:
    __version__: The installed distribution version.
"""

import importlib.metadata

from .address import Address, Indicator
from .assembly import (
    Assembly,
    data,
    datas,
    instruction,
    instructions,
    join,
    note,
    notes,
)
from .rom import Caller, Rom
from .segment import Segment, code, code_lines
from .sizing import (
    AnchorSizer,
    ComputedSizer,
    HybridSizer,
    Sizer,
    computed_size,
)
from .source import Block, Line, Pool, Source, data_size, instruction_shape

try:
    __version__ = importlib.metadata.version(__package__)
except importlib.metadata.PackageNotFoundError:
    # Imported from a bare checkout on PYTHONPATH (not installed) -- e.g. a
    # consumer that clones this repo and uses the library directly.
    __version__ = "0.0.0+local"

__all__ = [
    "Address",
    "AnchorSizer",
    "Assembly",
    "Block",
    "Caller",
    "ComputedSizer",
    "HybridSizer",
    "Indicator",
    "Line",
    "Pool",
    "Rom",
    "Segment",
    "Sizer",
    "Source",
    "__version__",
    "code",
    "code_lines",
    "computed_size",
    "data",
    "data_size",
    "datas",
    "instruction",
    "instruction_shape",
    "instructions",
    "join",
    "note",
    "notes",
]
