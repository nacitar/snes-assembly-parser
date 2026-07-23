"""Structured parsing of the disassembly's address-anchor labels.

The spannerisms disassembly annotates most byte-emitting lines with an anchor
label that carries the line's ROM offset:

* ``#_0EC440``            -- a plain (live) address anchor.
* ``#_0EC440o``           -- with a lowercase APU-bank tag (``o``/``u``/``c``).
* ``NULL_0EEDFB``         -- free ROM (an ``Indicator.NULL`` region marker).
* ``UNREACHABLE_0ED3CF``  -- dead code (an ``Indicator.UNREACHABLE`` marker).
* ``#UNREACHABLE_...``    -- an address-transparent (``#``) marker variant.

:class:`Address` parses all of these into one value, so callers ask
``Address.parse(label)`` instead of juggling several regexes. A *live* anchor
(``indicator is None``) is what participates in byte sizing; markers carry
their offset only as documentation of where the free/dead region sits. Named
``#`` labels that are not addresses (e.g. ``#Module0E_02_RenderText``) parse
to ``None`` -- they have no ``_<hex>`` body.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum, unique
from typing import ClassVar


@unique
class Indicator(StrEnum):
    """The role an anchored region plays, encoded in its label prefix."""

    NULL = "NULL"  #: free/unused ROM
    UNREACHABLE = "UNREACHABLE"  #: dead code, preserved byte-for-byte


@dataclass(frozen=True)
class Address:
    """A parsed address-anchor label: indicator, offset, tag, and hex width.

    ``offset`` is the ROM offset the anchor states; ``indicator`` is ``None``
    for a live anchor or the marker kind for a free/dead region; ``tag`` keeps
    any trailing lowercase APU-bank marker; ``width`` is the hex-digit count as
    written, so :meth:`render` reproduces the label exactly.
    """

    #: ``#_`` or ``#`` then an optional NULL/UNREACHABLE word, then ``_<hex>``
    #: and an optional lowercase tag. The leading ``#`` is optional so bare
    #: ``NULL_``/``UNREACHABLE_`` labels match too.
    PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"#?(?P<indicator>NULL|UNREACHABLE)?_(?P<hex>[0-9A-F]+)(?P<tag>[a-z]*)"
    )

    indicator: Indicator | None
    offset: int
    tag: str = ""
    width: int = 6

    @classmethod
    def parse(cls, label: str | None) -> Address | None:
        """The :class:`Address` a label encodes, or ``None`` if it is not one.

        A live anchor (``#_0EC440``) yields ``indicator=None``; a marker
        (``NULL_``/``UNREACHABLE_``) yields the matching :class:`Indicator`.
        Anything without an ``_<hex>`` body returns ``None``.
        """
        if label is None:
            return None
        match = cls.PATTERN.fullmatch(label)
        if match is None:
            return None
        indicator = (
            Indicator(match["indicator"]) if match["indicator"] else None
        )
        return cls(
            indicator=indicator,
            offset=int(match["hex"], 16),
            tag=match["tag"] or "",
            width=len(match["hex"]),
        )

    def at(self, offset: int) -> Address:
        """This anchor re-based to ``offset`` (kind, tag, and width kept)."""
        return Address(self.indicator, offset, self.tag, self.width)

    def render(self) -> str:
        """The exact label text for this address.

        A live anchor renders ``#_<hex><tag>``; a marker renders
        ``<INDICATOR>_<hex>`` (markers carry their address in the name and take
        no ``#`` or tag in the disassembly).
        """
        hexed = f"{self.offset:0{self.width}X}"
        if self.indicator is None:
            return f"#_{hexed}{self.tag}"
        return f"{self.indicator}_{hexed}"

    @property
    def is_anchor(self) -> bool:
        """Whether this is a live anchor (participates in byte sizing)."""
        return self.indicator is None

    @property
    def is_marker(self) -> bool:
        """Whether this marks a free (``NULL``) or dead (``UNREACHABLE``)
        region."""
        return self.indicator is not None
