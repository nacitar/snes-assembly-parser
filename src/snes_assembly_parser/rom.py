"""The :class:`Rom` -- a whole program assembled from many source files.

A :class:`Rom` is built from an entry file (``main.asm``), following every
``incsrc`` into a per-file :class:`~.assembly.Assembly` *unit* (cycle-checked).
It indexes every top-level function and pool across all units, and every
reference to them, so it can answer whole-program questions ("who calls this?")
that a single file cannot. That is what makes *hooking* automatic: freeing a JP
routine's name and pointing its callers at a relocated copy needs to know
whether any caller reaches it with a bank-local ``JSR`` (which cannot cross
banks) and so needs a landing-pad bridge.

Editing a routine is done on its unit's :class:`~.assembly.Assembly`; the
``Rom`` adds the operations that span files: :meth:`~Rom.rename` (label only,
callers untouched), :meth:`~Rom.hook`, free-space, and writing units back.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .assembly import Assembly
from .source import block_end

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

_INCSRC = re.compile(r'^\s*incsrc\s+"(?P<path>[^"]+)"')
# Control-transfer opcodes that are bank-local (a relative/same-bank call a
# landing pad must bridge). ``JSL``/``JML`` already cross banks and need none.
_BANK_LOCAL_CALLS = frozenset({"JSR", "BRA", "BRL", "JMP"})


@dataclass(frozen=True)
class Caller:
    """One reference to a symbol: the unit it lives in and the opcode used."""

    unit: Path
    opcode: str  # e.g. "JSR", "JSL", "dl", "dw"; "" for a bare operand ref

    @property
    def is_bank_local(self) -> bool:
        """Whether this is a same-bank call needing a pad to cross banks."""
        return self.opcode.upper() in _BANK_LOCAL_CALLS


@dataclass
class Rom:
    """A whole program: named :class:`~.assembly.Assembly` units + indexes.

    Build with :meth:`load`. ``units`` maps each source path to its parsed
    assembly; the function/pool/reference indexes are rebuilt lazily.
    """

    units: dict[Path, Assembly] = field(default_factory=dict)
    #: The entry file, and the source order the units were included in.
    entry: Path | None = None
    order: list[Path] = field(default_factory=list)

    # ---- construction ----
    @classmethod
    def load(cls, main_asm: Path) -> Rom:
        """Parse ``main_asm`` and every file it ``incsrc``s (recursively).

        Cycles and re-includes are detected: each file is loaded once. The
        entry file is a unit too (for its own ``org``/``db`` lines).
        """
        rom = cls(entry=main_asm)
        rom._include(main_asm, set())
        return rom

    def load_asm(self, path: Path) -> Assembly:
        """Load one more file as a unit and return it."""
        self._include(path, set())
        return self.units[path]

    def _include(self, path: Path, active: set[Path]) -> None:
        path = path.resolve()
        # Cycle check first: a file still on the active include stack is an
        # ancestor of itself (a true cycle). It is also already in ``units``,
        # so this must precede the dedup return or the cycle would be masked.
        if path in active:
            msg = f"incsrc cycle through {path}"
            raise ValueError(msg)
        if path in self.units:
            return  # already included elsewhere (diamond re-include)
        assembly = Assembly.from_path(path)
        self.units[path] = assembly
        self.order.append(path)
        active = active | {path}
        for line in assembly.lines:
            match = _INCSRC.match(str(line))
            if match:
                self._include(path.parent / match["path"], active)

    # ---- whole-program indexes ----
    def unit_of(self, name: str) -> Path:
        """The unit path that defines top-level ``name`` (function or pool)."""
        for path in self.order:
            unit = self.units[path]
            if name in unit.labels or name in unit.pools:
                return path
        msg = f"no top-level label or pool named {name!r}"
        raise KeyError(msg)

    def function(self, name: str, *, comments: bool = False) -> Assembly:
        """A fresh copy of function ``name`` (searched across all units)."""
        return self.units[self.unit_of(name)].function(name, comments=comments)

    @property
    def functions(self) -> list[str]:
        """Every top-level function name across all units, in include order."""
        return [
            name for path in self.order for name in self.units[path].functions
        ]

    def callers(self, name: str) -> list[Caller]:
        """Every reference to ``name`` across all units.

        A reference is an operand token equal to ``name`` (or ``name`` as the
        base of a pool-qualified ``name_sub`` token); the :class:`Caller`
        records the opcode so :meth:`hook` can tell a bank-local ``JSR`` from a
        cross-bank ``JSL`` or a data pointer.
        """
        token = re.compile(rf"(?<![\w.])({re.escape(name)})(?![\w])")
        return [
            Caller(path, line.opcode or "")
            for path in self.order
            for line in self.units[path].lines
            if line.opcode is not None
            and any(token.search(arg) for arg in line.arguments)
        ]

    # ---- editing that spans files ----
    def rename(self, old: str, new: str) -> None:
        """Rename the *definition* of ``old`` to ``new``, callers untouched.

        The label ``old:`` (and an ``#old:`` transparent form, and a ``pool
        old`` directive) becomes ``new`` in its unit; every ``JSR old`` /
        ``dl old`` elsewhere is left alone. This is the freeing half of a hook
        (not a refactor -- use :meth:`~.assembly.Assembly.suffix` for that).
        """
        self.units[self.unit_of(old)].rename_label(old, new)

    def rename_all(self, pairs: Iterable[tuple[str, str]]) -> None:
        """Apply a batch of :meth:`rename` ``(old, new)`` pairs."""
        for old, new in pairs:
            self.rename(old, new)

    # ---- free space ----
    def free_regions(self) -> list[tuple[Path, int, int]]:
        """Every free-ROM (``NULL_``) region as ``(unit, address, bytes)``.

        Sized from the region's own ``#_`` byte lines, so an allocator knows
        exactly how much room each hole has.
        """
        regions: list[tuple[Path, int, int]] = []
        for path in self.order:
            unit = self.units[path]
            for index, line in enumerate(unit.lines):
                if not line.is_null_label:
                    continue
                end = block_end(unit.lines, index)
                span = unit.lines[index:end]
                address = next(
                    (ln.address for ln in span if ln.address is not None), None
                )
                if address is not None:
                    regions.append(
                        (path, address, sum(ln.size for ln in span))
                    )
        return regions

    # ---- output ----
    def write(self, remap: dict[Path, Path] | None = None) -> None:
        """Write every unit back to its path (or a remapped one).

        ``remap`` overrides individual output paths (e.g. write the patched
        pristine banks into the working tree). Units are written verbatim, so
        an unedited unit round-trips exactly.
        """
        remap = remap or {}
        for path, unit in self.units.items():
            unit.write(remap.get(path, path))
