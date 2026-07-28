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
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from .assembly import Assembly
from .source import block_end

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from .assembly import Entry
    from .hooking import LandingPad
    from .patcher import Patcher
    from .source import Block, Pool


class Placed(Protocol):
    """A body pinned at a ROM address (e.g. ``graft.Placement``).

    ``org`` gives the absolute address (so :meth:`Rom.write` can group pieces
    by bank); :meth:`render` emits the piece's own source text (the ``org``
    directive, an optional note, and the body) -- so the emitted form lives
    with the piece, not here.
    """

    org: int

    def render(self) -> str: ...


class Relocatable(Protocol):
    """Anything yielding :class:`Placed` pieces (e.g. ``graft.Relocation``)."""

    def placements(self) -> Sequence[Placed]: ...


_INCSRC = re.compile(r'^\s*incsrc\s+"(?P<path>[^"]+)"')
# Control-transfer opcodes that are bank-local (a relative/same-bank call a
# landing pad must bridge). ``JSL``/``JML`` already cross banks and need none.
_BANK_LOCAL_CALLS = frozenset({"JSR", "BRA", "BRL", "JMP"})


@dataclass(frozen=True)
class Caller:
    """One reference to a symbol: where it lives, the opcode, its block."""

    unit: Path
    opcode: str  # e.g. "JSR", "JSL", "dl", "dw"; "" for a bare operand ref
    #: The top-level block the reference sits in (``None`` if it is loose at a
    #: bank's top). Lets a hook tell an in-subsystem caller (which relocates
    #: with it) from one that stays behind.
    block: str | None = None

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
    #: Relocated pieces added by :meth:`add`, materialised into ``bank_XX.asm``
    #: units by :meth:`write` (grouped by ``org >> 16``).
    _relocated: list[Placed] = field(default_factory=list)

    def copy(self) -> Rom:
        """An independent working copy (units deep-copied, ready to mutate)."""
        return Rom(
            units={path: unit.copy() for path, unit in self.units.items()},
            entry=self.entry,
            order=list(self.order),
            _relocated=list(self._relocated),
        )

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
        """The unit path that defines ``name`` (function, pool, or ``#name``).

        Matches a plain ``name:`` label, the address-transparent ``#name:``
        form (same symbol, still emits its address -- so it is not in the
        top-level label index and must be found by scanning), or a ``pool
        name``.
        """
        marker = f"#{name}"
        for path in self.order:
            unit = self.units[path]
            if name in unit.pools:
                return path
            for line in unit.lines:
                if line.label in (name, marker) or (
                    line.opcode == "pool" and line.arguments == [name]
                ):
                    return path
        msg = f"no top-level label or pool named {name!r}"
        raise KeyError(msg)

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
        records the opcode (so a bank-local ``JSR`` is told from a cross-bank
        ``JSL`` or a data pointer) and the top-level block it sits in (scanning
        every line, so a reference outside any function is found too).
        """
        token = re.compile(rf"(?<![\w.])({re.escape(name)})(?![\w])")
        found: list[Caller] = []
        for path in self.order:
            block: str | None = None
            for line in self.units[path].lines:
                if line.is_top_level_label and line.label is not None:
                    block = line.label
                if line.opcode is not None and any(
                    token.search(arg) for arg in line.arguments
                ):
                    found.append(Caller(path, line.opcode, block))
        return found

    def needs_landing_pad(
        self, name: str, *, relocated: Iterable[str] = ()
    ) -> bool:
        """Whether hooking ``name`` needs a landing pad (not a bare alias).

        A relocated routine sits a bank away, so a same-bank caller
        (``JSR``/``BRL``/...) can only reach it if that caller *also* moves to
        the same bank -- i.e. its block is among ``relocated`` (the source
        blocks this hook's relocation carries with it). A pad is needed exactly
        when some bank-local caller stays behind: its block is not relocated
        (or it is loose at a bank's top). Cross-bank callers (``JSL``, data
        pointers) always reach the copy directly, so they never force a pad.
        """
        kept = frozenset(relocated)
        return any(
            caller.is_bank_local and caller.block not in kept
            for caller in self.callers(name)
        )

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

    def hook(self, name: str, *, comment: Iterable[str] = ()) -> None:
        """Free ``name`` (prefix ``UNREACHABLE_``) in its defining unit.

        This is the freeing half of an interception: the original body stays
        byte-identical under its new dead-code name, the bare name is left for
        a relocated copy to claim, and same-bank ``JSR`` callers -- which
        cannot reach that copy across banks -- are found with :meth:`callers`,
        so the driver knows where a landing pad is still needed. ``comment``
        lines go above the freed definition.
        """
        from .patcher import Patcher

        Patcher(self.units[self.unit_of(name)]).free(name, comment=comment)

    # ---- extraction (by name, from the owning unit) ----
    @property
    def pool_names(self) -> set[str]:
        """Every ``pool`` name across all units."""
        return {name for path in self.order for name in self.units[path].pools}

    @property
    def label_names(self) -> set[str]:
        """Every top-level label across all units."""
        return {
            name for path in self.order for name in self.units[path].labels
        }

    def _unit(self, name: str) -> Assembly:
        return self.units[self.unit_of(name)]

    def block(self, name: str, *, comments: bool = False) -> Assembly:
        """A fresh copy of block ``name`` (searched across all units)."""
        return self._unit(name).block(name, comments=comments)

    def pool(self, name: str, *, comments: bool = False) -> Assembly:
        """A fresh copy of pool ``name`` (searched across all units)."""
        return self._unit(name).pool(name, comments=comments)

    def blocks_until(self, name: str) -> Assembly:
        """The run of blocks up to (excluding) ``name``, from its unit."""
        return self._unit(name).blocks_until(name)

    def region_at(self, start: int, stop: int) -> Assembly:
        """A fresh copy of the ``[start, stop)`` address range, from the unit
        holding ``start`` -- the fallback for an unlabelled fragment (prefer
        :meth:`block`/:meth:`subblock` when a label exists)."""
        return self.units[self.unit_at(start)].region_at(start, stop)

    def extract(
        self,
        roots: str | Iterable[str],
        *,
        recursive: bool,
        external: Iterable[str] = (),
        comments: bool = False,
        gap_notes: dict[str, str] | None = None,
    ) -> Assembly:
        """Recursively pull ``roots`` + their references from the owning unit.

        The roots must be co-located in one unit (ALTTP subsystems are), so the
        closure is taken within it -- the same as loading that bank and calling
        :meth:`~.assembly.Assembly.extract`, but by name, no file named.
        """
        names = (roots,) if isinstance(roots, str) else tuple(roots)
        return self._unit(names[0]).extract(
            names,
            recursive=recursive,
            external=external,
            comments=comments,
            gap_notes=gap_notes,
        )

    def closure(
        self,
        roots: str | Iterable[str],
        *,
        recursive: bool,
        external: Iterable[str] = (),
    ) -> list[Block | Pool]:
        """The source-ordered closure of ``roots``, from the owning unit."""
        names = (roots,) if isinstance(roots, str) else tuple(roots)
        return self._unit(names[0]).closure(
            names, recursive=recursive, external=external
        )

    def concat(
        self,
        items: Iterable[Entry],
        *,
        comments: bool = False,
        gap_notes: dict[str, str] | None = None,
    ) -> Assembly:
        """Concatenate the named blocks/pools (from their unit) in order.

        Each item is a :class:`Block`, a :class:`Pool`, or a bare ``str``
        (a block name); the owning unit is the one defining the first item.
        """
        items = list(items)
        first = items[0]
        name = first if isinstance(first, str) else first.name
        return self._unit(name).concat(
            items, comments=comments, gap_notes=gap_notes
        )

    # ---- whole-program edits (routed by name/address to the owning unit) ----
    def unit_at(self, address: int) -> Path:
        """The unit whose lines carry the ``#_<hex>`` anchor at ``address``."""
        for path in self.order:
            if any(line.address == address for line in self.units[path].lines):
                return path
        msg = f"no line anchored at ${address:06X}"
        raise KeyError(msg)

    def _patcher_at(self, address: int) -> Patcher:
        from .patcher import Patcher

        return Patcher(self.units[self.unit_at(address)])

    def set_operand(
        self, address: int, instruction: str, *, comment: str | None = None
    ) -> None:
        """Rewrite the instruction at ``address`` (finds the bank itself)."""
        self._patcher_at(address).set_operand(
            address, instruction, comment=comment
        )

    def rewrite_reference(self, address: int, old: str, new: str) -> None:
        """Retarget operand ``old`` -> ``new`` at ``address``."""
        self._patcher_at(address).rewrite_reference(address, old, new)

    def relocate_block(
        self,
        address: int,
        jml_target: str,
        *,
        resume: int,
        orphan: tuple[int, ...] = (),
        comment: Iterable[str] = (),
    ) -> None:
        """Redirect the inline block at ``address`` to ``jml_target``."""
        self._patcher_at(address).relocate_block(
            address, jml_target, resume=resume, orphan=orphan, comment=comment
        )

    def insert_before(self, locator: str | int, lines: Iterable[str]) -> None:
        """Insert ``lines`` above label/anchor ``locator`` (finds its unit)."""
        from .patcher import Patcher

        path = (
            self.unit_at(locator)
            if isinstance(locator, int)
            else self.unit_of(locator)
        )
        Patcher(self.units[path]).insert_before(locator, lines)

    def landing_pads(
        self,
        free_label: str,
        pads: list[LandingPad],
        *,
        header: Iterable[str] = (),
    ) -> None:
        """Fill the ``NULL_`` region ``free_label`` with forwarding stubs."""
        from .patcher import Patcher

        Patcher(self.units[self.unit_of(free_label)]).landing_pads(
            free_label, pads, header=header
        )

    # ---- adding relocated code + writing the whole program ----
    def add(self, relocation: Relocatable) -> None:
        """Fold a relocation's placed pieces in; :meth:`write` banks them."""
        self._relocated.extend(relocation.placements())

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
                    (
                        line.address
                        for line in span
                        if line.address is not None
                    ),
                    None,
                )
                if address is not None:
                    regions.append(
                        (path, address, sum(line.size for line in span))
                    )
        return regions

    # ---- output ----
    def write(
        self, out_dir: Path, *, bank_header: Callable[[int], str] | None = None
    ) -> list[str]:
        """Write the program into ``out_dir``; return the generated banks.

        Every loaded unit (``main.asm``, ``bank_XX.asm``, support files) is
        written back at its path *relative to the program root* (the entry
        file's directory), so a subdir like ``resources/`` is preserved and an
        untouched unit round-trips byte-for-byte. The pieces from :meth:`add`
        are grouped by ROM bank (``org >> 16``) into fresh ``bank_XX.asm``
        files beside the base banks; wiring them into ``main.asm`` (the
        ``incsrc`` lines and any ROM padding) is the driver's job, as that
        block is program-specific. ``bank_header(bank)`` names the header for a
        generated bank (a plain one by default). Returns the generated
        ``bank_XX.asm`` names, in bank order.
        """
        out_dir = Path(out_dir)
        root = (self.entry.parent if self.entry else out_dir).resolve()

        by_bank: dict[int, list[Placed]] = defaultdict(list)
        for piece in self._relocated:
            by_bank[piece.org >> 16].append(piece)
        generated: list[str] = []
        for bank in sorted(by_bank):
            name = f"bank_{bank:02X}.asm"
            blocks = sorted(by_bank[bank], key=lambda piece: piece.org)
            body = "\n\n".join(piece.render() for piece in blocks)
            header = (
                bank_header(bank)
                if bank_header
                else f"; {name} -- generated; do not hand-edit."
            )
            dest = out_dir / name
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(header.rstrip("\n") + "\n\n" + body + "\n")
            generated.append(name)

        for path in self.order:
            try:
                rel: Path = path.resolve().relative_to(root)
            except ValueError:
                rel = Path(path.name)
            dest = out_dir / rel
            dest.parent.mkdir(parents=True, exist_ok=True)
            self.units[path].write(dest)

        return generated
