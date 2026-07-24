"""The :class:`Patcher` -- surgical, hook-oriented edits to one source file.

Where :class:`~.assembly.Assembly` edits by substring or line, a ``Patcher``
edits a pristine bank by *label* or *``#_`` address anchor* (never a line
number), so an edit survives upstream reformatting and fails loud if its target
has drifted. Its operations are the ones a base-disassembly *hook* needs:
free a
name for a relocated copy to claim, splice a forwarding stub into free ROM,
rewrite one operand, redirect an inline block. :class:`~.rom.Rom` composes
these
across a whole program (see :meth:`~.rom.Rom.hook`).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from .assembly import Assembly
from .hooking import (
    UNREACHABLE_PREFIX,
    LandingPad,
    anchor_label,
    byte_rows,
    free_block,
    landing_pad_lines,
)
from .source import Line

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


def _lines(texts: Iterable[str]) -> list[Line]:
    return [Line.from_line(text) for text in texts]


@dataclass
class Patcher:
    """A pristine :class:`Assembly` under hook edits, by label/anchor.

    Build with :meth:`from_path` (or wrap an existing ``assembly``); every edit
    mutates it in place and re-derives its indexes.
    """

    assembly: Assembly

    @classmethod
    def from_path(cls, path: Path) -> Patcher:
        """Load the file at ``path`` as the patch target."""
        return cls(Assembly.from_path(path))

    # ---- locating -------------------------------------------------------
    def _index_of(self, locator: str | int) -> int:
        """The line index a label name or ``#_``/int address anchor names."""
        if isinstance(locator, int):
            anchor = anchor_label(locator)
            for index, line in enumerate(self.assembly.lines):
                if line.label == anchor:
                    return index
            msg = f"no line anchored at ${locator:06X}"
            raise KeyError(msg)
        if locator in self.assembly.labels:
            return self.assembly.labels[locator]
        msg = f"no top-level label named {locator!r}"
        raise KeyError(msg)

    def _block_boundary(self, start: int) -> int:
        """Index one past the block at ``start`` (its next top-level label or
        pool) -- where a ``NULL_``/routine block ends."""
        boundaries = sorted(
            {
                *self.assembly.labels.values(),
                *(begin for begin, _ in self.assembly.pools.values()),
            }
        )
        after = (index for index in boundaries if index > start)
        return next(after, len(self.assembly.lines))

    # ---- primitives -----------------------------------------------------
    def free(self, name: str, *, comment: Iterable[str] = ()) -> None:
        """Free ``name`` by prefixing ``UNREACHABLE_`` onto its definition(s).

        Rewrites the routine label ``name:``, an address-transparent
        ``#name:``,
        and/or a ``pool name`` directive -- whichever exist -- so the bare name
        frees for a relocated copy to claim while the original body stays
        put and
        byte-identical. ``comment`` lines go above the first definition.
        References are left untouched (they resolve to the relocated copy); use
        :meth:`rewrite_reference` for one that must reach the original body.
        """
        marker = f"{UNREACHABLE_PREFIX}{name}"
        first: int | None = None
        for index, line in enumerate(self.assembly.lines):
            if line.opcode == "pool" and line.arguments == [name]:
                line.arguments = [marker]
            elif line.label == name:
                line.label = marker
            elif line.label == f"#{name}":
                line.label = f"#{marker}"
            else:
                continue
            first = index if first is None else first
        if first is None:
            msg = f"free: no definition of {name!r} found"
            raise KeyError(msg)
        if comment:
            self.assembly.lines[first:first] = _lines(comment)
        self._reindex()

    def set_operand(
        self, address: int, instruction: str, *, comment: str | None = None
    ) -> None:
        """Replace the whole instruction on the ``address`` line, byte-neutral.

        ``instruction`` is the opcode + operands (no label); the ``#_`` anchor
        and optional trailing ``comment`` are re-attached. The caller keeps the
        byte width identical -- for an operand swap that moves nothing.
        """
        index = self._index_of(address)
        text = f"{anchor_label(address)}: {instruction}"
        if comment:
            text += f" ; {comment}"
        self.assembly.lines[index] = Line.from_line(text)
        self._reindex()

    def insert_before(self, locator: str | int, lines: Iterable[str]) -> None:
        """Insert verbatim ``lines`` above the located label/anchor line (e.g.
        an ``org`` that re-pins data a relocation left behind)."""
        index = self._index_of(locator)
        self.assembly.lines[index:index] = _lines(lines)
        self._reindex()

    def rewrite_reference(self, address: int, old: str, new: str) -> None:
        """Retarget the symbol ``old`` -> ``new`` in the ``address`` line's
        operands (a live reference that must reach the preserved original)."""
        line = self.assembly.lines[self._index_of(address)]
        if old not in line.arguments:
            msg = f"rewrite_reference: ${address:06X} has no operand {old!r}"
            raise KeyError(msg)
        line.arguments = [new if arg == old else arg for arg in line.arguments]
        self._reindex()

    def landing_pads(
        self,
        free_label: str,
        pads: list[LandingPad],
        *,
        header: Iterable[str] = (),
    ) -> None:
        """Fill the ``NULL_`` free-ROM block ``free_label`` with forwarding
        stubs.

        The pads lay out from the block's start address; the leftover tail
        stays
        free as a shrunk ``NULL_`` block (so following code keeps its offset).
        ``header`` lines are emitted above the first pad.
        """
        start = self._index_of(free_label)
        end = self._block_boundary(start)
        span = self.assembly.lines[start:end]
        base = next(
            (line.address for line in span if line.address is not None), None
        )
        if base is None:
            msg = f"landing_pads: {free_label!r} has no address anchor"
            raise KeyError(msg)
        region_size = sum(line.size for line in span)

        stubs, end_address = landing_pad_lines(pads, base)
        replacement = [
            *_lines(header),
            *stubs,
            *free_block(end_address, base + region_size - end_address),
        ]
        self.assembly.lines[start:end] = replacement
        self._reindex()

    def relocate_block(
        self,
        address: int,
        jml_target: str,
        *,
        resume: int,
        orphan: tuple[int, ...] = (),
        comment: Iterable[str] = (),
    ) -> None:
        """Redirect an inline block to a relocated copy with a single ``JML``.

        Only the 4-byte ``JML jml_target`` at ``address`` is a real change: the
        relocated copy runs the logic and jumps back. Everything from
        ``resume``
        onward -- the rest of the displaced block -- is left as the original
        instructions (now unreachable, but assembled byte-for-byte, so the
        source
        still reads like the original). ``resume`` is the address of the first
        original instruction the JML did not land inside; ``orphan`` is the
        bytes
        between ``address+4`` and ``resume`` (the tail of the instruction the
        JML's opcode overwrote), emitted as a lone ``db``.
        """
        start = self._index_of(address)
        jml_end = address + 4
        if resume < jml_end or len(orphan) != resume - jml_end:
            needed = resume - jml_end
            msg = (
                f"relocate_block: resume ${resume:06X} needs {needed} orphan "
                f"byte(s) after the JML, got {len(orphan)}"
            )
            raise ValueError(msg)
        stop = next(
            index
            for index in range(start + 1, len(self.assembly.lines))
            if (address_here := self.assembly.lines[index].address) is not None
            and address_here >= resume
        )
        replacement = [
            *_lines(comment),
            Line.from_line(f"{anchor_label(address)}: JML {jml_target}"),
            *(byte_rows(jml_end, list(orphan)) if orphan else []),
        ]
        self.assembly.lines[start:stop] = replacement
        self._reindex()

    # ---- output ---------------------------------------------------------
    def _reindex(self) -> None:
        """Re-derive label/pool/size indexes after a line mutation."""
        self.assembly.resize()

    def render(self) -> str:
        """The patched file's text."""
        return self.assembly.text()

    def save(self, path: Path) -> None:
        """Write :meth:`render` to ``path``."""
        path.write_text(self.render())
