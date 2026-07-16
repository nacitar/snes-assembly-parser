"""Relocatable, editable runs of assembly with per-line byte sizes.

A :class:`Segment` owns an ordered list of :class:`~.source.Line`, each already
carrying its byte :attr:`~.source.Line.size`. Because every line knows its size,
:meth:`Segment.render` can walk a program counter from any origin and re-stamp
each address anchor -- so a segment can be relocated or edited (insert / delete /
replace) and still emit correct, contiguous addresses.

Use :func:`code`, :func:`data`, and :func:`note` to build lines for insertion.
"""

from __future__ import annotations

from dataclasses import dataclass

from .source import Line, data_size


def code(text: str, size: int) -> Line:
    """A hand-sized line (e.g. an inserted instruction of known byte length)."""
    line = Line.from_line(text)
    line.size = size
    return line


def data(text: str) -> Line:
    """A ``db``/``dw``/``dl`` line, auto-sized from its operands."""
    line = Line.from_line(text)
    line.size = data_size(line)
    return line


def note(text: str) -> Line:
    """A zero-size line: a label, comment, or blank that emits no bytes."""
    return Line.from_line(text)  # size defaults to 0


@dataclass
class Segment:
    """An ordered, editable run of sized :class:`~.source.Line` with addresses.

    Edits mutate ``lines`` in place; :meth:`render` re-derives every anchor
    address from a supplied origin and the per-line sizes.
    """

    lines: list[Line]

    @property
    def start_address(self) -> int | None:
        """Address of the first anchor, or ``None`` if the segment has none."""
        return next(
            (line.address for line in self.lines if line.address is not None),
            None,
        )

    @property
    def end_address(self) -> int | None:
        """Address just past the segment's footprint (``start`` plus total size).

        This is where :meth:`render` leaves the PC, so it reflects the bytes the
        segment actually emits -- for a contiguous run it equals the last
        anchor's end, but when dead blocks are omitted (see
        :meth:`Source.concat`) it correctly shrinks by the omitted size.
        """
        start = self.start_address
        if start is None:
            return None
        return start + sum(line.size for line in self.lines)

    def _find(self, needle: str, start: int = 0) -> int:
        """Index of the first line at/after ``start`` whose text contains ``needle``."""
        for index in range(start, len(self.lines)):
            if needle in str(self.lines[index]):
                return index
        msg = f"no line containing {needle!r}"
        raise KeyError(msg)

    def replace(self, old: str, new: str, count: int) -> None:
        """Replace exactly ``count`` occurrences of ``old`` with ``new``.

        Raises if the segment does not contain exactly ``count`` occurrences --
        that loud failure is how upstream drift (e.g. an operand that changed)
        is caught. Byte-neutral: each line keeps its recorded size, so this does
        not self-correct if ``new`` emits a different number of bytes.
        """
        found = sum(str(line).count(old) for line in self.lines)
        if found != count:
            msg = f"expected {count} occurrence(s) of {old!r}, found {found}"
            raise ValueError(msg)
        for index, line in enumerate(self.lines):
            if old in str(line):
                replaced = Line.from_line(str(line).replace(old, new))
                replaced.size = line.size
                self.lines[index] = replaced

    def annotate(self, needle: str, comment: str) -> None:
        """Append ``comment`` to the first line containing ``needle`` (byte-neutral)."""
        line = self.lines[self._find(needle)]
        if line.comment is None:
            if line.has_content and not line.trail:
                line.trail = " "
            line.comment = f" {comment}"
        else:
            line.comment = f"{line.comment} {comment}"

    def insert_after(self, needle: str, lines: list[Line]) -> None:
        """Insert ``lines`` right after the first line containing ``needle``."""
        index = self._find(needle) + 1
        self.lines[index:index] = lines

    def append(self, lines: list[Line]) -> None:
        """Append ``lines`` to the end of the segment."""
        self.lines.extend(lines)

    def delete(self, first: str, stop: str) -> None:
        """Delete lines from ``first`` up to (excluding) ``stop``, by substring."""
        begin = self._find(first)
        end = self._find(stop, begin)
        del self.lines[begin:end]

    def render(self, org: int) -> str:
        """Emit the segment as text, tracking the PC from ``org``.

        Each address anchor is re-stamped to the current PC before it is
        emitted, and the PC advances by every line's size. Rendering an
        unedited segment at its original origin reproduces the source exactly;
        rendering elsewhere relocates every anchor, and inserts/deletes shift
        all downstream anchors by the net size change.
        """
        pc = org
        out: list[str] = []
        for line in self.lines:
            if line.is_address_label:
                line.set_address(pc)
            out.append(str(line))
            pc += line.size
        return "\n".join(out)
