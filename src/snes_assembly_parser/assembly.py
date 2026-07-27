"""The :class:`Assembly` -- one editable, sized, addressable run of lines.

An :class:`Assembly` is the single unit the library is built around: an ordered
list of :class:`~.source.Line` that knows each line's byte size and so can be
relocated, edited, extracted from, and rendered. A whole source file is an
``Assembly``; so is a single routine pulled out of one. It merges what used to
be split across ``Source`` (parse + index + extract) and ``Segment`` (edit +
render).

Two ways in:

* :meth:`Assembly.from_path` / :meth:`~Assembly.from_content` parse text and
  *size* it (via a :class:`~.sizing.Sizer`, anchor-based by default).
* ``Assembly(lines)`` wraps lines that already carry their sizes -- what
  extraction and the line builders (:func:`instructions`, :func:`data`, ...)
  produce -- and trusts them.

Extraction (:meth:`~Assembly.function`, :meth:`~Assembly.pool`,
:meth:`~Assembly.extract`) returns fresh ``Assembly`` copies, so edits never
disturb the original.
"""

from __future__ import annotations

import copy
import itertools
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from .sizing import AnchorSizer, Sizer, computed_size, data_size
from .source import (
    Block,
    Line,
    Pool,
    block_end,
    is_position_marker,
    leading_comments,
    trim_trailing,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

# A symbol name appearing in an operand (a potential reference to another
# block/pool). The lookbehind skips sublabel refs (``.foo``), mid-identifier
# hits, and hex literals (``$FF``); immediate refs (``#TheFont``) still match.
_OPERAND_IDENT = re.compile(r"(?<![.\w$])[A-Za-z_]\w*")

Entry = Block | Pool | str


# --------------------------------------------------------------------------
# line builders (sized, ready to insert)
# --------------------------------------------------------------------------
def _ensure_anchor(line: Line) -> Line:
    """Give a byte-emitting, label-less line a placeholder ``#_`` anchor.

    Inserted code/data always wants an anchor so :meth:`Assembly.render` stamps
    its real address; the six zero digits fix the emitted width at six.
    """
    if line.size and line.label is None:
        line.label = "#_000000"
        line.label_sep = " "
    return line


def instruction(text: str) -> Line:
    """An instruction line, sized from its opcode (see
    :func:`~.sizing.computed_size`)."""
    line = Line.from_line(text)
    size = computed_size(line)
    if size is None:
        msg = f"cannot size instruction {text!r}"
        raise ValueError(msg)
    line.size = size
    return _ensure_anchor(line)


def instructions(texts: Iterable[str]) -> list[Line]:
    """A list of sized :func:`instruction` lines."""
    return [instruction(text) for text in texts]


def data(text: str) -> Line:
    """A ``db``/``dw``/``dl`` line, auto-sized; anchor added if omitted."""
    line = Line.from_line(text)
    line.size = data_size(line)
    return _ensure_anchor(line)


def datas(texts: Iterable[str]) -> list[Line]:
    """A list of :func:`data` lines (each auto-sized and anchored)."""
    return [data(text) for text in texts]


def note(text: str) -> Line:
    """A zero-size line: a label, comment, or blank that emits no bytes."""
    return Line.from_line(text)


def notes(texts: Iterable[str]) -> list[Line]:
    """A list of :func:`note` lines (labels/comments/blanks)."""
    return [note(text) for text in texts]


def _coerce(lines: Iterable[Line | str]) -> list[Line]:
    """Accept ready :class:`Line` objects or raw strings (parsed as notes)."""
    return [line if isinstance(line, Line) else note(line) for line in lines]


# --------------------------------------------------------------------------
# the Assembly
# --------------------------------------------------------------------------
@dataclass
class Assembly:
    """An ordered, editable, sized run of :class:`~.source.Line`.

    ``lines`` carry their own byte sizes; construct via :meth:`from_path` /
    :meth:`from_content` to parse-and-size text, or pass already-sized lines
    directly. Label/pool indexes are rebuilt lazily after any mutation.
    """

    lines: list[Line] = field(default_factory=list)
    sizer: Sizer = field(default_factory=AnchorSizer)
    #: Whether the source ended with a trailing newline (preserved by
    #: :meth:`text` so an untouched file round-trips byte-for-byte).
    final_newline: bool = True
    _labels: dict[str, int] | None = field(
        default=None, init=False, repr=False
    )
    _pools: dict[str, tuple[int, int]] | None = field(
        default=None, init=False, repr=False
    )

    # ---- construction ----
    @classmethod
    def from_lines(
        cls, lines: Iterable[Line], *, sizer: Sizer | None = None
    ) -> Assembly:
        """Parse-size a run of :class:`Line` (their sizes are (re)computed)."""
        assembly = cls(list(lines), sizer or AnchorSizer())
        assembly.resize()
        return assembly

    @classmethod
    def from_content(
        cls,
        content: Iterable[str],
        *,
        sizer: Sizer | None = None,
        final_newline: bool = True,
    ) -> Assembly:
        """Parse text lines into a sized :class:`Assembly`."""
        assembly = cls.from_lines(
            (Line.from_line(text) for text in content), sizer=sizer
        )
        assembly.final_newline = final_newline
        return assembly

    @classmethod
    def from_path(cls, path: Path, *, sizer: Sizer | None = None) -> Assembly:
        """Parse an asar source file into a sized :class:`Assembly`."""
        raw = path.read_text()
        return cls.from_content(
            raw.splitlines(), sizer=sizer, final_newline=raw.endswith("\n")
        )

    # ---- indexing (lazy; invalidated on mutation) ----
    def _mutated(self) -> None:
        self._labels = None
        self._pools = None

    def resize(self) -> None:
        """Re-run the sizer over the lines (call after inserting raw lines)."""
        self.sizer.size_all(self.lines)
        self._mutated()

    def _build_index(self) -> None:
        labels: dict[str, int] = {}
        pools: dict[str, tuple[int, int]] = {}
        pool_name: str | None = None
        pool_start = 0
        for index, line in enumerate(self.lines):
            if line.opcode == "pool" and line.arguments:
                if line.arguments[0] == "off":
                    if pool_name is not None:
                        pools[pool_name] = (pool_start, index + 1)
                        pool_name = None
                else:
                    pool_name, pool_start = line.arguments[0], index
            elif (
                pool_name is None
                and line.is_top_level_label
                and line.label is not None
            ):
                labels[line.label] = index
        self._labels, self._pools = labels, pools

    @property
    def labels(self) -> dict[str, int]:
        """Top-level label -> line index (rebuilt after edits)."""
        if self._labels is None:
            self._build_index()
        return self._labels or {}

    @property
    def pools(self) -> dict[str, tuple[int, int]]:
        """asar pool name -> ``(start, end)`` line span (rebuilt after
        edits)."""
        if self._pools is None:
            self._build_index()
        return self._pools or {}

    @property
    def functions(self) -> list[str]:
        """Every top-level label name, in source order."""
        return sorted(self.labels, key=lambda name: self.labels[name])

    def _boundaries(self) -> list[int]:
        return sorted(
            {*self.labels.values(), *(s for s, _ in self.pools.values())}
        )

    # ---- addressing ----
    @property
    def start_address(self) -> int | None:
        """Address of the first live anchor, or ``None``."""
        return next(
            (line.address for line in self.lines if line.address is not None),
            None,
        )

    @property
    def end_address(self) -> int | None:
        """One past the footprint (start + total byte size), or ``None``."""
        start = self.start_address
        if start is None:
            return None
        return start + sum(line.size for line in self.lines)

    def offset(self, delta: int) -> Assembly:
        """Shift every live anchor by ``delta`` in place, and return self.

        Relocating pulled code into the 2nd-MB mirror is
        ``asm.offset(0x200000)``
        -- the anchors move with the code, no ``render`` origin juggling.
        Markers (``NULL_``/``UNREACHABLE_``) and non-anchors are untouched.
        """
        for line in self.lines:
            address = line.address
            if address is not None:
                line.set_address(address + delta)
        return self

    def validate(self) -> list[tuple[int, int, str]]:
        """Anchors whose stated address disagrees with the computed placement.

        Walks the PC from the first anchor using *computed* instruction/data
        sizes (not the anchor-derived ones, which would make this circular) and
        returns ``(stated, computed, text)`` for every live anchor that lands
        somewhere other than where its label claims -- the drift check for
        hand-edited or mis-sized source. Between anchors the PC follows the
        computed sizes; at each anchor it realigns to the stated address so one
        bad anchor does not cascade into every later one. An empty list means
        every anchor is where it says it is.
        """
        mismatches: list[tuple[int, int, str]] = []
        pc = self.start_address
        for line in self.lines:
            if pc is not None and line.is_address_label:
                stated = line.address
                if stated is not None and stated != pc:
                    mismatches.append((stated, pc, str(line)))
                pc = stated if stated is not None else pc
            if pc is not None:
                pc += computed_size(line) or 0
        return mismatches

    def render(self, org: int | None = None) -> str:
        """Emit the assembly as text, tracking the PC from ``org``.

        Each live anchor is re-stamped to the running PC; the PC advances by
        every line's size. ``org`` defaults to the assembly's own start
        address, so an unedited assembly round-trips. A gap marker
        (``Line.org_gap``) advances the PC and emits an ``org`` instead of its
        own text (reserving space a dropped block held).
        """
        pc = self.start_address if org is None else org
        if pc is None:
            return "\n".join(str(line) for line in self.lines)
        out: list[str] = []
        for line in self.lines:
            if line.org_gap:
                pc += line.size
                out.append(f"org ${pc:06X}")
                continue
            if line.is_address_label:
                line.set_address(pc)
            out.append(str(line))
            pc += line.size
        return "\n".join(out)

    # ---- extraction (returns fresh copies) ----
    def _copy(self, lines: list[Line]) -> Assembly:
        return Assembly([copy.copy(line) for line in lines])

    def _block_span(self, label: str) -> tuple[int, int]:
        if label not in self.labels:
            msg = f"no top-level label named {label!r}"
            raise KeyError(msg)
        start = self.labels[label]
        return start, block_end(self.lines, start)

    def _block_lines(self, label: str, *, comments: bool) -> list[Line]:
        start, end = self._block_span(label)
        body = trim_trailing(self.lines[start:end])
        if comments:
            return leading_comments(self.lines, start) + body
        return body

    def function(self, name: str, *, comments: bool = False) -> Assembly:
        """The top-level block ``name`` as a fresh :class:`Assembly`.

        Spans from the label to the next top-level label or pool; trailing
        blanks are trimmed. With ``comments`` the block's leading comment
        comes along.
        """
        return self._copy(self._block_lines(name, comments=comments))

    def _pool_lines(self, name: str, *, comments: bool) -> list[Line]:
        if name not in self.pools:
            msg = f"no pool named {name!r}"
            raise KeyError(msg)
        start, end = self.pools[name]
        body = trim_trailing(self.lines[start:end])
        header = leading_comments(self.lines, start) if comments else []
        return header + body

    def pool(self, name: str, *, comments: bool = False) -> Assembly:
        """The asar label pool ``name`` (its ``pool``..``pool off`` span)."""
        return self._copy(self._pool_lines(name, comments=comments))

    def region(self, first: str, last: str) -> Assembly:
        """Everything from ``first`` through the end of ``last``'s block."""
        start, _ = self._block_span(first)
        _, end = self._block_span(last)
        return self._copy(trim_trailing(self.lines[start:end]))

    def blocks_until(self, start: str) -> Assembly:
        """From ``start`` up to the first ``NULL_``/``UNREACHABLE_`` marker.

        The tool for data runs too large to list block-by-block (e.g. the
        message table, which ends at a free-ROM pad or EOF).
        """
        if start not in self.labels:
            msg = f"no top-level label named {start!r}"
            raise KeyError(msg)
        begin = self.labels[start]
        end = len(self.lines)
        for index in range(begin + 1, len(self.lines)):
            line = self.lines[index]
            if line.is_null_label or line.is_unreachable_label:
                end = index
                break
        return self._copy(trim_trailing(self.lines[begin:end]))

    # ---- closure / concat (from the old Source) ----
    def _entries_for(self, name: str) -> list[Block | Pool]:
        entries: list[Block | Pool] = []
        if name in self.pools:
            entries.append(Pool(name))
        if name in self.labels:
            entries.append(Block(name))
        if not entries:
            msg = f"no top-level label or pool named {name!r}"
            raise KeyError(msg)
        return entries

    def _entry_span(self, entry: Block | Pool) -> tuple[int, int]:
        if isinstance(entry, Pool):
            if entry.name not in self.pools:
                msg = f"no pool named {entry.name!r}"
                raise KeyError(msg)
            return self.pools[entry.name]
        return self._block_span(entry.name)

    def _entry_lines(
        self, entry: Block | Pool, *, comments: bool
    ) -> list[Line]:
        if isinstance(entry, Pool):
            return self._pool_lines(entry.name, comments=comments)
        return self._block_lines(entry.name, comments=comments)

    def _references(self, name: str) -> set[str]:
        refs: set[str] = set()
        for entry in self._entries_for(name):
            for line in self._entry_lines(entry, comments=False):
                for argument in line.arguments:
                    for token in _OPERAND_IDENT.findall(argument):
                        if (
                            token in self.labels or token in self.pools
                        ) and not is_position_marker(token):
                            refs.add(token)
        return refs

    def closure(
        self,
        roots: Iterable[str],
        *,
        recursive: bool,
        external: Iterable[str] = (),
    ) -> list[Block | Pool]:
        """Resolve ``roots`` to a source-ordered :meth:`concat` list.

        With ``recursive`` the set is closed over operand references, so a
        caller lists only what it directly needs. ``external`` names symbols
        referenced but provided elsewhere (neither followed nor emitted).
        Position markers are never included; the result is source-ordered with
        each name's pool before its block.
        """
        skip = set(external)
        seen: set[str] = set()
        queue = [
            name
            for name in roots
            if name not in skip and not is_position_marker(name)
        ]
        while queue:
            name = queue.pop()
            if name in seen:
                continue
            self._entries_for(name)  # validate existence
            seen.add(name)
            if recursive:
                queue.extend(
                    ref for ref in self._references(name) if ref not in skip
                )
        entries: list[tuple[int, Block | Pool]] = [
            (self._entry_span(entry)[0], entry)
            for name in seen
            for entry in self._entries_for(name)
        ]
        entries.sort(key=lambda item: item[0])
        return [entry for _, entry in entries]

    def extract(
        self,
        roots: Iterable[str],
        *,
        recursive: bool,
        external: Iterable[str] = (),
        comments: bool = False,
        gap_notes: dict[str, str] | None = None,
    ) -> Assembly:
        """Pull ``roots`` (and, if ``recursive``, their references) as one
        :class:`Assembly`, dropped interior space reserved with an ``org``."""
        return self.concat(
            self.closure(roots, recursive=recursive, external=external),
            comments=comments,
            gap_notes=gap_notes,
        )

    def _assert_gap_declared(
        self, start: int, end: int, before: Block | Pool, after: Block | Pool
    ) -> None:
        index = start
        while index < end:
            line = self.lines[index]
            if line.is_null_label or line.is_unreachable_label:
                index = next(
                    (
                        boundary
                        for boundary in self._boundaries()
                        if boundary > index
                    ),
                    len(self.lines),
                )
                continue
            if line.has_content:
                what = line.label or line.opcode
                msg = (
                    f"unnamed content {what!r} between {before.name!r} and "
                    f"{after.name!r}; declare it as Block()/Pool() or it "
                    f"would be silently dropped"
                )
                raise ValueError(msg)
            index += 1

    @staticmethod
    def _gap_marker(
        gap_bytes: int, dead: list[str], gap_notes: dict[str, str]
    ) -> list[Line]:
        names = ", ".join(dead) if dead else "dead/free data"
        marker = [
            Line.from_line(
                f"; +{gap_bytes} byte gap: dropped {names}; the org below "
                "reserves that space so the"
            ),
            Line.from_line(
                "; following blocks keep their original offset (dropping "
                "it bare would shift them)."
            ),
        ]
        marker += [
            Line.from_line(f"; {name}: {gap_notes[name]}")
            for name in dead
            if name in gap_notes
        ]
        marker.append(Line(size=gap_bytes, org_gap=True))
        return marker

    def concat(
        self,
        items: Iterable[Entry],
        *,
        comments: bool = False,
        gap_notes: dict[str, str] | None = None,
    ) -> Assembly:
        """Copy the declared blocks/pools, in source order, as one Assembly.

        Each item is a :class:`Block`, a :class:`Pool`, or a bare ``str``
        (treated as ``Block``). Items must be in source order with only
        free/dead content between them; a dropped block's space is reserved
        with an ``org`` so survivors keep their offset. ``gap_notes`` maps a
        dropped block's label to an explanation appended to its gap comment.
        """
        gap_notes = gap_notes or {}
        entries: list[Block | Pool] = [
            Block(item) if isinstance(item, str) else item for item in items
        ]
        spans = [self._entry_span(entry) for entry in entries]
        for (before, (_bs, before_end)), (
            after,
            (after_start, _ae),
        ) in itertools.pairwise(zip(entries, spans, strict=True)):
            if after_start < before_end:
                msg = (
                    f"{after.name!r} is not after "
                    f"{before.name!r} in source order"
                )
                raise ValueError(msg)
            self._assert_gap_declared(before_end, after_start, before, after)
        lines: list[Line] = []
        prev_end_addr: int | None = None
        prev_src_end: int | None = None
        for entry, (start, end) in zip(entries, spans, strict=True):
            span = self.lines[start:end]
            entry_addr = next(
                (line.address for line in span if line.address is not None),
                None,
            )
            entry_size = sum(line.size for line in span)
            if (
                prev_end_addr is not None
                and entry_addr is not None
                and entry_addr > prev_end_addr
            ):
                gap_bytes = entry_addr - prev_end_addr
                dead = [
                    line.label
                    for line in self.lines[(prev_src_end or start) : start]
                    if line.label is not None
                    and (line.is_null_label or line.is_unreachable_label)
                ]
                lines.append(Line.from_line(""))
                lines.extend(self._gap_marker(gap_bytes, dead, gap_notes))
            if lines:
                lines.append(Line.from_line(""))
            lines.extend(self._entry_lines(entry, comments=comments))
            if entry_addr is not None:
                prev_end_addr = entry_addr + entry_size
            elif prev_end_addr is not None:
                prev_end_addr += entry_size
            prev_src_end = end
        return self._copy(lines)

    # ---- editing (in place) ----
    def find(self, needle: str, start: int = 0) -> int:
        """Index of the first line at/after ``start`` containing ``needle``."""
        for index in range(start, len(self.lines)):
            if needle in str(self.lines[index]):
                return index
        msg = f"no line containing {needle!r}"
        raise KeyError(msg)

    def line(self, needle: str) -> Line:
        """The first line containing ``needle`` (edit its fields directly)."""
        return self.lines[self.find(needle)]

    def replace(self, old: str, new: str, count: int) -> None:
        """Replace exactly ``count`` substring occurrences (byte-neutral).

        Raises unless ``old`` occurs exactly ``count`` times, so upstream drift
        is caught. Each edited line keeps its recorded size.
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
        self._mutated()

    def replace_all(self, edits: Iterable[tuple[str, str, int]]) -> None:
        """Apply a batch of :meth:`replace` ``(old, new, count)`` triples."""
        for old, new, count in edits:
            self.replace(old, new, count)

    def annotate(self, needle: str, comment: str) -> None:
        """Append ``comment`` to the first line containing ``needle``."""
        line = self.lines[self.find(needle)]
        if line.comment is None:
            if line.has_content and not line.trail:
                line.trail = " "
            line.comment = f" {comment}"
        else:
            line.comment = f"{line.comment} {comment}"

    def line_at(self, address: int) -> Line:
        """The line anchored at ROM ``address`` (``#_<hex>``)."""
        for line in self.lines:
            if line.address == address:
                return line
        msg = f"no line anchored at ${address:06X}"
        raise KeyError(msg)

    def set_operand(
        self,
        address: int,
        instruction_text: str,
        *,
        comment: str | None = None,
    ) -> None:
        """Replace the whole instruction on the ``address`` line, byte-neutral.

        ``instruction_text`` is the opcode + operands (no label); the anchor
        and optional trailing ``comment`` are re-attached, and the line keeps
        its recorded size (the caller keeps the byte width identical). For
        swapping an operand -- a literal or address -- in place.
        """
        index = next(
            index
            for index, line in enumerate(self.lines)
            if line.address == address
        )
        anchor = self.lines[index].label
        size = self.lines[index].size
        text = f"{anchor}: {instruction_text}"
        if comment:
            text += f" ; {comment}"
        replaced = Line.from_line(text)
        replaced.size = size
        self.lines[index] = replaced
        self._mutated()

    def rename_label(self, old: str, new: str) -> None:
        """Rename the *definition* of ``old`` to ``new`` here (callers kept).

        Rewrites the routine label ``old:``, an address-transparent ``#old:``,
        and/or a ``pool old`` directive -- whichever exist. References are left
        alone (they resolve to whatever now claims the old name); this is the
        freeing half of a hook, not a refactor (:meth:`suffix`).
        """
        found = False
        for line in self.lines:
            if line.opcode == "pool" and line.arguments[:1] == [old]:
                line.arguments = [new, *line.arguments[1:]]
                found = True
            elif line.label == old:
                line.label = new
                found = True
            elif line.label == f"#{old}":
                line.label = f"#{new}"
                found = True
        if not found:
            msg = f"no definition of {old!r} to rename"
            raise KeyError(msg)
        self._mutated()

    def rename_labels(self, pairs: Iterable[tuple[str, str]]) -> None:
        """Apply a batch of :meth:`rename_label` ``(old, new)`` pairs."""
        for old, new in pairs:
            self.rename_label(old, new)

    def insert_after(self, needle: str, lines: Iterable[Line | str]) -> None:
        """Insert ``lines`` after the first line containing ``needle``."""
        index = self.find(needle) + 1
        self.lines[index:index] = _coerce(lines)
        self._mutated()

    def insert_before(self, needle: str, lines: Iterable[Line | str]) -> None:
        """Insert ``lines`` before the first line containing ``needle``."""
        index = self.find(needle)
        self.lines[index:index] = _coerce(lines)
        self._mutated()

    def append(self, lines: Iterable[Line | str]) -> None:
        """Append ``lines`` to the end."""
        self.lines.extend(_coerce(lines))
        self._mutated()

    def delete(self, first: str, stop: str) -> None:
        """Delete lines from ``first`` up to (excluding) ``stop``, by
        substring."""
        begin = self.find(first)
        end = self.find(stop, begin)
        del self.lines[begin:end]
        self._mutated()

    def delete_block(self, label: str) -> None:
        """Delete the whole top-level block ``label`` begins."""
        for index, line in enumerate(self.lines):
            if line.is_top_level_label and line.label == label:
                del self.lines[index : block_end(self.lines, index)]
                self._mutated()
                return
        msg = f"no block labelled {label!r}"
        raise KeyError(msg)

    def delete_blocks(self, labels: Iterable[str]) -> None:
        """Delete each of the named top-level blocks."""
        for label in labels:
            self.delete_block(label)

    def comment_block(self, name: str) -> list[Line]:
        """The comment header directly above top-level block ``name``."""
        start = self._block_span(name)[0]
        return [
            copy.copy(line) for line in leading_comments(self.lines, start)
        ]

    def set_comment_block(
        self, name: str, lines: Iterable[Line | str]
    ) -> None:
        """Replace the comment header above block ``name`` with ``lines``."""
        start = self._block_span(name)[0]
        header_len = len(leading_comments(self.lines, start))
        del self.lines[start - header_len : start]
        self.insert_before(name + ":", _coerce(lines))

    # ---- refactor rename (definitions + references) ----
    def suffix(
        self, names: Iterable[str], suffix: str, *, prefix: str = ""
    ) -> None:
        """Rename ``names`` (and every reference to them) with a prefix/suffix.

        Unlike a hook rename (which frees a name for something else to claim),
        this is a *refactor*: a definition ``Foo:`` becomes the renamed form
        and so does every ``Foo`` / ``Foo_sub`` operand and ``pool Foo``, so
        pulled code refers only to its own renamed symbols. Comments and
        dot-sublabels are untouched.
        """
        wanted = set(names)
        if not wanted:
            return
        alt = "|".join(
            re.escape(name) for name in sorted(wanted, key=len, reverse=True)
        )
        token = re.compile(rf"(?<![\w.])(?:{alt})(?![\w])")

        def rename(match: re.Match[str]) -> str:
            return f"{prefix}{match.group(0)}{suffix}"

        for line in self.lines:
            # label definition (``Foo:`` or ``pool Foo``)
            if line.label is not None and line.label in wanted:
                line.label = f"{prefix}{line.label}{suffix}"
            if (
                line.opcode == "pool"
                and line.arguments[:1]
                and (line.arguments[0] in wanted)
            ):
                line.arguments = [
                    f"{prefix}{line.arguments[0]}{suffix}",
                    *line.arguments[1:],
                ]
            # operand references
            line.arguments = [token.sub(rename, arg) for arg in line.arguments]
        self._mutated()

    # ---- output ----
    def text(self) -> str:
        """Verbatim text (no address re-stamping); the exact round trip."""
        body = "\n".join(str(line) for line in self.lines)
        return body + "\n" if self.final_newline else body

    def write(self, path: Path) -> None:
        """Write :meth:`text` to ``path``."""
        path.write_text(self.text())

    def copy(self) -> Assembly:
        """A deep-ish copy (independent line objects, same sizer)."""
        return Assembly(
            [copy.copy(line) for line in self.lines],
            self.sizer,
            self.final_newline,
        )

    def extend(self, other: Assembly) -> None:
        """Append another assembly's lines (used to join placed pieces)."""
        self.lines.extend(copy.copy(line) for line in other.lines)
        self._mutated()


def join(pieces: Sequence[Assembly]) -> Assembly:
    """Concatenate several assemblies' lines into one (copies preserved)."""
    result = Assembly()
    for piece in pieces:
        result.extend(piece)
    return result
