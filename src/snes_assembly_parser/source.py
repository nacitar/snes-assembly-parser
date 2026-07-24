"""Parsing model for asar-compatible SNES assembly source.

:class:`Line` decomposes a single source line losslessly, and :class:`Source`
holds a list of them, indexing top-level labels and asar label pools so that
labelled blocks (routines, tilemaps, data tables, ...) can be extracted by
name.
"""

from __future__ import annotations

import copy
import itertools
import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

from .address import Address, Indicator

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path

    from .segment import Segment

# Data directives and the byte width of each operand.
_DATA_WIDTHS = {"db": 1, "dw": 2, "dl": 3, "dd": 4}
_HEX_RUN = re.compile(r"[0-9A-Fa-f]+")

# A symbol name appearing in an operand (a potential reference to another
# block/pool). The lookbehind skips sublabel refs (``.foo``), mid-identifier
# hits, and hex literals (``$FF``); immediate refs (``#TheFont``) still match.
_OPERAND_IDENT = re.compile(r"(?<![.\w$])[A-Za-z_]\w*")


def is_position_marker(name: str) -> bool:
    """Whether ``name`` marks free ROM (``NULL_``) or dead code
    (``UNREACHABLE_``).

    These carry their address in the name; they are position markers, not
    relocatable symbols, so extraction never follows or emits them (their space
    is instead reserved with an ``org`` -- see :meth:`Source.concat`).
    """
    return name.startswith(("NULL_", "UNREACHABLE_"))


def data_size(line: Line) -> int:
    """Bytes emitted by a ``db``/``dw``/``dl``/``dd`` line (operands x width).

    Returns 0 for any other line. Used to size inserted data and to derive the
    end address of a data region that runs to the end of the file (where there
    is no following anchor to measure against).
    """
    if line.opcode is None:
        return 0
    width = _DATA_WIDTHS.get(line.opcode.lower())
    if width is None:
        return 0
    return len(line.arguments) * width


def instruction_shape(line: Line) -> str:
    """A size-determining signature: opcode plus operand punctuation.

    Hex runs (operand *values*) are blanked, so ``LDA.w #$5959`` and
    ``LDA.w #$0004`` share a shape; the explicit ``.b``/``.w``/``.l`` width in
    the opcode is what fixes the size, so a shape maps to exactly one byte size
    (see :meth:`Source.instruction_sizes`).
    """
    operands = _HEX_RUN.sub("H", " ".join(line.arguments))
    return f"{line.opcode} {operands}".rstrip()


def _is_block_boundary(line: Line) -> bool:
    """Whether ``line`` begins a new block: a top-level label or a ``pool``
    open."""
    return line.is_top_level_label or (
        line.opcode == "pool"
        and bool(line.arguments)
        and line.arguments[0] != "off"
    )


def block_end(lines: list[Line], start: int) -> int:
    """Index one past the block beginning at ``start``.

    Scans forward to the next block boundary (top-level label or ``pool``
    open), or the end of ``lines``. Shared by :meth:`Source._block_span` and
    :meth:`~.segment.Segment.delete_block` so "where a block ends" is defined
    once.
    """
    for index in range(start + 1, len(lines)):
        if _is_block_boundary(lines[index]):
            return index
    return len(lines)


def _assert_no_org(lines: list[Line]) -> None:
    """Raise if ``lines`` cross an ``org`` directive.

    Byte sizes assume contiguous ROM, so a span that includes an ``org`` (which
    relocates the program counter) would be mis-sized. Callers keep runs within
    a single ``org`` section; this turns a violation into a loud failure rather
    than silently wrong addresses.
    """
    for line in lines:
        if line.opcode == "org":
            msg = f"span crosses an org directive: {str(line)!r}"
            raise ValueError(msg)


def _split_arguments(text: str) -> tuple[list[str], list[str]]:
    """Split an operand string on top-level commas.

    Commas inside ``()``/``[]`` (e.g. ``(.vectors,X)``) or inside a quoted
    string (e.g. ``incbin "a,b"``) are not separators.

    Returns ``(arguments, separators)`` where ``arguments`` are the
    whitespace-stripped operands and ``separators`` are the literal strings
    between them (a comma plus any surrounding whitespace). There is exactly
    one separator between each adjacent pair, so the original substring is
    reproduced by interleaving them.
    """
    parts: list[str] = []
    buffer: list[str] = []
    depth = 0
    quote: str | None = None
    for char in text:
        if quote is not None:
            buffer.append(char)
            if char == quote:
                quote = None
        elif char in "\"'":
            quote = char
            buffer.append(char)
        elif char in "([":
            depth += 1
            buffer.append(char)
        elif char in ")]":
            depth -= 1
            buffer.append(char)
        elif char == "," and depth == 0:
            parts.append("".join(buffer))
            buffer = []
        else:
            buffer.append(char)
    parts.append("".join(buffer))

    if parts == [""]:  # no operands at all
        return [], []

    arguments = [part.strip() for part in parts]
    separators = [
        parts[index][len(parts[index].rstrip()) :]  # trailing ws of left
        + ","
        + parts[index + 1][
            : len(parts[index + 1]) - len(parts[index + 1].lstrip())
        ]  # leading ws of right
        for index in range(len(parts) - 1)
    ]
    return arguments, separators


@dataclass
class Line:
    """A single line of asar-compatible source, decomposed losslessly.

    ``str(line)`` reproduces the original text exactly, including whitespace,
    while ``label``/``opcode``/``arguments`` expose the parsed structure. The
    ``*_sep`` and ``trail`` fields hold the exact whitespace runs so nothing is
    discarded on a round trip.
    """

    LINE_PATTERN: ClassVar[re.Pattern[str]] = re.compile(
        r"(?P<indent>[ \t]*)"
        # A label with a trailing colon (main labels, e.g. ``Foo:``), or a
        # colon-less sublabel (this codebase writes those as ``.foo``).
        r"(?:(?P<label>[^\s;]+):(?P<label_sep>[ \t]*)"
        r"|(?P<sublabel>\.[^\s;]+)(?P<sublabel_sep>[ \t]*))?"
        r"(?:(?P<opcode>[^\s;]+)(?:(?P<opcode_sep>[ \t]+)"
        r"(?P<arguments>[^;]*?))?)?"
        r"(?P<trail>[ \t]*)"
        r"(?:;(?P<comment>.*))?"
    )

    indent: str = ""
    label: str | None = None
    label_sep: str = ""
    #: Whether ``label`` was written with a trailing colon. Sublabels such as
    #: ``.foo`` in this codebase omit it; tracked so the round trip is exact.
    label_colon: bool = True
    opcode: str | None = None
    opcode_sep: str = ""
    arguments: list[str] = field(default_factory=list)
    arg_seps: list[str] = field(default_factory=list)
    trail: str = ""
    comment: str | None = None
    #: Bytes this line emits. Populated by :meth:`Source.reindex` from anchor
    #: adjacency (0 until then); carried on copies so extracted lines stay
    #: sized. Not part of equality -- it is derived context, not syntax.
    size: int = field(default=0, compare=False)
    #: When true this is a synthetic gap marker (see :meth:`Source.concat`):
    #: :meth:`Segment.render` skips ``size`` bytes and emits an ``org`` to the
    #: new PC instead of this line's own text -- reserving space a dropped
    #: block held so following blocks keep their address. Not part of
    #: equality/round-trip.
    org_gap: bool = field(default=False, compare=False)

    @classmethod
    def from_line(cls, line: str) -> Line:
        match = cls.LINE_PATTERN.fullmatch(line)
        if match is None:  # pragma: no cover - pattern accepts any line
            msg = f"unparseable line: {line!r}"
            raise ValueError(msg)
        arguments, arg_seps = _split_arguments(match["arguments"] or "")
        if match["label"] is not None:
            label, label_sep, label_colon = (
                match["label"],
                match["label_sep"],
                True,
            )
        elif match["sublabel"] is not None:
            label, label_sep, label_colon = (
                match["sublabel"],
                match["sublabel_sep"],
                False,
            )
        else:
            label, label_sep, label_colon = None, "", True
        return cls(
            indent=match["indent"],
            label=label,
            label_sep=label_sep or "",
            label_colon=label_colon,
            opcode=match["opcode"],
            opcode_sep=match["opcode_sep"] or "",
            arguments=arguments,
            arg_seps=arg_seps,
            trail=match["trail"],
            comment=match["comment"],
        )

    @property
    def is_top_level_label(self) -> bool:
        """Whether this line defines a top-level (scope-defining) label.

        Excludes sublabels (``.foo``) and asar's scope-transparent ``#``
        address labels (``#_0E8000``), leaving the named routine/data entry
        points that a caller would look up by name.
        """
        return self.label is not None and not self.label.startswith((".", "#"))

    @property
    def has_content(self) -> bool:
        """Whether the line carries a label or opcode.

        ``False`` for blank and comment-only lines.
        """
        return self.label is not None or self.opcode is not None

    @property
    def is_blank(self) -> bool:
        """Whether the line is empty/whitespace-only (no content, no
        comment)."""
        return not self.has_content and self.comment is None

    @property
    def anchor(self) -> Address | None:
        """The label parsed as an :class:`Address`, or ``None``.

        Covers live anchors (``#_<hex>``), APU-tagged anchors (``#_<hex>o``),
        and free/dead markers (``NULL_``/``UNREACHABLE_``). Named ``#`` labels
        such as ``#Module0E_02_RenderText`` are not addresses and yield
        ``None``.
        """
        return Address.parse(self.label)

    @property
    def is_address_label(self) -> bool:
        """Whether the label is a live address anchor (``#_<hex>``).

        Distinguishes live anchors (which participate in byte sizing) from
        free/dead markers and scope-neutral named ``#`` labels.
        """
        anchor = self.anchor
        return anchor is not None and anchor.is_anchor

    @property
    def address(self) -> int | None:
        """The integer offset of a live anchor line, or ``None``.

        The APU bank tag (``o``/``u``/``c``) does not affect the value, and
        free/dead markers (``NULL_``/``UNREACHABLE_``) are not live anchors so
        they return ``None`` here.
        """
        anchor = self.anchor
        if anchor is not None and anchor.is_anchor:
            return anchor.offset
        return None

    @property
    def is_null_label(self) -> bool:
        """Whether the label marks free ROM (``NULL_<hex>``)."""
        anchor = self.anchor
        return anchor is not None and anchor.indicator is Indicator.NULL

    @property
    def is_unreachable_label(self) -> bool:
        """Whether the label marks dead code (``UNREACHABLE_<hex>``)."""
        anchor = self.anchor
        return anchor is not None and anchor.indicator is Indicator.UNREACHABLE

    def set_address(self, value: int) -> None:
        """Re-stamp a live anchor line's address to ``value`` in place.

        Preserves the original hex width and any APU tag (via
        :meth:`Address.at`/:meth:`Address.render`), so an unedited line still
        round trips. Raises if the line is not a live address anchor.
        """
        anchor = self.anchor
        if anchor is None or not anchor.is_anchor:
            msg = f"line has no address label to set: {self.label!r}"
            raise ValueError(msg)
        self.label = anchor.at(value).render()

    def _render_arguments(self) -> str:
        if not self.arguments:
            return ""
        rendered = self.arguments[0]
        for separator, argument in zip(
            self.arg_seps, self.arguments[1:], strict=False
        ):
            rendered += separator + argument
        return rendered

    def __str__(self) -> str:
        out = self.indent
        if self.label is not None:
            colon = ":" if self.label_colon else ""
            out += f"{self.label}{colon}{self.label_sep}"
        if self.opcode is not None:
            # ``opcode_sep`` is the whitespace right after the opcode; it must
            # be emitted even with no arguments (an operand-less opcode before
            # an inline comment, e.g. ``INY ; +5``), else the space is lost.
            out += self.opcode + self.opcode_sep
            if self.arguments:
                out += self._render_arguments()
        out += self.trail
        if self.comment is not None:
            out += f";{self.comment}"
        return out


def trim_trailing(lines: list[Line]) -> list[Line]:
    """Return ``lines`` without trailing blank/comment-only lines.

    Interior lines (including blanks between content) are left untouched.
    """
    end = len(lines)
    while end > 0 and not lines[end - 1].has_content:
        end -= 1
    return lines[:end]


def leading_comments(lines: list[Line], index: int) -> list[Line]:
    """Return the comment block immediately preceding ``lines[index]``.

    Walks backwards from ``index`` over comment-only and blank lines, stopping
    at the previous line that carries content. Leading blank lines are dropped
    so the result starts at the first comment/separator; interior blanks are
    kept. This follows the convention (seen throughout the disassembly) that a
    routine's descriptive comments sit directly above its label.
    """
    start = index
    while start > 0 and not lines[start - 1].has_content:
        start -= 1
    header = lines[start:index]
    first = 0
    while first < len(header) and header[first].is_blank:
        first += 1
    return header[first:]


@dataclass(frozen=True)
class Block:
    """Declares a top-level labelled block to copy
    (via :meth:`Source.block`)."""

    name: str


@dataclass(frozen=True)
class Pool:
    """Declares an asar label pool to copy (via :meth:`Source.pool`)."""

    name: str


@dataclass
class Source:
    """A parsed assembly source: a list of :class:`Line` plus label indexes.

    Build one with :meth:`from_path`, :meth:`from_content`, or
    :meth:`from_lines`, then pull labelled blocks with :meth:`block` and asar
    label pools with :meth:`pool`.
    """

    lines: list[Line] = field(default_factory=list)
    #: Maps each top-level label to the index of its line in ``lines``.
    labels: dict[str, int] = field(default_factory=dict, init=False)
    #: Maps each asar label pool (``pool X`` ... ``pool off``) to its
    #: ``(start, end)`` line-index span (``end`` exclusive).
    pools: dict[str, tuple[int, int]] = field(default_factory=dict, init=False)

    def __post_init__(self) -> None:
        self.reindex()

    @classmethod
    def from_lines(cls, lines: Iterable[Line]) -> Source:
        return cls(list(lines))

    @classmethod
    def from_content(cls, content: Iterable[str]) -> Source:
        return cls.from_lines(Line.from_line(text) for text in content)

    @classmethod
    def from_path(cls, path: Path) -> Source:
        return cls.from_content(path.read_text().splitlines())

    def reindex(self) -> None:
        """Rebuild the label, pool, and per-line size indexes from ``lines``.

        Called automatically after the lines are set; call it again after
        mutating ``lines`` directly. Records top-level labels (name -> line
        index), asar label pools (``pool X`` ... ``pool off``, name ->
        ``(start, end)`` span), and every line's byte :attr:`~Line.size`.
        """
        self.labels = {}
        self.pools = {}
        pool_name: str | None = None
        pool_start = 0
        for index, line in enumerate(self.lines):
            if line.opcode == "pool" and line.arguments:
                if line.arguments[0] == "off":
                    if pool_name is not None:
                        self.pools[pool_name] = (pool_start, index + 1)
                        pool_name = None
                else:
                    pool_name, pool_start = line.arguments[0], index
            elif (
                pool_name is None
                and line.is_top_level_label
                and line.label is not None
            ):
                self.labels[line.label] = index
        self._size_lines()

    def _size_lines(self) -> None:
        """Set every line's byte size from ROM-anchor adjacency.

        An anchor's size is the gap to the next anchor in the *full* source, so
        a block's final line is sized against the true following address even
        when that anchor belongs to a block we do not extract. The last anchor
        in the file has no successor and falls back to :func:`data_size`.
        """
        prev_index = -1
        prev_address: int | None = None
        for index, line in enumerate(self.lines):
            line.size = 0
            address = line.address
            if address is None:
                continue
            if prev_index >= 0 and prev_address is not None:
                self.lines[prev_index].size = address - prev_address
            prev_index, prev_address = index, address
        if prev_index >= 0:
            self.lines[prev_index].size = data_size(self.lines[prev_index])

    def instruction_sizes(self) -> dict[str, int]:
        """Map each instruction shape to its byte size, learned from this
        source.

        Keyed by :func:`instruction_shape` (opcode + blanked operands). Because
        the disassembly writes explicit ``.b``/``.w``/``.l`` widths, each shape
        has exactly one size (processor M/X mode never changes it); a conflict
        raises. Pass the result to :func:`~.segment.code_lines` to size
        inserted instructions without hand-writing an opcode table.
        """
        sizes: dict[str, int] = {}
        for line in self.lines:
            if (
                line.opcode is None
                or line.address is None
                or line.size == 0
                or line.opcode.lower() in _DATA_WIDTHS
            ):
                continue
            key = instruction_shape(line)
            existing = sizes.setdefault(key, line.size)
            if existing != line.size:
                msg = (
                    f"instruction {key!r} has sizes {existing} and {line.size}"
                )
                raise ValueError(msg)
        return sizes

    def _boundaries(self) -> list[int]:
        """Sorted indices where a block ends: top-level labels and pools."""
        return sorted(
            {*self.labels.values(), *(s for s, _ in self.pools.values())}
        )

    def _block_span(self, label: str) -> tuple[int, int]:
        """``(start, end)`` line indices of ``label``'s block (``end``
        exclusive)."""
        if label not in self.labels:
            msg = f"no top-level label named {label!r}"
            raise KeyError(msg)
        start = self.labels[label]
        return start, block_end(self.lines, start)

    def _block_lines(self, label: str, *, comments: bool) -> list[Line]:
        """Source lines (not copies) for ``label``'s block."""
        start, end = self._block_span(label)
        body = trim_trailing(self.lines[start:end])
        if comments:
            return leading_comments(self.lines, start) + body
        return body

    def _segment(self, lines: list[Line]) -> Segment:
        """Wrap owning copies of ``lines`` in a Segment, asserting
        contiguity."""
        from .segment import Segment

        _assert_no_org(lines)
        return Segment([copy.copy(line) for line in lines])

    def block(self, label: str, *, comments: bool) -> Segment:
        """Extract the labelled block starting at ``label`` as a Segment.

        Works for any top-level label (routine, tilemap, data table, ...).
        Spans from the labelled line up to (but not including) the next
        top-level label or label pool, or the end of the file. Trailing blank
        and comment-only lines are trimmed. The returned lines are copies (with
        sizes already populated), so the caller may relocate or edit them
        without disturbing this Source. If ``comments`` is true, the block's
        leading comment header is prepended.
        """
        return self._segment(self._block_lines(label, comments=comments))

    def _pool_lines(self, name: str, *, comments: bool) -> list[Line]:
        """Source lines (not copies) for the pool declared ``pool name``."""
        if name not in self.pools:
            msg = f"no pool named {name!r}"
            raise KeyError(msg)
        start, end = self.pools[name]
        body = trim_trailing(self.lines[start:end])
        header = leading_comments(self.lines, start) if comments else []
        return header + body

    def pool(self, name: str, *, comments: bool) -> Segment:
        """Extract the asar label pool declared ``pool name`` as a Segment.

        Includes the ``pool name`` / ``pool off`` directives and everything
        between them. These sublabels are scoped to a routine defined
        elsewhere, so pools are fetched here rather than via :meth:`block`. If
        ``comments`` is true, the pool's leading comment header is prepended.
        """
        return self._segment(self._pool_lines(name, comments=comments))

    def _entry_span(self, entry: Block | Pool) -> tuple[int, int]:
        """``(start, end)`` line indices for a declared block or pool."""
        if isinstance(entry, Pool):
            if entry.name not in self.pools:
                msg = f"no pool named {entry.name!r}"
                raise KeyError(msg)
            return self.pools[entry.name]
        return self._block_span(entry.name)

    def _entry_lines(
        self, entry: Block | Pool, *, comments: bool
    ) -> list[Line]:
        """Source lines (not copies) for a declared block or pool."""
        if isinstance(entry, Pool):
            return self._pool_lines(entry.name, comments=comments)
        return self._block_lines(entry.name, comments=comments)

    def blocks_until(self, start: str) -> Segment:
        """Extract a run of blocks from ``start`` up to a stop marker.

        Walks forward from ``start``'s label, spanning every intervening
        top-level label, and stops *before* the first ``NULL_`` (free ROM) or
        ``UNREACHABLE_`` (dead code) label, or the end of the file. This is the
        tool for data runs too large to list block-by-block (e.g. the message
        table, which ends at a ``NULL_`` pad or EOF).
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
        return self._segment(trim_trailing(self.lines[begin:end]))

    def region(self, first: str, last: str) -> Segment:
        """Extract everything from ``first`` through the end of ``last``'s
        block.

        Convenience sugar over a contiguous span; prefer :meth:`concat` for an
        explicit, drift-checked list of blocks.
        """
        start, _ = self._block_span(first)
        _, end = self._block_span(last)
        return self._segment(trim_trailing(self.lines[start:end]))

    def _entries_for(self, name: str) -> list[Block | Pool]:
        """The block and/or pool declarations for a symbol ``name``.

        A single name can denote both a routine label and its same-named asar
        pool (the pool holds the routine's scoped data -- see
        :meth:`concat`); both are returned, pool first. Raises if the name
        denotes neither.
        """
        entries: list[Block | Pool] = []
        if name in self.pools:
            entries.append(Pool(name))
        if name in self.labels:
            entries.append(Block(name))
        if not entries:
            msg = f"no top-level label or pool named {name!r}"
            raise KeyError(msg)
        return entries

    def _references(self, name: str) -> set[str]:
        """Symbol names referenced by ``name``'s operands.

        Scans the operands of every line in ``name``'s block and pool for
        identifiers, keeping only those defined in this source (external
        symbols, registers, and hex drop out) and excluding position markers.
        This is the reference edge used by :meth:`closure`.
        """
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

        Each root names a routine/data label (or same-named pool) to pull. With
        ``recursive`` true the set is closed over operand references
        (:meth:`_references`) -- every block a root invokes or points at,
        transitively -- so a caller lists only what it directly needs and its
        dependencies come along. With ``recursive`` false only the named roots
        are pulled. Either way the result is sorted into source order (so a
        mirror relocation keeps the original layout) with each name's pool
        emitted before its block, and position markers are never included.

        ``external`` names symbols that are referenced but provided elsewhere
        (a shared asset, or a block relocated on its own): recursion neither
        follows into nor emits them, exactly as if the reference pointed out of
        this source. Implicit dependencies that are not operand references --
        e.g. an over-indexed table read reaching past its own end into the next
        block -- are invisible here; name such a block as an explicit root and
        :meth:`concat` reserves the intervening space with an ``org``.
        ``recursive`` is keyword-only so the traversal mode is always stated at
        the call site.
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
            self._entries_for(name)  # validate existence up front
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
    ) -> Segment:
        """Pull the blocks named by ``roots`` (and, if ``recursive``, their
        referenced dependencies) as one relocatable :class:`Segment`.

        Sugar over :meth:`closure` + :meth:`concat`: resolves the roots to a
        source-ordered block/pool list and copies them, reserving any dropped
        interior space with an ``org`` so every survivor keeps its address.
        This is the by-need counterpart to :meth:`region` (which takes two span
        endpoints): list the functions you actually want and let recursion
        gather the rest. ``external``, ``comments`` and ``gap_notes`` are
        forwarded to :meth:`closure`/:meth:`concat`.
        """
        return self.concat(
            self.closure(roots, recursive=recursive, external=external),
            comments=comments,
            gap_notes=gap_notes,
        )

    def _assert_gap_declared(
        self, start: int, end: int, before: Block | Pool, after: Block | Pool
    ) -> None:
        """Require the gap ``[start, end)`` to hold only free/dead content.

        Between two declared entries the source may contain blank/comment lines
        and complete ``NULL_``/``UNREACHABLE_`` (free/dead) blocks. Anything
        else -- a live label, a ``pool`` directive, a stray ``db``/opcode -- is
        unnamed byte-emitting content that :meth:`concat` would silently drop,
        so raise instead.
        """
        index = start
        while index < end:
            line = self.lines[index]
            if line.is_null_label or line.is_unreachable_label:
                # Skip the whole dead/free block to its next boundary. Those
                # labels are top-level, hence in self.labels and _boundaries().
                index = next(
                    (
                        boundary
                        for boundary in self._boundaries()
                        if boundary > index
                    ),
                    len(self.lines),
                )
                continue
            if line.has_content:  # live label, `pool` directive, or stray data
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
        """Comment + synthetic ``org`` reserving a ``gap_bytes`` dropped gap.

        The comment states the byte count and which dead/free block(s) were
        dropped; any ``gap_notes[name]`` adds why that space is load-bearing
        (e.g. an over-indexed read reaches into the block below).
        """
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
        items: Iterable[Block | Pool | str],
        *,
        comments: bool = False,
        gap_notes: dict[str, str] | None = None,
    ) -> Segment:
        """Copy the declared blocks/pools, in source order, as one Segment.

        Each item is a :class:`Block` (name), a :class:`Pool` (name), or a bare
        ``str`` (treated as ``Block``). The items must appear in source order
        and leave no unnamed byte-emitting content between them -- only
        ``NULL_``/``UNREACHABLE_`` (free/dead) blocks may be unnamed in a gap.
        A forgotten routine or pool is a loud error, not a silently broken ROM.

        Placement is driven by each entry's own address: when an entry
        declares an address that is *not* adjacent to where the previous entry
        ended, a synthetic ``org`` is emitted to reserve exactly the
        intervening bytes, so every entry keeps its declared offset even after
        the blocks between them were dropped -- essential when code reaches
        across blocks (an over-indexed table read) and safe if the list later
        names individual functions rather than whole regions. Entries that are
        byte-adjacent get no ``org``, and an entry with *no* address simply
        flows on (auto-numbered) from the previous one. ``gap_notes`` maps a
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
        return self._segment(lines)
