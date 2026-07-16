"""Parsing model for asar-compatible SNES assembly source.

:class:`Line` decomposes a single source line losslessly, and :class:`Source`
holds a list of them, indexing top-level labels and asar label pools so that
labelled blocks (routines, tilemaps, data tables, ...) can be extracted by name.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, ClassVar

if TYPE_CHECKING:
    from collections.abc import Iterable
    from pathlib import Path


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
        parts[i][len(parts[i].rstrip()) :]  # trailing ws of left operand
        + ","
        + parts[i + 1][
            : len(parts[i + 1]) - len(parts[i + 1].lstrip())
        ]  # leading ws of right
        for i in range(len(parts) - 1)
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
        r"(?:(?P<opcode>[^\s;]+)(?:(?P<opcode_sep>[ \t]+)(?P<arguments>[^;]*?))?)?"
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
        """Whether the line is empty/whitespace-only (no content, no comment)."""
        return not self.has_content and self.comment is None

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
            out += self.opcode
            if self.arguments:
                out += self.opcode_sep + self._render_arguments()
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
        """Rebuild the label and pool indexes from ``lines``.

        Called automatically after the lines are set; call it again after
        mutating ``lines`` directly. Records top-level labels (name -> line
        index) and asar label pools (``pool X`` ... ``pool off``, name ->
        ``(start, end)`` span). Pool bodies are indexed separately because
        their sublabels are scoped to a routine defined elsewhere.
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

    def _boundaries(self) -> list[int]:
        """Sorted indices where a block ends: top-level labels and pools."""
        return sorted(
            {*self.labels.values(), *(s for s, _ in self.pools.values())}
        )

    def block(self, label: str, *, comments: bool) -> list[Line]:
        """Return the lines of the labelled block starting at ``label``.

        Works for any top-level label (routine, tilemap, data table, ...).
        Spans from the labelled line up to (but not including) the next
        top-level label or label pool, or the end of the file if there is
        none. Trailing blank and comment-only lines are trimmed.

        If ``comments`` is true, the block's leading comment header (see
        :func:`leading_comments`) is prepended.
        """
        if label not in self.labels:
            msg = f"no top-level label named {label!r}"
            raise KeyError(msg)
        start = self.labels[label]
        end = next(
            (index for index in self._boundaries() if index > start),
            len(self.lines),
        )
        body = trim_trailing(self.lines[start:end])
        if comments:
            return leading_comments(self.lines, start) + body
        return body

    def pool(self, name: str, *, comments: bool) -> list[Line]:
        """Return the lines of the asar label pool declared ``pool name``.

        Includes the ``pool name`` / ``pool off`` directives and everything
        between them. These sublabels are scoped to a routine defined
        elsewhere, so pools are fetched here rather than via :meth:`block`.

        If ``comments`` is true, the pool's leading comment header (see
        :func:`leading_comments`) is prepended.
        """
        if name not in self.pools:
            msg = f"no pool named {name!r}"
            raise KeyError(msg)
        start, end = self.pools[name]
        body = trim_trailing(self.lines[start:end])
        if comments:
            return leading_comments(self.lines, start) + body
        return body
