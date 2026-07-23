"""Byte sizing for assembly lines, behind a pluggable :class:`Sizer` seam.

Two strategies, and a hybrid:

* :class:`AnchorSizer` -- the exact, self-validating method for the spannerisms
  disassembly: every byte-emitting line carries a ``#_<hex>`` anchor, so a
  line's size is the gap to the next anchor. Needs the whole line list.
* :class:`ComputedSizer` -- sizes each line on its own from the opcode and
  operands. Data directives use their operand width; instructions use the
  explicit ``.b``/``.w``/``.l`` operand-width suffix (``1 + {b:1,w:2,l:3}``)
  where present, and a small 65816 table for the no-suffix control/stack/branch
  opcodes. This is the seam a full M/X-flag-tracking engine slots into later;
  it currently covers what the disassembly writes and what we emit.
* :class:`HybridSizer` (default) -- anchor adjacency where it can, computed
  otherwise, so a parsed disassembly stays byte-exact while inserted (anchor-
  less) lines still get sized without a learned table.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .source import Line

# Data directives and the byte width of each operand.
DATA_WIDTHS = {"db": 1, "dw": 2, "dl": 3, "dd": 4}

# Directives/keywords that occupy a line but emit no ROM bytes.
_NON_EMITTING = frozenset(
    {"org", "warnpc", "pool", "incsrc", "namespace", "pushpc", "pullpc"}
)

# The explicit operand-width suffix -> operand byte count. A line's opcode is
# ``MNEMONIC.suffix`` (e.g. ``LDA.w``); the suffix fixes the operand encoding.
_SUFFIX_BYTES = {"b": 1, "w": 2, "l": 3}

# No-suffix opcodes grouped by their fixed operand byte count. Immediate
# opcodes here (``REP``/``SEP``/``BRK``/``COP``) always take one byte; the
# accumulator forms (``INC A``) and implied ops take none.
_IMPLIED = frozenset(
    {
        "NOP",
        "XBA",
        "XCE",
        "WAI",
        "STP",
        "RTS",
        "RTL",
        "RTI",
        "CLC",
        "SEC",
        "CLI",
        "SEI",
        "CLD",
        "SED",
        "CLV",
        "DEX",
        "DEY",
        "INX",
        "INY",
        "TAX",
        "TAY",
        "TXA",
        "TYA",
        "TSX",
        "TXS",
        "TXY",
        "TYX",
        "TCD",
        "TDC",
        "TCS",
        "TSC",
        "PHA",
        "PHX",
        "PHY",
        "PHP",
        "PHB",
        "PHD",
        "PHK",
        "PLA",
        "PLX",
        "PLY",
        "PLP",
        "PLB",
        "PLD",
        "INC",
        "DEC",
        "ASL",
        "LSR",
        "ROL",
        "ROR",  # accumulator forms (``A``)
    }
)
# No-suffix opcode -> fixed operand byte count (instruction size = 1 + this).
_OPERAND_BYTES: dict[str, int] = {
    **dict.fromkeys(_IMPLIED, 0),
    # relative-8 branches and immediate-8 ops
    **dict.fromkeys(
        ["BRA", "BEQ", "BNE", "BCC", "BCS", "BMI", "BPL", "BVC", "BVS"], 1
    ),
    **dict.fromkeys(["REP", "SEP", "BRK", "COP"], 1),
    # relative-16, absolute, and block-move
    **dict.fromkeys(["BRL", "PER", "JMP", "JSR", "PEA", "MVN", "MVP"], 2),
    # long
    **dict.fromkeys(["JML", "JSL"], 3),
}


def data_size(line: Line) -> int:
    """Bytes a ``db``/``dw``/``dl``/``dd`` line emits (operands x width), else
    0."""
    if line.opcode is None:
        return 0
    width = DATA_WIDTHS.get(line.opcode.lower())
    if width is None:
        return 0
    return len(line.arguments) * width


def computed_size(line: Line) -> int | None:
    """This line's byte size from its opcode/operands alone, or ``None``.

    ``None`` means the size is not statically known here (an unrecognised
    opcode) -- the hybrid sizer then falls back to an anchor, and
    :func:`~.segment.code_lines` raises. Data directives, the explicit-width
    instruction suffix, and the no-suffix control/stack/branch opcodes are all
    covered; anything the line does not emit (labels, comments, ``org``) is 0.
    """
    if line.opcode is None:
        return 0
    lowered = line.opcode.lower()
    if lowered in DATA_WIDTHS:
        return data_size(line)
    if lowered in _NON_EMITTING:
        return 0

    mnemonic, _, suffix = line.opcode.partition(".")
    if suffix:  # explicit operand width: 1 opcode byte + operand bytes
        operand = _SUFFIX_BYTES.get(suffix.lower())
        return 1 + operand if operand is not None else None

    operand = _OPERAND_BYTES.get(mnemonic.upper())
    return 1 + operand if operand is not None else None


class Sizer(Protocol):
    """Assigns a byte :attr:`~.source.Line.size` to every line in a list."""

    def size_all(self, lines: list[Line]) -> None: ...


class ComputedSizer:
    """Sizes each line independently via :func:`computed_size`.

    Anchor-less source (arbitrary asar) sizes without any ``#_<hex>`` markers.
    Raises if a byte-emitting line's size cannot be computed, so an unknown
    opcode is a loud failure rather than a silent miscount.
    """

    def size_all(self, lines: list[Line]) -> None:
        for line in lines:
            size = computed_size(line)
            if size is None:
                msg = f"cannot size line: {str(line)!r}"
                raise ValueError(msg)
            line.size = size


class AnchorSizer:
    """Sizes anchored lines by ``#_<hex>`` adjacency (exact for a disassembly).

    An anchor's size is the gap to the next anchor in the list, so a block's
    final line is sized against the true following address. The last anchor
    falls back to :func:`data_size`. Lines without a live anchor get size 0.
    """

    def size_all(self, lines: list[Line]) -> None:
        prev_index = -1
        prev_address: int | None = None
        for index, line in enumerate(lines):
            line.size = 0
            address = line.address
            if address is None:
                continue
            if prev_index >= 0 and prev_address is not None:
                lines[prev_index].size = address - prev_address
            prev_index, prev_address = index, address
        if prev_index >= 0:
            lines[prev_index].size = data_size(lines[prev_index])


class HybridSizer:
    """Anchor adjacency where available, computed sizing elsewhere.

    Runs :class:`AnchorSizer` first (exact for the disassembly, where every
    emitter carries its own ``#_<hex>`` anchor), then fills any byte-emitting
    line still at size 0 via :func:`computed_size` -- an inserted or
    arbitrary-asar line, or the last anchor of a run (whose adjacency has no
    successor to measure against). Interior anchored emitters keep their exact
    adjacency size, so a parsed disassembly is byte-for-byte unchanged. Unknown
    opcodes are left at 0 (lenient); use :class:`ComputedSizer` for a strict,
    anchor-free pass.
    """

    def size_all(self, lines: list[Line]) -> None:
        AnchorSizer().size_all(lines)
        for line in lines:
            if line.size:
                continue
            size = computed_size(line)
            if size:
                line.size = size
