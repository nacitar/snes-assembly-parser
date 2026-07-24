"""Function-interception (hooking) primitives shared by :class:`Assembly` and
:class:`Rom`.

Hooking means freeing an original routine's name so a relocated copy can claim
it, and -- when same-bank ``JSR`` callers cannot reach the copy across banks --
filling a free-ROM hole with a small forwarding stub (a *landing pad*)
under the
freed name. This module holds the value types and pure line-builders for that;
the addressing and mutation live on :class:`Assembly`/:class:`Rom`.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from .source import Line

#: Prefix that frees a routine's name (the disassembly's "dead code" marker).
UNREACHABLE_PREFIX = "UNREACHABLE_"
#: Prefix marking a free-ROM hole.
NULL_PREFIX = "NULL_"
#: A full-width separator rule between blocks.
RULE = ";" + "=" * 99

# The forwarding stub a landing pad emits: a register-transparent JSL/RTS
# bridge. A same-bank ``JSR name`` lands here; the ``JSL`` calls the relocated
# body across the bank (pushing a 3-byte return to this pad), the body returns
# with ``RTL`` to the ``RTS``, which pops the JSR's 2-byte return to its
# caller.
# No register is touched, so a caller's argument in A/X/Y passes straight
# through. 5 bytes (JSL 4 + RTS 1).
_STUB_OPCODE_SIZES = {"JSL": 4, "RTS": 1}


def anchor_label(address: int) -> str:
    """The ``#_<hex>`` address-anchor label for ``address``."""
    return f"#_{address:06X}"


@dataclass(frozen=True)
class LandingPad:
    """A forwarding stub: repoint the freed name ``name`` to ``target``.

    The relocated body ``target`` (in another bank) must end in ``RTL``.
    ``comment`` lines are emitted just under the pad's label.
    """

    name: str
    target: str
    comment: tuple[str, ...] = field(default=())


def landing_pad_lines(
    pads: list[LandingPad], base_address: int
) -> tuple[list[Line], int]:
    """Build the stub lines for ``pads`` laid out from ``base_address``.

    Returns the lines and the program counter just past the last stub, so the
    caller can back-fill the remaining free-ROM tail.
    """
    lines: list[Line] = []
    program_counter = base_address
    for pad in pads:
        lines.append(Line.from_line(f"{pad.name}:"))
        lines.extend(Line.from_line(text) for text in pad.comment)
        for opcode, operand in (("JSL", pad.target), ("RTS", "")):
            text = f"{opcode} {operand}".rstrip()
            lines.append(
                Line.from_line(f"{anchor_label(program_counter)}: {text}")
            )
            program_counter += _STUB_OPCODE_SIZES[opcode]
    return lines, program_counter


def byte_rows(address: int, values: list[int], per_row: int = 8) -> list[Line]:
    """``db`` rows for ``values`` (``per_row`` per line), anchored from
    ``address``."""
    rows: list[Line] = []
    for offset in range(0, len(values), per_row):
        chunk = values[offset : offset + per_row]
        body = ", ".join(f"${value:02X}" for value in chunk)
        rows.append(
            Line.from_line(f"{anchor_label(address + offset)}: db {body}")
        )
    return rows


def free_block(address: int, size: int) -> list[Line]:
    """A ``NULL_`` free-ROM marker of ``size`` ``$FF`` bytes."""
    header = [
        Line.from_line(RULE),
        Line.from_line(f"; FREE ROM: 0x{size:X}"),
        Line.from_line(RULE),
        Line.from_line(f"{NULL_PREFIX}{address:06X}:"),
    ]
    return header + byte_rows(address, [0xFF] * size)
