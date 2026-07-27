"""Tests for :class:`snes_assembly_parser.rom.Rom`."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

from snes_assembly_parser.rom import Caller, Placed, Rom

if TYPE_CHECKING:
    from pathlib import Path

# A tiny two-bank program: main.asm includes both banks. bank_00 defines
# Alpha (reached by a cross-bank JSL and a data pointer) and a free-ROM hole;
# bank_01 defines Beta (reached by a same-bank JSR) and calls Alpha.
MAIN = 'incsrc "bank_00.asm"\nincsrc "bank_01.asm"\n'
# Alpha is reached by a cross-bank JSL and a data pointer (dl); a NULL_ hole
# marks free ROM; PointerTable holds the pointer.
BANK_00 = """\
Alpha:
#_008000: LDA.w #$0000
#_008003: RTL

;---
NULL_008004:
#_008004: db $FF, $FF, $FF, $FF

PointerTable:
#_008008: dl Alpha
"""
# Beta self-calls with a same-bank JSR and reaches Alpha with a cross-bank JSL.
BANK_01 = """\
Beta:
#_018000: JSR Beta
#_018003: JSL Alpha
#_018007: RTS
"""


@pytest.fixture
def program(tmp_path: Path) -> Path:
    (tmp_path / "main.asm").write_text(MAIN)
    (tmp_path / "bank_00.asm").write_text(BANK_00)
    (tmp_path / "bank_01.asm").write_text(BANK_01)
    return tmp_path / "main.asm"


def test_load_follows_incsrc(program: Path) -> None:
    rom = Rom.load(program)
    # entry + the two included banks
    assert len(rom.units) == 3
    assert program.resolve() in rom.units


def test_functions_across_units(program: Path) -> None:
    rom = Rom.load(program)
    assert "Alpha" in rom.functions
    assert "Beta" in rom.functions
    assert "PointerTable" in rom.functions


def test_unit_of(program: Path) -> None:
    rom = Rom.load(program)
    assert rom.unit_of("Alpha").name == "bank_00.asm"
    assert rom.unit_of("Beta").name == "bank_01.asm"
    with pytest.raises(KeyError):
        rom.unit_of("Nonexistent")


def test_callers_classifies_bank_local_vs_cross_bank(program: Path) -> None:
    rom = Rom.load(program)
    callers = rom.callers("Alpha")
    opcodes = sorted(c.opcode for c in callers)
    # a cross-bank JSL and a data pointer (dl), no bank-local caller
    assert opcodes == ["JSL", "dl"]
    assert not any(c.is_bank_local for c in callers)

    beta = rom.callers("Beta")
    assert [c.opcode for c in beta] == ["JSR"]
    assert beta[0].is_bank_local


def test_caller_is_bank_local() -> None:
    assert Caller("x", "JSR").is_bank_local  # type: ignore[arg-type]
    assert Caller("x", "BRA").is_bank_local  # type: ignore[arg-type]
    assert not Caller("x", "JSL").is_bank_local  # type: ignore[arg-type]
    assert not Caller("x", "dl").is_bank_local  # type: ignore[arg-type]


def test_free_regions(program: Path) -> None:
    rom = Rom.load(program)
    regions = rom.free_regions()
    assert len(regions) == 1
    path, address, size = regions[0]
    assert path.name == "bank_00.asm"
    assert address == 0x008004
    assert size == 4


def test_rename_leaves_callers_untouched(program: Path) -> None:
    rom = Rom.load(program)
    rom.rename("Alpha", "UNREACHABLE_Alpha")
    # the definition moved...
    assert "UNREACHABLE_Alpha" in rom.functions
    assert "Alpha" not in rom.units[rom.unit_of("UNREACHABLE_Alpha")].labels
    # ...but the JSL/dl references still say "Alpha"
    assert any(c.opcode == "JSL" for c in rom.callers("Alpha"))


def test_hook_frees_the_name_and_leaves_callers(program: Path) -> None:
    rom = Rom.load(program)
    rom.hook("Alpha", comment=("; relocated to bank $2N",))
    unit = rom.units[rom.unit_of("UNREACHABLE_Alpha")]
    # the definition is freed (marker prefix), with the comment above it...
    assert "UNREACHABLE_Alpha" in unit.labels
    assert "Alpha" not in unit.labels
    assert any(
        line.comment and "relocated" in line.comment for line in unit.lines
    )
    # ...and the JSL / dl references are untouched (resolve to the new copy)
    assert any(c.opcode == "JSL" for c in rom.callers("Alpha"))


def test_write_roundtrips_units_preserves_subdirs_and_banks(
    tmp_path: Path,
) -> None:
    root = tmp_path / "src"
    (root / "resources").mkdir(parents=True)
    (root / "main.asm").write_text(
        'incsrc "bank_00.asm"\nincsrc "resources/data.asm"\n'
    )
    (root / "bank_00.asm").write_text(BANK_00)
    # a support file in a subdir, deliberately WITHOUT a trailing newline
    (root / "resources" / "data.asm").write_text("Data:\n#_038000: db $01")
    rom = Rom.load(root / "main.asm")

    class _Piece:
        org = 0x208000

        def render(self) -> str:
            return "; relocated\norg $208000\nEN_Foo:\n#_208000: RTL\n"

    class _Reloc:
        def placements(self) -> list[Placed]:
            return [_Piece()]

    rom.add(_Reloc())
    out = tmp_path / "out"
    rom.write(out)
    # unchanged units round-trip byte-for-byte: subdir kept, no newline added
    assert (out / "bank_00.asm").read_text() == BANK_00
    assert (
        out / "resources" / "data.asm"
    ).read_text() == "Data:\n#_038000: db $01"
    # the added piece lands in a generated bank named for its org's bank
    generated = (out / "bank_20.asm").read_text()
    assert "; relocated" in generated
    assert "org $208000" in generated


def test_incsrc_cycle_detected(tmp_path: Path) -> None:
    (tmp_path / "a.asm").write_text('incsrc "b.asm"\n')
    (tmp_path / "b.asm").write_text('incsrc "a.asm"\n')
    with pytest.raises(ValueError, match="cycle"):
        Rom.load(tmp_path / "a.asm")


def test_reinclude_is_deduplicated(tmp_path: Path) -> None:
    # a.asm and b.asm both include shared.asm: loaded once, no cycle error.
    (tmp_path / "main.asm").write_text('incsrc "a.asm"\nincsrc "b.asm"\n')
    (tmp_path / "a.asm").write_text('incsrc "shared.asm"\n')
    (tmp_path / "b.asm").write_text('incsrc "shared.asm"\n')
    (tmp_path / "shared.asm").write_text("Shared:\n#_028000: RTL\n")
    rom = Rom.load(tmp_path / "main.asm")
    shared = [p for p in rom.units if p.name == "shared.asm"]
    assert len(shared) == 1
