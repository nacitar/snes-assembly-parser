"""Tests for :class:`snes_assembly_parser.assembly.Assembly`."""

from __future__ import annotations

import pytest

from snes_assembly_parser.assembly import (
    Assembly,
    dbr_trampolines,
    instructions,
)

SAMPLE = [
    "; header for Foo",
    "Foo:",
    "#_008000: LDA.w #$0000",
    "#_008003: JSR Bar",
    "#_008006: RTS",
    "",
    "Bar:",
    "#_008007: LDA.w Table,Y",
    "#_00800A: RTL",
    "",
    "pool Table",
    ".data",
    "#_00800B: dw $0000, $0001",
    "pool off",
]


@pytest.fixture
def asm() -> Assembly:
    return Assembly.from_content(SAMPLE)


def test_indexes_functions_and_pools(asm: Assembly) -> None:
    assert asm.functions == ["Foo", "Bar"]
    assert "Table" in asm.pools


def test_function_extract_copies(asm: Assembly) -> None:
    foo = asm.function("Foo", comments=True)
    assert foo.render().splitlines()[0] == "; header for Foo"
    # editing the copy does not touch the parent
    foo.replace("LDA.w #$0000", "LDA.w #$FFFF", count=1)
    assert "FFFF" not in asm.render()


def test_offset_shifts_anchors(asm: Assembly) -> None:
    foo = asm.function("Foo")
    foo.offset(0x200000)
    assert foo.start_address == 0x208000
    assert "#_208000: LDA.w #$0000" in foo.render()


def test_suffix_renames_defs_and_refs(asm: Assembly) -> None:
    asm.suffix(["Bar", "Table"], "", prefix="EN_")
    text = asm.render()
    assert "EN_Bar:" in text
    assert "JSR EN_Bar" in text
    assert "LDA.w EN_Table,Y" in text
    assert "pool EN_Table" in text


def test_validate_flags_mismatched_anchor() -> None:
    bad = Assembly.from_content(
        ["#_008000: PHB", "#_009999: PHK"]  # 2nd anchor is wrong (should be 1)
    )
    mismatches = bad.validate()
    assert len(mismatches) == 1
    stated, computed, _ = mismatches[0]
    assert stated == 0x009999
    assert computed == 0x008001


def test_validate_clean_when_consistent(asm: Assembly) -> None:
    assert asm.validate() == []


def test_comment_block_read(asm: Assembly) -> None:
    header = asm.comment_block("Foo")
    assert [str(line) for line in header] == ["; header for Foo"]


def test_insert_uses_computed_sizes(asm: Assembly) -> None:
    foo = asm.function("Foo")
    foo.insert_after("LDA.w #$0000", instructions(["DEX", "DEX"]))
    # +2 bytes (two 1-byte DEX): RTS was at $008006, now at $008008
    assert "#_008008: RTS" in foo.render()


def test_replace_all_batches(asm: Assembly) -> None:
    asm.replace_all([("LDA.w #$0000", "LDA.w #$0001", 1), ("RTS", "NOP", 1)])
    text = asm.render()
    assert "LDA.w #$0001" in text
    assert "NOP" in text


def test_render_relocates(asm: Assembly) -> None:
    foo = asm.function("Foo")
    assert "#_108000: LDA.w #$0000" in foo.render(0x108000)


def _routine() -> Assembly:
    return Assembly.from_content(
        ["Foo:", "#_008000: LDA.w #$0000", "#_008003: RTS"]
    )


def _opcodes(asm: Assembly) -> list[str]:
    return [line.opcode for line in asm.lines if line.opcode]


def test_return_long_rewrites_terminal_rts() -> None:
    asm = _routine()
    asm.return_long()
    assert _opcodes(asm)[-1] == "RTL"  # RTS -> RTL


def test_return_long_restore_bank_pulls_bank_then_returns() -> None:
    asm = _routine()
    asm.return_long(restore_bank=True)
    # RTS -> PLB (restore the trampoline's pushed data bank) + a fresh RTL.
    assert _opcodes(asm)[-2:] == ["PLB", "RTL"]


def test_return_long_raises_without_rts() -> None:
    asm = Assembly.from_content(["Foo:", "#_008000: RTL"])
    with pytest.raises(ValueError, match="does not end in RTS"):
        asm.return_long()


def test_dbr_trampolines_builds_entry_stubs() -> None:
    text = dbr_trampolines(["Foo", "Bar"]).render(0x2D8000)
    assert "Foo:" in text and "Bar:" in text
    assert "PHB" in text and "PHK" in text and "PLB" in text
    assert "JMP.w Foo_body" in text and "JMP.w Bar_body" in text
