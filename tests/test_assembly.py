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
    foo = asm.block("Foo", comments=True)
    assert foo.render().splitlines()[0] == "; header for Foo"
    # editing the copy does not touch the parent
    foo.replace("LDA.w #$0000", "LDA.w #$FFFF", count=1)
    assert "FFFF" not in asm.render()


def test_offset_shifts_anchors(asm: Assembly) -> None:
    foo = asm.block("Foo")
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
    foo = asm.block("Foo")
    foo.insert_after("LDA.w #$0000", instructions(["DEX", "DEX"]))
    # +2 bytes (two 1-byte DEX): RTS was at $008006, now at $008008
    assert "#_008008: RTS" in foo.render()


def test_replace_all_batches(asm: Assembly) -> None:
    asm.replace_all([("LDA.w #$0000", "LDA.w #$0001", 1), ("RTS", "NOP", 1)])
    text = asm.render()
    assert "LDA.w #$0001" in text
    assert "NOP" in text


def test_render_relocates(asm: Assembly) -> None:
    foo = asm.block("Foo")
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


def test_splice_replaces_a_single_line() -> None:
    asm = Assembly.from_content(SAMPLE)
    asm.splice("LDA.w #$0000", ["#_008000: NOP"])
    text = asm.render()
    assert "LDA.w #$0000" not in text
    assert "#_008000: NOP" in text


def test_splice_replaces_a_range_excluding_stop() -> None:
    asm = Assembly.from_content(SAMPLE)
    asm.splice("LDA.w #$0000", ["#_008000: NOP"], until="RTS")
    text = asm.render()
    # both the LDA and the JSR in [first, until) are gone; RTS survives
    assert "JSR Bar" not in text
    assert "RTS" in text
    assert "#_008000: NOP" in text


def test_delete_single_line_without_until() -> None:
    asm = Assembly.from_content(SAMPLE)
    asm.delete("JSR Bar")
    assert "JSR Bar" not in asm.render()
    assert "LDA.w #$0000" in asm.render()  # neighbours untouched


HASH_SAMPLE = [
    "Enclosing:",
    "#_00C000: LDA.b #$03",
    "#Helper:",
    "#_00C002: STA.b $14",
    ".exit",
    "#_00C004: RTS",
    "",
    "Next:",
    "#_00C005: RTL",
]


def test_block_pulls_scope_transparent_hash_label() -> None:
    asm = Assembly.from_content(HASH_SAMPLE)
    helper = asm.block("Helper")
    text = helper.render()
    # the # is dropped so it is a standalone, namespaceable block
    assert text.splitlines()[0].startswith("Helper:")
    assert "#Helper:" not in text
    # spans only Helper's body, up to the next top-level label
    assert "STA.b $14" in text
    assert "RTL" not in text  # did not run into Next
    # a #-label is not a top-level boundary, so Enclosing is unaffected
    assert "Helper" not in asm.functions


def test_apply_edits_mixes_tuples_and_callables() -> None:
    asm = Assembly.from_content(SAMPLE)

    def to_nop(block: Assembly) -> None:
        block.replace("JSR Bar", "NOP", 1)

    asm.apply_edits([("LDA.w #$0000", "LDA.w #$FFFF", 1), to_nop])
    text = asm.render()
    assert "LDA.w #$FFFF" in text  # tuple replace
    assert "JSR Bar" not in text and "NOP" in text  # callable


def test_apply_edit_table_applies_every_block() -> None:
    asm = Assembly.from_content(SAMPLE)
    asm.apply_edit_table(
        {
            "Foo": [("LDA.w #$0000", "LDA.w #$FFFF", 1)],
            "Bar": [("LDA.w Table,Y", "LDA.w Other,Y", 1)],
        }
    )
    text = asm.render()
    assert "LDA.w #$FFFF" in text
    assert "LDA.w Other,Y" in text


def test_subblock_pulls_sublabel_span_by_name() -> None:
    asm = Assembly.from_content(
        [
            "Enclosing:",
            ".not_mode7",
            "#_008200: LDA.w $0128",
            "#_008205: LDA.w TIMEUP",
            "#_008208: STA.w VTIMEL",
            "",
            ".IRQ_inactive",
            "#_00821B: LDA.b $13",
        ]
    )
    frag = asm.subblock(".not_mode7")
    text = frag.render()
    assert text.splitlines()[0] == ".not_mode7"
    assert "LDA.w TIMEUP" in text and "STA.w VTIMEL" in text
    # stops at the next same-level sublabel; trailing blank trimmed
    assert ".IRQ_inactive" not in text
    assert "LDA.b $13" not in text
    assert frag.lines[-1].opcode == "STA.w"


def test_subblock_raises_when_missing() -> None:
    asm = Assembly.from_content(["Foo:", "#_008000: RTS"])
    with pytest.raises(KeyError, match="no sublabel"):
        asm.subblock(".nope")


def test_region_at_pulls_address_range_and_trims_trailing() -> None:
    asm = Assembly.from_content(
        [
            "Enclosing:",
            "#_008200: SEP #$20",
            "#_008205: LDA.w TIMEUP",
            "#_008208: STA.w VTIMEL",
            "",
            "; trailing comment of the next block",
            ".next",
            "#_00821B: LDA.b $13",
        ]
    )
    frag = asm.region_at(0x008205, 0x00821B)
    text = frag.render()
    assert "LDA.w TIMEUP" in text and "STA.w VTIMEL" in text
    assert "SEP #$20" not in text  # started at 0x008205
    # trailing blank/comment/sublabel inside the span are dropped
    assert ".next" not in text
    assert "LDA.b $13" not in text  # stop is exclusive
    assert frag.lines[-1].opcode == "STA.w"


def test_region_at_raises_when_unanchored() -> None:
    asm = Assembly.from_content(["Foo:", "#_008000: RTS"])
    with pytest.raises(KeyError, match="anchored"):
        asm.region_at(0x009999, 0x00999A)


def test_address_of_returns_first_anchor(asm: Assembly) -> None:
    # label on its own line; the address is the code it names (next anchor)
    assert asm.address_of("Foo") == 0x008000
    assert asm.address_of("Bar") == 0x008007


def test_address_of_raises_for_unknown_label(asm: Assembly) -> None:
    with pytest.raises(KeyError):
        asm.address_of("Nope")
