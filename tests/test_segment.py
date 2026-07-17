"""Tests for parse-time byte sizing, extraction, and the editable Segment."""

from __future__ import annotations

import pytest

from snes_assembly_parser.segment import code, data, note
from snes_assembly_parser.source import Block, Line, Pool, Source, data_size

# Two adjacent routines with known address gaps, so sizes are exact. Alpha
# spans 018000..018007, Beta 018007..01800A; Gamma is the boundary whose first
# anchor gives Beta's last line its size via full-source adjacency.
BASE = [
    "Alpha:",  # 0
    "#_018000: LDA.w #$0000",  # 1  size 3
    "#_018003: STA.w $2100",  # 2  size 3
    "#_018006: RTS",  # 3  size 1
    "",  # 4
    ";---",  # 5
    "Beta:",  # 6
    "#_018007: NOP",  # 7  size 1
    "#_018008: RTL",  # 8  size 2 (to Gamma's 01800A)
    "",  # 9
    "Gamma:",  # 10
    "#_01800A: RTS",  # 11
]


def anchors(rendered: str) -> list[str]:
    return [line for line in rendered.splitlines() if line.startswith("#_")]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("#_1C8000: db $00, $01, $02", 3),
        ("#_1C8000: dw $0000, $0001", 4),
        ("#_1C8000: dl $000000", 3),
        ("#_1C8000: LDA $00", 0),  # not a data directive
        ("Foo:", 0),
    ],
)
def test_data_size(text: str, *, expected: int) -> None:
    assert data_size(Line.from_line(text)) == expected


def test_sizes_are_populated_at_parse_time() -> None:
    source = Source.from_content(BASE)
    assert [line.size for line in source.lines[1:4]] == [3, 3, 1]
    assert source.lines[0].size == 0  # label, not an anchor
    assert source.lines[8].size == 2  # sized across the Beta/Gamma boundary


def test_block_carries_sizes_and_addresses() -> None:
    segment = Source.from_content(BASE).block("Alpha", comments=False)
    assert (segment.start_address, segment.end_address) == (0x018000, 0x018007)
    assert [line.size for line in segment.lines] == [0, 3, 3, 1]


def test_adjacent_blocks_tile_exactly() -> None:
    source = Source.from_content(BASE)
    alpha = source.block("Alpha", comments=False)
    beta = source.block("Beta", comments=False)
    assert alpha.end_address == beta.start_address == 0x018007


def test_render_round_trip_is_exact() -> None:
    segment = Source.from_content(BASE).block("Alpha", comments=False)
    assert segment.render(0x018000) == "\n".join(BASE[0:4])


def test_render_relocates_every_anchor() -> None:
    segment = Source.from_content(BASE).region("Alpha", "Beta")
    shifted = anchors(segment.render(0x018000 + 0x200000))
    assert shifted[0] == "#_218000: LDA.w #$0000"
    assert shifted[-1] == "#_218008: RTL"


def test_insert_shifts_only_downstream_anchors() -> None:
    segment = Source.from_content(BASE).region("Alpha", "Beta")
    segment.insert_after("#_018000", [code("#_000000: JSR $9999", 6)])
    got = anchors(segment.render(0x018000))
    assert got[0] == "#_018000: LDA.w #$0000"  # before insert: unchanged
    assert got[1] == "#_018003: JSR $9999"  # inserted, stamped at pc
    assert got[2] == "#_018009: STA.w $2100"  # was 018003, now +6
    assert got[-1] == "#_01800E: RTL"  # was 018008, now +6


def test_all_extractors_return_independent_copies() -> None:
    source = Source.from_content(BASE)
    before = [str(line) for line in source.lines]
    source.block("Alpha", comments=True).render(0x300000)
    source.region("Alpha", "Beta").render(0x300000)
    source.concat(["Alpha", "Beta"]).render(0x300000)
    source.blocks_until("Alpha").render(0x300000)
    assert [str(line) for line in source.lines] == before


def test_pool_extractors_return_independent_copies() -> None:
    source = Source.from_content(POOLED)
    before = [str(line) for line in source.lines]
    source.pool("PerformVWFing", comments=True).render(0x300000)
    source.concat(
        [Block("DrawChar"), Pool("PerformVWFing"), Block("PerformVWFing")]
    ).render(0x300000)
    assert [str(line) for line in source.lines] == before


# --- Explicit lists exclude dead code without a drop flag -------------------

DEAD = [
    "Alpha:",  # 0
    "#_018000: LDA #$00",  # 1  size 2
    "#_018002: RTS",  # 2  size 1 (to the dead block's anchor)
    "UNREACHABLE_018003:",  # 3
    "#_018003: NOP",  # 4  dead
    "#_018004: NOP",  # 5  dead
    "Beta:",  # 6
    "#_018005: RTL",  # 7
    "Gamma:",  # 8
    "#_018006: RTS",  # 9
]


def test_concat_drops_dead_and_reserves_gap_with_org() -> None:
    rendered = (
        Source.from_content(DEAD).concat(["Alpha", "Beta"]).render(0x018000)
    )
    got = anchors(rendered)
    assert "#_018003: NOP" not in rendered  # dead block's bytes/label dropped
    # The 2 dead bytes are reserved by an org, so Beta keeps its own 018005
    # (not collapsed to 018003).
    assert "org $018005" in rendered
    assert got == ["#_018000: LDA #$00", "#_018002: RTS", "#_018005: RTL"]


def test_concat_adjacent_blocks_emit_no_org() -> None:
    # Alpha ends at 018007 and Beta begins there -- byte-adjacent, so no gap
    # and no org; the two blocks simply flow together.
    rendered = (
        Source.from_content(BASE).concat(["Alpha", "Beta"]).render(0x018000)
    )
    assert "org $" not in rendered
    assert anchors(rendered) == [
        "#_018000: LDA.w #$0000",
        "#_018003: STA.w $2100",
        "#_018006: RTS",
        "#_018007: NOP",
        "#_018008: RTL",
    ]


def test_concat_rejects_unexplained_live_label() -> None:
    # Skipping the live Beta between Alpha and Gamma must fail loudly.
    with pytest.raises(ValueError, match="unnamed content"):
        Source.from_content(DEAD).concat(["Alpha", "Gamma"])


def test_concat_allows_dead_label_in_gap() -> None:
    # The UNREACHABLE_ block between Alpha and Beta is permitted.
    segment = Source.from_content(DEAD).concat(["Alpha", "Beta"])
    assert segment.start_address == 0x018000


# --- concat with declared pools --------------------------------------------

# DrawChar, then a `pool PerformVWFing` data block (no top-level label), then a
# PerformVWFing routine that shares the pool's name -- mirroring bank_0E.
POOLED = [
    "DrawChar:",  # 0
    "#_00E000: LDA #$00",  # 1  size 2
    "#_00E002: RTS",  # 2  size 1 (to the pool's anchor)
    "",  # 3
    ";---",  # 4
    "pool PerformVWFing",  # 5  pool start (no top-level label)
    ".width",  # 6
    "#_00E003: db $01, $02",  # 7  2 bytes
    ".masks",  # 8
    "#_00E005: db $FF",  # 9  1 byte
    "pool off",  # 10
    "",  # 11
    "PerformVWFing:",  # 12  block sharing the pool's name
    "#_00E006: NOP",  # 13  size 1
    "#_00E007: RTL",  # 14  size 1 (to Next's anchor)
    "Next:",  # 15
    "#_00E008: RTS",  # 16
]


def test_declared_pool_is_included() -> None:
    source = Source.from_content(POOLED)
    items: list[Block | Pool | str] = [
        Block("DrawChar"),
        Pool("PerformVWFing"),
        Block("PerformVWFing"),
    ]
    segment = source.concat(items)
    region = source.region("DrawChar", "PerformVWFing")
    # Pool bytes are present, so the footprint matches the contiguous region.
    assert segment.start_address == region.start_address
    assert segment.end_address == region.end_address
    start = segment.start_address
    assert start is not None
    rendered = segment.render(start)
    assert "pool PerformVWFing" in rendered
    assert ".width" in rendered
    assert "#_00E003: db $01, $02" in rendered


def test_forgotten_pool_is_a_loud_error() -> None:
    source = Source.from_content(POOLED)
    with pytest.raises(ValueError, match="unnamed content"):
        source.concat([Block("DrawChar"), Block("PerformVWFing")])


def test_bare_string_is_treated_as_block() -> None:
    source = Source.from_content(BASE)
    assert source.concat(["Alpha", "Beta"]).render(0x018000) == source.concat(
        [Block("Alpha"), Block("Beta")]
    ).render(0x018000)


def test_concat_reserving_dead_keeps_end_address() -> None:
    source = Source.from_content(DEAD)
    kept = source.concat([Block("Alpha"), Block("Beta")])
    region = source.region("Alpha", "Beta")
    dead = source.block("UNREACHABLE_018003", comments=False)
    dead_size = sum(line.size for line in dead.lines)
    assert dead_size == 2
    # The dropped dead block's space is reserved with an org, so the footprint
    # equals the contiguous region -- the end is NOT shrunk by the dead size.
    assert kept.end_address == region.end_address


# --- blocks_until: data runs that stop at a marker --------------------------


def test_blocks_until_stops_before_null_and_spans_rules() -> None:
    table = [
        "Msg0:",
        "#_1C8000: db $7F",
        ";---",  # a rule between every message; must not truncate
        "Msg1:",
        "#_1C8001: db $7A, $00",
        ";---",
        "#_1C8003: db $76",  # sized against the NULL pad anchor below
        "NULL_1C8004:",  # stop marker (free ROM)
        "#_1C8004: db $FF",
    ]
    segment = Source.from_content(table).blocks_until("Msg0")
    assert not any(line.is_null_label for line in segment.lines)
    assert segment.lines[-1].size == 1  # 1C8004 - 1C8003, from the pad anchor
    assert segment.end_address == 0x1C8004


def test_blocks_until_to_eof_sizes_last_line_via_data_size() -> None:
    table = ["Msg0:", "#_1C8000: db $7F", ";---", "#_1C8001: db $7A, $00"]
    segment = Source.from_content(table).blocks_until("Msg0")
    assert segment.lines[-1].size == 2  # data_size(db $7A, $00)
    assert segment.end_address == 0x1C8003


# --- org contiguity assert --------------------------------------------------


def test_span_crossing_org_raises() -> None:
    crossing = [
        "Alpha:",
        "#_018000: RTS",
        "org $028000",
        "Beta:",
        "#_028000: RTS",
    ]
    with pytest.raises(ValueError, match="org"):
        Source.from_content(crossing).region("Alpha", "Beta")


# --- edit API ---------------------------------------------------------------


def test_replace_requires_exact_count() -> None:
    segment = Source.from_content(BASE).block("Alpha", comments=False)
    segment.replace("$0000", "$00FF", count=1)
    assert "#_018000: LDA.w #$00FF" in segment.render(0x018000)


def test_replace_wrong_count_fails_loud() -> None:
    segment = Source.from_content(BASE).block("Alpha", comments=False)
    with pytest.raises(ValueError, match="expected 2"):
        segment.replace("$0000", "$00FF", count=2)


def test_annotate_appends_comment() -> None:
    segment = Source.from_content(BASE).block("Alpha", comments=False)
    segment.annotate("STA.w", "video port")
    assert "STA.w $2100 ; video port" in segment.render(0x018000)


def test_delete_by_label_removes_block_and_collapses() -> None:
    table = [
        "M0:",
        "#_1C8000: db $01",
        "M1:",
        "#_1C8001: db $02",
        "M2:",
        "#_1C8002: db $03",
    ]
    segment = Source.from_content(table).blocks_until("M0")
    segment.delete("M1:", "M2:")  # match on the label-with-colon
    rendered = segment.render(0x1C8000)
    assert "M1:" not in rendered
    assert "db $02" not in rendered
    assert "#_1C8001: db $03" in rendered  # M2 collapsed back by 1 byte


def test_insert_and_append_with_constructors() -> None:
    segment = Source.from_content(BASE).block("Alpha", comments=False)
    segment.insert_after("#_018000", [data("#_000000: db $AA, $BB")])
    segment.append([note("; end of Alpha")])
    rendered = segment.render(0x018000)
    assert "#_018003: db $AA, $BB" in rendered  # data auto-sized to 2
    assert anchors(rendered)[-1] == "#_018008: RTS"  # RTS drifted +2
    assert rendered.endswith("; end of Alpha")


def test_constructors_size_lines() -> None:
    assert data("#_000000: db $01, $02").size == 2
    assert code("#_000000: JSR $1234", 3).size == 3
    assert note("; comment").size == 0
