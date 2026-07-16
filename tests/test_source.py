"""Tests for :class:`snes_assembly_parser.source.Source`."""

from __future__ import annotations

import pytest

from snes_assembly_parser.source import (
    Line,
    Source,
    leading_comments,
    trim_trailing,
)

# A small but representative source. Index annotations are relied on by the
# tests below, so keep them in sync if you edit the list.
SAMPLE = [
    ";===============",  # 0
    "; header for Foo",  # 1
    ";---------------",  # 2
    "Foo:",  # 3
    "  LDA.w #$00",  # 4
    ".loop",  # 5  (sublabel, not top-level)
    "  DEX",  # 6
    "  BNE .loop",  # 7
    "  RTS",  # 8
    "",  # 9
    ";===============",  # 10
    "",  # 11
    "pool Bar",  # 12
    ".data",  # 13 (sublabel scoped to Bar)
    "  dw $0000, $0001",  # 14
    "pool off",  # 15
    "",  # 16
    "Bar:",  # 17
    "  RTL",  # 18
]


@pytest.fixture
def source() -> Source:
    return Source.from_content(SAMPLE)


def as_text(lines: list[Line]) -> list[str]:
    return [str(line) for line in lines]


def test_from_content_matches_from_lines() -> None:
    from_content = Source.from_content(SAMPLE)
    from_lines = Source.from_lines(Line.from_line(t) for t in SAMPLE)
    assert as_text(from_content.lines) == as_text(from_lines.lines)


def test_labels_are_top_level_only(source: Source) -> None:
    # Sublabels (.loop, .data) and the pool name are excluded.
    assert source.labels == {"Foo": 3, "Bar": 17}


def test_pools_span_directive_to_directive(source: Source) -> None:
    assert source.pools == {"Bar": (12, 16)}


def test_block_stops_at_pool_boundary_and_trims(source: Source) -> None:
    # Foo's next boundary is the pool at 12, and trailing blank/comment
    # lines (9, 10, 11) are trimmed away.
    assert as_text(source.block("Foo", comments=False)) == [
        "Foo:",
        "  LDA.w #$00",
        ".loop",
        "  DEX",
        "  BNE .loop",
        "  RTS",
    ]


def test_block_with_comments_prepends_header(source: Source) -> None:
    got = as_text(source.block("Foo", comments=True))
    assert got[:4] == [
        ";===============",
        "; header for Foo",
        ";---------------",
        "Foo:",
    ]


def test_block_to_end_of_source(source: Source) -> None:
    assert as_text(source.block("Bar", comments=False)) == ["Bar:", "  RTL"]


def test_pool_body(source: Source) -> None:
    assert as_text(source.pool("Bar", comments=False)) == [
        "pool Bar",
        ".data",
        "  dw $0000, $0001",
        "pool off",
    ]


def test_pool_with_comments_keeps_interior_blank(source: Source) -> None:
    # Leading blank (9) is dropped; the blank after the divider (11) is kept.
    got = as_text(source.pool("Bar", comments=True))
    assert got[:3] == [";===============", "", "pool Bar"]


def test_missing_label_raises(source: Source) -> None:
    with pytest.raises(KeyError):
        source.block("Nope", comments=False)


def test_missing_pool_raises(source: Source) -> None:
    with pytest.raises(KeyError):
        source.pool("Nope", comments=False)


def test_comments_is_required_keyword() -> None:
    source = Source.from_content(SAMPLE)
    with pytest.raises(TypeError):
        source.block("Foo")  # type: ignore[call-arg]
    with pytest.raises(TypeError):
        source.pool("Bar")  # type: ignore[call-arg]


def test_reindex_after_mutation(source: Source) -> None:
    source.lines.append(Line.from_line("Baz:"))
    source.lines.append(Line.from_line("  RTL"))
    assert "Baz" not in source.labels  # not seen until reindex
    source.reindex()
    assert source.labels["Baz"] == len(source.lines) - 2


def test_trim_trailing_keeps_interior_blanks() -> None:
    lines = [Line.from_line(t) for t in ["A:", "", "  RTS", "", "; tail"]]
    assert as_text(trim_trailing(lines)) == ["A:", "", "  RTS"]


def test_leading_comments_drops_leading_blanks_only() -> None:
    lines = [Line.from_line(t) for t in ["  RTS", "", "; h1", "", "Foo:"]]
    # Walk back from Foo: (index 4): blank(1) dropped, comment/blank kept.
    assert as_text(leading_comments(lines, 4)) == ["; h1", ""]
