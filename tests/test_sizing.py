"""Tests for :mod:`snes_assembly_parser.sizing`."""

from __future__ import annotations

import pytest

from snes_assembly_parser.sizing import (
    AnchorSizer,
    ComputedSizer,
    HybridSizer,
    computed_size,
    data_size,
)
from snes_assembly_parser.source import Line


@pytest.mark.parametrize(
    ("text", "size"),
    [
        # explicit-width suffix: 1 opcode byte + {b:1, w:2, l:3}
        ("LDA.b #$06", 2),
        ("STA.b $14", 2),
        ("STA.w $0AB6", 3),
        ("LDA.w #$0004", 3),
        ("LDA.l $7003E1,X", 4),
        # no-suffix control/stack/branch table
        ("DEX", 1),
        ("PLB", 1),
        ("RTS", 1),
        ("RTL", 1),
        ("REP #$20", 2),
        ("BNE .loop", 2),
        ("JMP.w Foo", 3),
        ("JSR Foo", 3),
        ("JML EN_x", 4),
        ("JSL EN_x", 4),
        # data + non-emitting
        ("db $EA, $EA", 2),
        ("dw $018B", 2),
        ("org $3FFFFF", 0),
        ("warnpc $00821B", 0),
        ("Foo:", 0),
        ("; comment", 0),
        ("", 0),
    ],
)
def test_computed_size(text: str, size: int) -> None:
    assert computed_size(Line.from_line(text)) == size


def test_computed_size_unknown_opcode_is_none() -> None:
    assert computed_size(Line.from_line("FOOBAR $12")) is None


def test_data_size_widths() -> None:
    assert data_size(Line.from_line("db $01, $02, $03")) == 3
    assert data_size(Line.from_line("dw $0001, $0002")) == 4
    assert data_size(Line.from_line("dl $010000")) == 3
    assert data_size(Line.from_line("LDA.w #$00")) == 0


def test_anchor_sizer_uses_adjacency() -> None:
    lines = [
        Line.from_line(t)
        for t in ["#_008000: PHB", "#_008001: PHK", "#_008002: PLB"]
    ]
    AnchorSizer().size_all(lines)
    assert [line.size for line in lines] == [
        1,
        1,
        0,
    ]  # last falls to data_size


def test_computed_sizer_needs_no_anchors() -> None:
    lines = [Line.from_line(t) for t in ["PHB", "PHK", "PLB", "RTL"]]
    ComputedSizer().size_all(lines)
    assert [line.size for line in lines] == [1, 1, 1, 1]


def test_computed_sizer_raises_on_unknown() -> None:
    with pytest.raises(ValueError, match="cannot size"):
        ComputedSizer().size_all([Line.from_line("FOOBAR $12")])


def test_hybrid_fills_anchorless_emitters() -> None:
    # A mix: anchored lines keep adjacency; the trailing anchor-less RTL is
    # computed-filled.
    lines = [
        Line.from_line(t) for t in ["#_008000: PHB", "#_008001: PHK", "RTL"]
    ]
    HybridSizer().size_all(lines)
    assert [line.size for line in lines] == [1, 1, 1]
