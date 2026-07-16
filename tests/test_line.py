"""Tests for :class:`snes_assembly_parser.source.Line`."""

from __future__ import annotations

import pytest

from snes_assembly_parser.source import Line

# Lines that must survive a parse -> str round trip byte-for-byte. These cover
# indentation, colon labels, colon-less sublabels, operand spacing quirks,
# nested/quoted operands, comments, and blank/comment-only lines.
ROUND_TRIP = [
    "",
    "   ",
    "\t",
    "; a standalone comment",
    "   ; indented comment",
    "Foo:",
    "Foo:   ",
    ".skip_sfx",
    ".end",
    ".sub:",
    '#_0E8000: incbin "bin/gfx/font.2bpp"',
    "#_0E9000: dw $0000, $0000, $0080",
    "#_0E9000: dw   $0000 ,$0001,  $0002",
    "#_0E9885: JSR (.vectors,X)",
    "  LDA.w #$0030   ; load value",
    "  RTS",
    "Label: LDA.b $11 ; inline",
    "pool CheckForSpecialOverworldTrigger",
    "pool off",
]


@pytest.mark.parametrize("text", ROUND_TRIP)
def test_round_trip_is_exact(text: str) -> None:
    assert str(Line.from_line(text)) == text


def test_colon_label_fields() -> None:
    line = Line.from_line("#_0E9000: dw $0000, $0080")
    assert line.label == "#_0E9000"
    assert line.label_colon is True
    assert line.opcode == "dw"
    assert line.arguments == ["$0000", "$0080"]
    assert line.comment is None


def test_colon_less_sublabel_is_a_label() -> None:
    line = Line.from_line(".skip_sfx")
    assert line.label == ".skip_sfx"
    assert line.label_colon is False
    assert line.opcode is None


def test_comment_captured_without_semicolon() -> None:
    line = Line.from_line("  LDA.w #$30 ; go")
    assert line.opcode == "LDA.w"
    assert line.arguments == ["#$30"]
    assert line.comment == " go"


def test_indexed_indirect_comma_is_not_a_separator() -> None:
    assert Line.from_line("JSR (.vectors,X)").arguments == ["(.vectors,X)"]


def test_string_operand_comma_is_not_a_separator() -> None:
    assert Line.from_line('incbin "a,b.bin"').arguments == ['"a,b.bin"']


def test_no_operands_yields_empty_arguments() -> None:
    line = Line.from_line("  RTS")
    assert line.opcode == "RTS"
    assert line.arguments == []
    assert line.arg_seps == []


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Foo:", True),
        ("Routine_Name:", True),
        (".sublabel", False),
        ("#_0E8000:", False),
        ("  LDA $00", False),
        ("", False),
    ],
)
def test_is_top_level_label(text: str, *, expected: bool) -> None:
    assert Line.from_line(text).is_top_level_label is expected


@pytest.mark.parametrize(
    ("text", "content", "blank"),
    [
        ("Foo:", True, False),
        ("  RTS", True, False),
        ("; comment", False, False),
        ("", False, True),
        ("   ", False, True),
    ],
)
def test_content_and_blank_flags(
    text: str, *, content: bool, blank: bool
) -> None:
    line = Line.from_line(text)
    assert line.has_content is content
    assert line.is_blank is blank


@pytest.mark.parametrize(
    ("text", "is_address", "address"),
    [
        ("#_0EC440: LDA $00", True, 0x0EC440),
        ("#_3C00: dw $1234", True, 0x3C00),
        ("#_D000o: dw Song01", True, 0xD000),  # APU bank tag ignored
        ("#Module0E_02_RenderText:", False, None),  # named #, not an address
        ("RenderText:", False, None),
        ("NULL_0BFE5E:", False, None),
        ("  LDA $00", False, None),
    ],
)
def test_address_label(
    text: str, *, is_address: bool, address: int | None
) -> None:
    line = Line.from_line(text)
    assert line.is_address_label is is_address
    assert line.address == address


def test_set_address_preserves_width_tag_and_round_trip() -> None:
    line = Line.from_line("#_0EC440: LDA $00")
    line.set_address(0x0EC446)
    assert str(line) == "#_0EC446: LDA $00"  # 6-wide, formatting intact
    apu = Line.from_line("#_3C00o: dw $1234")
    apu.set_address(0x3C02)
    assert str(apu) == "#_3C02o: dw $1234"  # 4-wide, tag kept


def test_set_address_on_non_anchor_raises() -> None:
    with pytest.raises(ValueError, match="no address label"):
        Line.from_line("RenderText:").set_address(0x1234)


@pytest.mark.parametrize(
    ("text", "null", "unreachable"),
    [
        ("NULL_0BFE5E:", True, False),
        ("UNREACHABLE_0ABAB6:", False, True),
        ("#UNREACHABLE_0CFDF9:", False, True),  # may be #-prefixed
        ("#_0EC440: LDA $00", False, False),
        ("Foo:", False, False),
    ],
)
def test_null_and_unreachable_labels(
    text: str, *, null: bool, unreachable: bool
) -> None:
    line = Line.from_line(text)
    assert line.is_null_label is null
    assert line.is_unreachable_label is unreachable
