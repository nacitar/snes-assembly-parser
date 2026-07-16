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
