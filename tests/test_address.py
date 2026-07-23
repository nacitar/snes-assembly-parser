"""Tests for :class:`snes_assembly_parser.address.Address`."""

from __future__ import annotations

import pytest

from snes_assembly_parser.address import Address, Indicator


@pytest.mark.parametrize(
    ("label", "indicator", "offset", "tag", "width"),
    [
        ("#_0EC440", None, 0x0EC440, "", 6),
        ("#_0EC440o", None, 0x0EC440, "o", 6),
        ("#_008205", None, 0x008205, "", 6),
        ("NULL_0EEDFB", Indicator.NULL, 0x0EEDFB, "", 6),
        ("UNREACHABLE_0ED3CF", Indicator.UNREACHABLE, 0x0ED3CF, "", 6),
        ("#UNREACHABLE_0CFDF9", Indicator.UNREACHABLE, 0x0CFDF9, "", 6),
    ],
)
def test_parse_fields(
    label: str, indicator: Indicator | None, offset: int, tag: str, width: int
) -> None:
    address = Address.parse(label)
    assert address is not None
    assert address.indicator is indicator
    assert address.offset == offset
    assert address.tag == tag
    assert address.width == width


@pytest.mark.parametrize(
    "label",
    ["#Module0E_02_RenderText", "RenderText", "pool", ".loop", "", "  "],
)
def test_non_addresses_parse_to_none(label: str) -> None:
    assert Address.parse(label) is None


def test_parse_of_none_is_none() -> None:
    assert Address.parse(None) is None


@pytest.mark.parametrize(
    "label", ["#_0EC440", "#_0EC440o", "NULL_0EEDFB", "UNREACHABLE_0ED3CF"]
)
def test_render_round_trips(label: str) -> None:
    address = Address.parse(label)
    assert address is not None
    assert address.render() == label


def test_marker_render_drops_leading_hash() -> None:
    # A ``#``-prefixed marker renders bare (markers take no ``#`` in output);
    # the Line keeps the raw string for its own exact round trip.
    address = Address.parse("#UNREACHABLE_0CFDF9")
    assert address is not None
    assert address.render() == "UNREACHABLE_0CFDF9"


def test_at_rebases_keeping_width_and_tag() -> None:
    address = Address.parse("#_0EC440o")
    assert address is not None
    moved = address.at(0x2EC440)
    assert moved.render() == "#_2EC440o"


def test_is_anchor_vs_marker() -> None:
    live = Address.parse("#_0EC440")
    null = Address.parse("NULL_0EEDFB")
    assert live is not None and live.is_anchor and not live.is_marker
    assert null is not None and null.is_marker and not null.is_anchor
