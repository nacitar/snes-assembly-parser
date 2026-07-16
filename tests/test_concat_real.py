"""Real-disassembly checks for pool-aware concat (skipped if unavailable).

These mirror the exact bank_0E structure that motivated declared pools: a
``pool RenderText_PerformVWFing`` data block sits between two named routines,
and an ``UNREACHABLE_`` block sits between two others.
"""

from __future__ import annotations

import pathlib

import pytest

from snes_assembly_parser.source import Block, Pool, Source

BANK_0E = pathlib.Path.home() / "scm/personal/alttp-jpdasm/usdasm/bank_0E.asm"

pytestmark = pytest.mark.skipif(
    not BANK_0E.exists(), reason="alttp-jpdasm disassembly not available"
)


@pytest.fixture(scope="module")
def bank_0e() -> Source:
    return Source.from_content(BANK_0E.read_text().splitlines())


def test_declared_pool_included_matches_region(bank_0e: Source) -> None:
    items: list[Block | Pool | str] = [
        Block("RenderText_DrawSingleCharacter"),
        Pool("RenderText_PerformVWFing"),
        Block("RenderText_PerformVWFing"),
    ]
    segment = bank_0e.concat(items)
    region = bank_0e.region(
        "RenderText_DrawSingleCharacter", "RenderText_PerformVWFing"
    )
    assert segment.start_address == region.start_address
    assert segment.end_address == region.end_address
    start = segment.start_address
    assert start is not None
    rendered = segment.render(start)
    assert "pool RenderText_PerformVWFing" in rendered
    assert ".width" in rendered


def test_forgotten_pool_errors(bank_0e: Source) -> None:
    with pytest.raises(ValueError, match="unnamed content"):
        bank_0e.concat(
            [
                Block("RenderText_DrawSingleCharacter"),
                Block("RenderText_PerformVWFing"),
            ]
        )


def test_dead_block_allowed_and_shrinks_end(bank_0e: Source) -> None:
    kept = bank_0e.concat(
        [
            Block("RenderText_PerformVWFing"),
            Block("RenderText_TickDownDrawDelay"),
        ]
    )
    region = bank_0e.region(
        "RenderText_PerformVWFing", "RenderText_TickDownDrawDelay"
    )
    dead = bank_0e.block("UNREACHABLE_0ECCF8", comments=False)
    dead_size = sum(line.size for line in dead.lines)
    assert dead_size > 0
    region_end = region.end_address
    assert region_end is not None
    assert kept.end_address == region_end - dead_size
