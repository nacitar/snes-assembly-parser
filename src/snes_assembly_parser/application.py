"""Command-line entry point for snes-assembly-parser."""

from __future__ import annotations

import argparse
import importlib.metadata
import logging
from pathlib import Path
from typing import TYPE_CHECKING

from sevaht_utility.log_utility import add_log_arguments, configure_logging

from .source import Source

if TYPE_CHECKING:
    from collections.abc import Sequence

logger = logging.getLogger(__name__)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=importlib.metadata.metadata(__package__).get("summary")
    )
    add_log_arguments(parser)
    args = parser.parse_args(args=argv)
    configure_logging(args)

    source = Source.from_path(
        Path.home() / "scm/personal/alttp-jpdasm/usdasm/bank_0E.asm"
    )
    segment = source.block("OverworldOverlay_MiseryMire", comments=True)
    print(segment.render(segment.start_address or 0))

    return 0
