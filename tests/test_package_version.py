from snes_assembly_parser import __version__


def test_version_defined() -> None:
    assert bool(__version__)
