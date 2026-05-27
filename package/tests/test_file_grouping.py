"""Tests for _group_files and FileReader behavior."""

from typing import Callable

from pyaslreport.modalities.asl.processor import ASLProcessor


def test_make_processor_factory(make_processor: Callable[..., ASLProcessor]) -> None:
    """The make_processor fixture produces an ASLProcessor with the given files."""
    proc = make_processor(files=[])
    assert proc.data["files"] == []
