"""Tests for M0 contradiction and TSV validation branches in ASLProcessor."""

from typing import Callable

from pyaslreport.modalities.asl.processor import ProcessingContext


def test_make_context_factory(make_context: Callable[..., ProcessingContext]) -> None:
    """The make_context fixture produces a ProcessingContext with overrides applied."""
    ctx = make_context(m0_type="Separate")
    assert ctx.m0_type == "Separate"
    assert ctx.errors == []
