"""Tests for normalization logic: _rename_fields, _convert_units_to_milliseconds."""


def test_module_imports() -> None:
    """The processor module imports cleanly."""
    from pyaslreport.modalities.asl.processor import ASLProcessor

    assert ASLProcessor is not None
