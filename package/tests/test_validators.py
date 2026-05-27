"""Tests for the six validator classes in modalities/asl/validators/."""


def test_module_imports() -> None:
    """All six validator classes import cleanly from the package."""
    from pyaslreport.modalities.asl.validators import (
        BooleanValidator,
        ConsistencyValidator,
        NumberArrayValidator,
        NumberOrNumberArrayValidator,
        NumberValidator,
        StringValidator,
    )

    assert all(
        [
            BooleanValidator,
            ConsistencyValidator,
            NumberArrayValidator,
            NumberOrNumberArrayValidator,
            NumberValidator,
            StringValidator,
        ]
    )
