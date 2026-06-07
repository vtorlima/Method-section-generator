"""Tests for normalization logic in ASLProcessor."""

from typing import Callable

import pytest

from pyaslreport.modalities.asl.processor import ASLProcessor
from pyaslreport.utils.math_utils import MathUtils
from pyaslreport.utils.unit_conversion_utils import UnitConverterUtils

# ---------- _rename_fields ----------


@pytest.mark.parametrize(
    "old_key,new_key,value",
    [
        ("RepetitionTime", "RepetitionTimePreparation", 4.5),
        ("InversionTime", "PostLabelingDelay", 1.8),
        ("BolusDuration", "BolusCutOffDelayTime", 0.7),
        ("InitialPostLabelDelay", "PostLabelingDelay", 1.8),
    ],
)
def test_rename_fields_renames_and_deletes_legacy(
    make_processor: Callable[..., ASLProcessor],
    old_key: str,
    new_key: str,
    value: float,
) -> None:
    """Each of the four mappings copies old->new and deletes the old key."""
    proc = make_processor()
    session = {old_key: value}
    proc._rename_fields(session)
    assert session[new_key] == value
    assert old_key not in session


def test_rename_fields_numrfblocks_derives_labeling_duration(
    make_processor: Callable[..., ASLProcessor],
) -> None:
    """NumRFBlocks=100 derives LabelingDuration=1.84 (pins the numeric contract)."""
    proc = make_processor()
    session = {"NumRFBlocks": 100}
    proc._rename_fields(session)
    assert session["LabelingDuration"] == pytest.approx(1.84)


def test_rename_fields_numrfblocks_retains_source(
    make_processor: Callable[..., ASLProcessor],
) -> None:
    """CURRENT behavior: NumRFBlocks is NOT deleted after deriving LabelingDuration.

    Retention vs removal is an open contract question with mentors (provenance).
    If they decide to remove it, change this to `assert "NumRFBlocks" not in session`.
    """
    proc = make_processor()
    session = {"NumRFBlocks": 100}
    proc._rename_fields(session)
    assert "NumRFBlocks" in session


def test_rename_fields_no_legacy_keys_is_noop(
    make_processor: Callable[..., ASLProcessor],
) -> None:
    """A modern session with no legacy keys is unchanged."""
    proc = make_processor()
    session = {"PostLabelingDelay": 1.8, "RepetitionTimePreparation": 4.0}
    original = dict(session)
    proc._rename_fields(session)
    assert session == original


# ---------- _convert_units_to_milliseconds ----------

TIME_FIELDS = [
    "EchoTime",
    "RepetitionTimePreparation",
    "LabelingDuration",
    "BolusCutOffDelayTime",
    "BackgroundSuppressionPulseTime",
    "PostLabelingDelay",
]


@pytest.mark.parametrize("field", TIME_FIELDS)
def test_convert_scalar(
    make_processor: Callable[..., ASLProcessor], field: str
) -> None:
    """Each time field scalar is multiplied by 1000."""
    proc = make_processor()
    session = {field: 1.5}
    proc._convert_units_to_milliseconds(session)
    assert session[field] == 1500


@pytest.mark.parametrize("field", TIME_FIELDS)
def test_convert_list(make_processor: Callable[..., ASLProcessor], field: str) -> None:
    """Each time field list maps element-wise."""
    proc = make_processor()
    session = {field: [1.0, 2.0, 3.0]}
    proc._convert_units_to_milliseconds(session)
    assert session[field] == [1000, 2000, 3000]


def test_convert_leaves_non_time_fields(
    make_processor: Callable[..., ASLProcessor],
) -> None:
    """Non-time fields untouched; time field converted."""
    proc = make_processor()
    session = {"FlipAngle": 90, "EchoTime": 0.012}
    proc._convert_units_to_milliseconds(session)
    assert session["FlipAngle"] == 90
    assert session["EchoTime"] == 12


# ---------- MathUtils.round_if_close ----------


@pytest.mark.parametrize(
    "input_val,expected",
    [
        (2000.0, 2000),
        (2000.0000001, 2000),
        (2000.5, 2000.5),
        (1999.9999999, 2000),
        (2000.123456, 2000.123),
        (0.0, 0),
        (-2000.0, -2000),
    ],
)
def test_round_if_close(input_val: float, expected: int | float) -> None:
    """Snaps to int within 1e-6 of an integer; else rounds to 3 decimals."""
    result = MathUtils.round_if_close(input_val)
    assert result == expected
    if isinstance(expected, int):
        assert isinstance(result, int)


# ---------- UnitConverterUtils.convert_to_milliseconds ----------


def test_convert_to_ms_scalar() -> None:
    """A scalar in seconds becomes milliseconds (int when close to integer)."""
    assert UnitConverterUtils.convert_to_milliseconds(2.0) == 2000


def test_convert_to_ms_list() -> None:
    """A list maps element-wise."""
    assert UnitConverterUtils.convert_to_milliseconds([1.0, 2.0]) == [1000, 2000]


def test_convert_to_ms_rejects_string() -> None:
    """A non-numeric scalar raises TypeError."""
    with pytest.raises(TypeError):
        UnitConverterUtils.convert_to_milliseconds("not a number")


def test_convert_to_ms_rejects_list_with_string() -> None:
    """A list containing a non-number raises TypeError."""
    with pytest.raises(TypeError):
        UnitConverterUtils.convert_to_milliseconds([1.0, "two"])
