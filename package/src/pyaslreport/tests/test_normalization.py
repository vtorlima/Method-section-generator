from typing import Any, Dict

import pytest

from pyaslreport.modalities.asl.constants import DURATION_OF_EACH_RFBLOCK
from pyaslreport.modalities.asl.processor import ASLProcessor
from pyaslreport.utils.math_utils import MathUtils
from pyaslreport.utils.unit_conversion_utils import UnitConverterUtils


def make_processor() -> ASLProcessor:
    """
    Create an ASLProcessor instance for testing.

    Returns:
        ASLProcessor: A processor instance with empty configuration.
    """
    return ASLProcessor({"files": [], "nifti_file": None, "dcm_files": []})


# _rename_fields tests


@pytest.mark.parametrize(
    "old_key,new_key,value",
    [
        ("RepetitionTime", "RepetitionTimePreparation", 4.2),
        ("InversionTime", "PostLabelingDelay", 1.8),
        ("BolusDuration", "BolusCutOffDelayTime", 0.7),
        ("InitialPostLabelDelay", "PostLabelingDelay", 0.5),
    ],
)
def test_field_rename(old_key: str, new_key: str, value: float) -> None:
    """
    Test field renaming in session dictionary.

    The old key must be deleted and the new key must contain the original value.

    Args:
        old_key: Original field name in session.
        new_key: New field name expected after renaming.
        value: The value associated with the field.
    """
    processor = make_processor()
    session: Dict[str, Any] = {old_key: value}
    processor._rename_fields(session)
    assert new_key in session
    assert old_key not in session
    assert session[new_key] == value


def test_num_rf_blocks_produces_labeling_duration() -> None:
    """
    Test that NumRFBlocks is converted to LabelingDuration.

    The LabelingDuration should be computed as NumRFBlocks * DURATION_OF_EACH_RFBLOCK.
    """
    processor = make_processor()
    session: Dict[str, Any] = {"NumRFBlocks": 100}
    processor._rename_fields(session)
    assert "LabelingDuration" in session
    assert session["LabelingDuration"] == pytest.approx(
        100 * DURATION_OF_EACH_RFBLOCK
    )


def test_num_rf_blocks_original_key_not_deleted() -> None:
    """
    Test that NumRFBlocks key is preserved after computing LabelingDuration.

    Unlike field renames, NumRFBlocks should remain alongside the computed LabelingDuration.
    """
    processor = make_processor()
    session: Dict[str, Any] = {"NumRFBlocks": 100}
    processor._rename_fields(session)
    assert "NumRFBlocks" in session


def test_unrelated_field_not_modified() -> None:
    """
    Test that fields not in the mapping remain untouched.

    Fields that are not part of the normalization mapping should not be modified.
    """
    processor = make_processor()
    session: Dict[str, Any] = {"ArterialSpinLabelingType": "PCASL", "EchoTime": 0.037}
    processor._rename_fields(session)
    assert "ArterialSpinLabelingType" in session
    assert "EchoTime" in session


# _convert_units_to_milliseconds tests


@pytest.mark.parametrize(
    "field,input_val,expected",
    [
        ("EchoTime", 0.037, 37),
        ("RepetitionTimePreparation", 4.2, 4200),
        ("LabelingDuration", 1.8, 1800),
        ("BolusCutOffDelayTime", 0.7, 700),
        ("BackgroundSuppressionPulseTime", 0.15, 150),
        ("PostLabelingDelay", 1.8, 1800),
    ],
)
def test_unit_conversion_single_value(field: str, input_val: float, expected: float) -> None:
    """
    Test unit conversion from seconds to milliseconds for single values.

    All specified fields must be multiplied by 1000.

    Args:
        field: The field name to convert.
        input_val: The input value in seconds.
        expected: The expected value in milliseconds.
    """
    processor = make_processor()
    session: Dict[str, Any] = {field: input_val}
    processor._convert_units_to_milliseconds(session)
    assert session[field] == pytest.approx(expected)


def test_unit_conversion_list_value() -> None:
    """
    Test unit conversion for fields with list values.

    List values should have each element multiplied by 1000.
    """
    processor = make_processor()
    session: Dict[str, Any] = {"PostLabelingDelay": [1.8, 2.0]}
    processor._convert_units_to_milliseconds(session)
    assert session["PostLabelingDelay"] == pytest.approx([1800, 2000])


def test_unit_conversion_field_not_in_session_is_skipped() -> None:
    """
    Test that fields not present in session are skipped.

    The conversion should only affect fields that exist in the session.
    """
    processor = make_processor()
    session: Dict[str, Any] = {"ArterialSpinLabelingType": "PCASL"}
    processor._convert_units_to_milliseconds(session)
    assert "ArterialSpinLabelingType" in session
    assert len(session) == 1
