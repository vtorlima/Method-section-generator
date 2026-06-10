"""Tests for the six validator classes plus schema loading."""

from typing import Any

from pyaslreport.core.config import config
from pyaslreport.modalities.asl.validators import (
    BooleanValidator,
    ConsistencyValidator,
    NumberArrayValidator,
    NumberOrNumberArrayValidator,
    NumberValidator,
    StringValidator,
)

MAJOR_SLOT, ERROR_SLOT, WARNING_SLOT = 0, 2, 4


def fired_major(r: tuple[Any, ...]) -> bool:
    """True if the major-error slot is populated."""
    return r[MAJOR_SLOT] is not None


def fired_error(r: tuple[Any, ...]) -> bool:
    """True if the (non-major) error slot is populated."""
    return r[ERROR_SLOT] is not None


def fired_warning(r: tuple[Any, ...]) -> bool:
    """True if the warning slot is populated."""
    return r[WARNING_SLOT] is not None


def all_clear(r: tuple[Any, ...]) -> bool:
    """True if no slot is populated."""
    return all(x is None for x in r)


class TestNumberValidator:
    def test_within_bounds(self) -> None:
        assert all_clear(NumberValidator(min_error=0, max_error=10).validate(5))

    def test_min_error_is_strict(self) -> None:
        v = NumberValidator(min_error=0)
        assert v.validate(0)[ERROR_SLOT] == "Value must be > 0"
        assert all_clear(v.validate(0.0001))

    def test_above_max_error(self) -> None:
        assert fired_error(NumberValidator(max_error=10).validate(20))

    def test_min_error_include_is_inclusive(self) -> None:
        v = NumberValidator(min_error_include=0)
        assert all_clear(v.validate(0))
        assert fired_error(v.validate(-1))

    def test_max_error_include_boundary(self) -> None:
        v = NumberValidator(max_error_include=360)
        assert all_clear(v.validate(360))
        assert v.validate(360.001)[ERROR_SLOT] == "Value must be <= 360"

    def test_warning_threshold(self) -> None:
        assert fired_warning(NumberValidator(min_warning=0).validate(-1))

    def test_enforce_integer_alone(self) -> None:
        v = NumberValidator(enforce_integer=True)
        assert all_clear(v.validate(5))
        assert fired_error(v.validate(5.5))

    def test_enforce_integer_rule_precedes_range(self) -> None:
        """Integer rule is added first, so it fires before the range check."""
        v = NumberValidator(min_error=0, enforce_integer=True)
        assert v.validate(2.5)[ERROR_SLOT] == "Value must be an integer"
        assert v.validate(-1)[ERROR_SLOT] == "Value must be > 0"
        assert all_clear(v.validate(3))


class TestStringValidator:
    def test_allowed_passes(self) -> None:
        assert all_clear(
            StringValidator(allowed_values=["PCASL", "PASL"]).validate("PCASL")
        )

    def test_case_insensitive(self) -> None:
        assert all_clear(StringValidator(allowed_values=["PCASL"]).validate("pcasl"))

    def test_disallowed_routes_to_error(self) -> None:
        r = StringValidator(allowed_values=["PCASL"]).validate("XYZ")
        assert fired_error(r) and not fired_major(r)

    def test_major_flag_routes_to_major(self) -> None:
        r = StringValidator(allowed_values=["PCASL"], major_error=True).validate("XYZ")
        assert fired_major(r) and not fired_error(r)

    def test_no_allowed_values_accepts_anything(self) -> None:
        assert all_clear(StringValidator().validate("anything"))


class TestBooleanValidator:
    def test_true_false_pass(self) -> None:
        assert all_clear(BooleanValidator().validate(True))
        assert all_clear(BooleanValidator().validate(False))

    def test_string_errors(self) -> None:
        assert fired_error(BooleanValidator().validate("true"))

    def test_int_errors(self) -> None:
        """isinstance(1, bool) is False, so 1 is rejected."""
        assert fired_error(BooleanValidator().validate(1))


class TestNumberArrayValidator:
    def test_exact_size(self) -> None:
        v = NumberArrayValidator(size_error=3)
        assert all_clear(v.validate([1, 2, 3]))
        assert fired_error(v.validate([1, 2]))
        assert fired_error(v.validate([1, 2, "x"]))

    def test_min_error_skips_non_numbers(self) -> None:
        """Range rule's guarded comprehension ignores non-numeric elements."""
        v = NumberArrayValidator(min_error=0)
        assert fired_error(v.validate([1, -1, 2]))
        assert all_clear(v.validate([1, 2, "x"]))

    def test_ascending_allows_equal_neighbors(self) -> None:
        v = NumberArrayValidator(check_ascending=True)
        assert all_clear(v.validate([1, 1, 2]))
        assert fired_error(v.validate([3, 1, 2]))


class TestNumberOrNumberArrayValidator:
    def test_scalar_dispatch(self) -> None:
        v = NumberOrNumberArrayValidator(min_error=0)
        assert fired_error(v.validate(-1))
        assert all_clear(v.validate(5))

    def test_array_dispatch(self) -> None:
        v = NumberOrNumberArrayValidator(min_error=0)
        assert fired_error(v.validate([1, -1, 2]))
        assert all_clear(v.validate([1, 2, 3]))

    def test_wrong_type_reports_in_major_slot(self) -> None:
        """Type mismatch lands in slot 0; pins current (arguably odd) behavior."""
        assert NumberOrNumberArrayValidator().validate("x")[MAJOR_SLOT] is not None


class TestConsistencyValidator:
    def test_string_same_passes(self) -> None:
        v = ConsistencyValidator(validation_type="string")
        assert all_clear(v.validate([("PCASL", "a"), ("PCASL", "b")]))

    def test_string_mismatch_error_tier(self) -> None:
        v = ConsistencyValidator(validation_type="string", is_major=False)
        assert fired_error(v.validate([("PCASL", "a"), ("PASL", "b")]))

    def test_string_mismatch_major_tier(self) -> None:
        v = ConsistencyValidator(validation_type="string", is_major=True)
        assert fired_major(v.validate([("PCASL", "a"), ("PASL", "b")]))

    def test_float_within_warning(self) -> None:
        v = ConsistencyValidator(
            "floatOrArray", error_variation=10, warning_variation=0.1
        )
        assert all_clear(v.validate([(2000, "a"), (2000.05, "b")]))

    def test_float_crosses_warning(self) -> None:
        v = ConsistencyValidator(
            "floatOrArray", error_variation=10, warning_variation=0.1
        )
        assert fired_warning(v.validate([(2000, "a"), (2000.5, "b")]))

    def test_float_crosses_error(self) -> None:
        v = ConsistencyValidator(
            "floatOrArray", error_variation=10, warning_variation=0.1
        )
        assert fired_error(v.validate([(2000, "a"), (2020, "b")]))

    def test_boolean_same_passes(self) -> None:
        assert all_clear(
            ConsistencyValidator("boolean").validate([(True, "a"), (True, "b")])
        )

    def test_boolean_mixed_errors(self) -> None:
        assert fired_error(
            ConsistencyValidator("boolean").validate([(True, "a"), (False, "b")])
        )


def test_all_expected_schemas_loaded() -> None:
    """config['schemas'] contains every shipped schema."""
    expected = {
        "major_error_schema",
        "required_validator_schema",
        "required_condition_schema",
        "recommended_validator_schema",
        "recommended_condition_schema",
        "consistency_schema",
    }
    assert expected.issubset(config["schemas"].keys())


def test_major_error_schema_fields() -> None:
    """The four documented major-error fields are present."""
    schema = config["schemas"]["major_error_schema"]
    for field in (
        "PLDType",
        "ArterialSpinLabelingType",
        "MRAcquisitionType",
        "PulseSequenceType",
    ):
        assert field in schema
