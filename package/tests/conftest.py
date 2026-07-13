"""Shared fixtures and pytest configuration for pyaslreport tests.

Fixtures provided:
    minimal_nifti_path: Path to a tiny on-disk NIfTI file (auto-deleted after test).
    make_context: Factory for building ProcessingContext with sensible defaults.
    make_processor: Factory for building ASLProcessor without triggering validation.
    examples_dir: Path to the integration examples directory (pytest-configurable).
    minimal_asl_json: In-memory dict representing a minimal valid ASL JSON.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable

import nibabel as nib
import numpy as np
import pytest

from pyaslreport.modalities.asl.processor import ASLProcessor, ProcessingContext


# ---------------------------------------------------------------------------
# CLI option for integration test directory
# ---------------------------------------------------------------------------
def pytest_addoption(parser: pytest.Parser) -> None:
    """Register the --examples-dir CLI option for the integration runner.

    Args:
        parser: The pytest CLI parser, supplied by pytest at collection time.

    Notes:
        Local usage:  pytest --examples-dir=/path/to/examples
        CI usage:     omit the flag; defaults to the committed set in
                      tests/integration/examples.
    """
    parser.addoption(
        "--examples-dir",
        action="store",
        default=None,
        help="Path to integration examples dir; falls back to the committed CI set.",
    )
    parser.addoption(
        "--update-expected",
        action="store_true",
        default=False,
        help="Regenerate each example's expected_output.json from current tool output.",
    )


# ---------------------------------------------------------------------------
# File-based fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def minimal_nifti_path(tmp_path: Path) -> Path:
    """Create a tiny valid NIfTI file at tmp_path/asl.nii.gz.

    Args:
        tmp_path: Pytest-provided per-test temporary directory.

    Returns:
        Path to the written NIfTI file with shape (4, 4, 20, 2). The third
        axis matches a sensible default for ProcessingContext.nifti_slice_number
        so most tests do not need a custom shape.
    """
    data = np.zeros((4, 4, 20, 2), dtype=np.float32)
    img = nib.Nifti1Image(data, affine=np.eye(4))

    path = tmp_path / "asl.nii.gz"
    nib.save(img, str(path))

    return path


@pytest.fixture
def examples_dir(request: pytest.FixtureRequest) -> Path:
    """Return the path to the integration examples directory.

    Args:
        request: Pytest fixture request, used to read the --examples-dir flag.

    Returns:
        Resolved path to an existing examples directory.

    Raises:
        pytest.skip.Exception: If the resolved path does not exist. Resolution
            order is (1) the --examples-dir CLI flag, then (2) the committed
            tests/integration/examples directory beside this conftest.
    """
    cli_value = request.config.getoption("--examples-dir")

    if cli_value:
        path = Path(cli_value).expanduser().resolve()
    else:
        path = Path(__file__).parent / "integration" / "examples"

    if not path.is_dir():
        pytest.skip(f"Examples directory not found: {path}")

    return path


# ---------------------------------------------------------------------------
# Factory fixtures
# ---------------------------------------------------------------------------
@pytest.fixture
def make_context() -> Callable[..., ProcessingContext]:
    """Return a factory for building ProcessingContext with sensible defaults.

    Returns:
        A callable that accepts keyword overrides and returns a fully-formed
        ProcessingContext. All ten required fields receive defaults; pass
        keyword arguments to override any of them.

    Example:
        >>> def test_something(make_context):
        ...     ctx = make_context(m0_type="Separate", errors=["something"])
        ...     assert ctx.m0_type == "Separate"
    """

    def _make(**overrides: Any) -> ProcessingContext:
        defaults: dict[str, Any] = {
            "asl_json_data": [],
            "m0_prep_times_collection": [],
            "errors": [],
            "warnings": [],
            "all_absent": True,
            "bs_all_off": True,
            "m0_type": None,
            "global_pattern": None,
            "total_acquired_pairs": None,
            "nifti_slice_number": 20,
        }

        defaults.update(overrides)

        return ProcessingContext(**defaults)

    return _make


@pytest.fixture
def make_processor(minimal_nifti_path: Path) -> Callable[..., ASLProcessor]:
    """Return a factory for building ASLProcessor without input validation.

    Args:
        minimal_nifti_path: Auto-injected fixture providing a real on-disk
            NIfTI file. The factory uses it as the default nifti_file unless
            the caller overrides it.

    Returns:
        A callable that accepts keyword overrides and returns an ASLProcessor.
        Useful for testing private methods like _group_files in isolation,
        because BaseProcessor.__init__ stores self.data without validating.

    Example:
        >>> def test_grouping(make_processor):
        ...     proc = make_processor(files=["/path/to/asl.json"])
        ...     groups = proc._group_files("nifti")
    """

    def _make(**overrides: Any) -> ASLProcessor:
        defaults: dict[str, Any] = {
            "modality": "asl",
            "files": [],
            "dcm_files": [],
            "nifti_file": str(minimal_nifti_path),
        }

        defaults.update(overrides)

        return ASLProcessor(defaults)

    return _make


# ---------------------------------------------------------------------------
# Data fixtures (in-memory JSON dicts)
# ---------------------------------------------------------------------------
@pytest.fixture
def minimal_asl_json() -> dict[str, Any]:
    """Return a minimal ASL JSON with all major-error fields valid.

    Returns:
        Dictionary intended as a starting point for normalization and
        validation tests. Spread it into a new dict and override individual
        fields to introduce specific missing or invalid values.
    """
    return {
        "ArterialSpinLabelingType": "PCASL",
        "MRAcquisitionType": "3D",
        "PulseSequenceType": "GRASE",
        "M0Type": "Separate",
        "BackgroundSuppression": False,
        "PostLabelingDelay": 1.8,
        "LabelingDuration": 1.8,
        "EchoTime": 0.012,
        "RepetitionTimePreparation": 4.0,
        "FlipAngle": 90,
        "MagneticFieldStrength": 3,
        "Manufacturer": "Siemens",
        "ManufacturersModelName": "TrioTim",
        "AcquisitionVoxelSize": [3, 3, 4],
    }


# ---------------------------------------------------------------------------
# Integration example discovery (parametrizes test_integration over cases)
# ---------------------------------------------------------------------------
def pytest_generate_tests(metafunc: pytest.Metafunc) -> None:
    """Parametrize integration tests over discovered example case folders.

    Reads --examples-dir (registered above): when absent, falls back to the
    committed set at tests/integration/examples. An empty or missing dir yields
    zero cases, which pytest reports as a single skip rather than an error.

    Args:
        metafunc: The pytest metafunc for the test being collected.
    """
    if "integration_case" in metafunc.fixturenames:
        from tests.integration.runner import discover_examples

        cli = metafunc.config.getoption("--examples-dir")
        base = (
            Path(cli).expanduser().resolve()
            if cli
            else Path(__file__).parent / "integration" / "examples"
        )
        cases = discover_examples(base)
        metafunc.parametrize("integration_case", cases, ids=[p.name for p in cases])
