"""
Shared pytest configuration, fixtures, and markers for the pyaslreport test suite.
"""

import json
import pytest
import nibabel as nib
import numpy as np
from pathlib import Path


def pytest_addoption(parser):
    """
    Register the custom CLI option for the examples directory.

    Args:
        parser: Pytest parser object.
    """
    parser.addoption(
        "--examples-dir",
        action="store",
        default=None,
        help="Path to external examples directory for integration tests"
    )


def pytest_configure(config):
    """
    Register custom pytest markers.

    Args:
        config: Pytest configuration object.
    """
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (run with -m integration)"
    )


@pytest.fixture
def minimal_nifti_path(tmp_path):
    """
    Create a minimal valid NIfTI file for tests.

    Args:
        tmp_path: Temporary path provided by pytest.

    Returns:
        String path to the generated NIfTI file.
    """
    path = tmp_path / "asl.nii"

    # Create a 64x64x20 NIfTI image with zero values
    img = nib.Nifti1Image(np.zeros((64, 64, 20)), affine=np.eye(4))
    nib.save(img, str(path))

    return str(path)


@pytest.fixture
def make_context():
    """
    Create a factory for ProcessingContext instances.

    Returns:
        Factory function that creates ProcessingContext objects with optional overrides.
    """
    from pyaslreport.modalities.asl.processor import ProcessingContext

    def _make(**kwargs):
        """
        Build a ProcessingContext object.

        Args:
            **kwargs: Values to override default context fields.

        Returns:
            ProcessingContext instance.
        """
        defaults = dict(
            asl_json_data=[],
            m0_prep_times_collection=[],
            errors=[],
            warnings=[],
            all_absent=True,
            bs_all_off=True,
            m0_type=None,
            global_pattern=None,
            total_acquired_pairs=None,
            nifti_slice_number=20,
        )
        defaults.update(kwargs)
        return ProcessingContext(**defaults)

    return _make


@pytest.fixture
def make_processor():
    """
    Create a factory for ASLProcessor instances.

    Returns:
        Factory function that creates ASLProcessor objects with optional data overrides.
    """
    from pyaslreport.modalities.asl.processor import ASLProcessor

    def _make(**data_overrides):
        """
        Build an ASLProcessor object.

        Args:
            **data_overrides: Values to override default processor data.

        Returns:
            ASLProcessor instance.
        """
        data = {"files": [], "nifti_file": None, "dcm_files": []}
        data.update(data_overrides)
        return ASLProcessor(data)

    return _make


@pytest.fixture
def examples_dir(request):
    """
    Resolve the examples directory for integration tests.

    Args:
        request: Pytest request object.

    Returns:
        Path to the examples directory.
    """
    custom = request.config.getoption("--examples-dir")

    if custom:
        return Path(custom)

    return Path(__file__).parent / "examples"