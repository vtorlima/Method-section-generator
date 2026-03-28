import json
import pytest
import nibabel as nib
import numpy as np
from pathlib import Path


# --- CLI option for external examples directory ---

def pytest_addoption(parser):
    parser.addoption(
        "--examples-dir",
        action="store",
        default=None,
        help="Path to external examples directory for integration tests"
    )


# --- Pytest markers ---

def pytest_configure(config):
    config.addinivalue_line(
        "markers",
        "integration: marks tests as integration tests (run with -m integration)"
    )


# --- NIfTI generator fixture ---

@pytest.fixture
def minimal_nifti_path(tmp_path):
    """
    Generates a minimal valid NIfTI file at tmp_path/asl.nii.
    shape=(64, 64, 20) — nifti_slice_number in output will be 20.
    """
    path = tmp_path / "asl.nii"
    img = nib.Nifti1Image(np.zeros((64, 64, 20)), affine=np.eye(4))
    nib.save(img, str(path))
    return str(path)


# --- Minimal ProcessingContext factory ---

@pytest.fixture
def make_context():
    """
    Returns a factory function that creates a ProcessingContext with sensible defaults.
    Override any field by passing keyword arguments.
    """
    from pyaslreport.modalities.asl.processor import ProcessingContext

    def _make(**kwargs):
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


# --- Minimal ASLProcessor factory ---

@pytest.fixture
def make_processor():
    """
    Returns a factory function that creates an ASLProcessor with minimal data.
    Safe to call internal methods on.
    """
    from pyaslreport.modalities.asl.processor import ASLProcessor

    def _make(**data_overrides):
        data = {"files": [], "nifti_file": None, "dcm_files": []}
        data.update(data_overrides)
        return ASLProcessor(data)

    return _make


# --- Examples directory fixture for integration tests ---

@pytest.fixture
def examples_dir(request):
    """
    Returns the path to the examples directory.
    Uses --examples-dir CLI option if provided, otherwise the committed examples.
    """
    custom = request.config.getoption("--examples-dir")
    if custom:
        return Path(custom)
    return Path(__file__).parent / "examples"
