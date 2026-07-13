"""Example-based integration runner for the BIDS report path.

Two groups of tests live here:

1. ``test_example_matches_expected`` — parametrized over committed example
   folders (discovered in conftest). It is a golden / characterization test:
   the FULL 21-key output, including generated report text, must equal the
   committed expected_output.json. With ``--update-expected`` it rewrites the
   expected file instead of asserting. Until real examples are committed it
   collects zero cases and skips.

2. ``TestRunnerHarness`` — self-tests that synthesize a throwaway example in
   tmp_path and exercise discovery, input ordering, comparison, and the
   golden-update path. They need no committed fixtures, are NOT marked
   integration, and run in the unit job so the machinery is verified on every
   push regardless of whether any real example exists yet.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import nibabel as nib
import numpy as np
import pytest

from tests.integration.runner import (
    ExampleStructureError,
    PairingFallbackWarning,
    build_inputs,
    discover_examples,
    load_expected,
    normalize,
    run_example,
    write_expected,
)


# ---------------------------------------------------------------------------
# Golden test over committed examples
# ---------------------------------------------------------------------------
@pytest.mark.integration
def test_example_matches_expected(
    integration_case: Path, request: pytest.FixtureRequest
) -> None:
    """Full output (text included) equals expected_output.json, key by key.

    Args:
        integration_case: A discovered example directory.
        request: Used to read the --update-expected flag.
    """
    report = run_example(integration_case)

    if request.config.getoption("--update-expected"):
        write_expected(integration_case, report)
        pytest.skip(f"updated expected_output.json for {integration_case.name}")

    if not (integration_case / "expected_output.json").exists():
        pytest.fail(
            f"{integration_case.name}: no expected_output.json. "
            f"Generate it with: pytest -m integration --update-expected"
        )

    expected = load_expected(integration_case)
    actual = normalize(report)
    assert set(actual.keys()) == set(
        expected.keys()
    ), f"{integration_case.name}: key set drift"
    for key in sorted(expected.keys()):
        assert actual[key] == expected[key], f"{integration_case.name}: '{key}' differs"


# ---------------------------------------------------------------------------
# Harness self-tests (no committed fixtures required)
# ---------------------------------------------------------------------------
def _make_example(root: Path, *, slices: int = 18, with_m0: bool = True) -> Path:
    """Build a minimal single-acquisition BIDS example under root/case/sub-X/perf.

    Returns:
        The case directory (root/case).
    """
    case = root / "case"
    _write_acq(case / "sub-X" / "perf", "sub-X", slices=slices, with_m0=with_m0)
    return case


def _write_acq(
    perf: Path, prefix: str, *, slices: int = 18, with_m0: bool = True
) -> None:
    """Write one acquisition's asl/m0/tsv/nii into a perf dir under a prefix."""
    perf.mkdir(parents=True, exist_ok=True)
    asl = {
        "ArterialSpinLabelingType": "PCASL",
        "MRAcquisitionType": "3D",
        "PulseSequenceType": "GRASE",
        "M0Type": "Separate",
        "BackgroundSuppression": False,
        "EchoTime": 0.012,
        "RepetitionTimePreparation": 4.0,
        "FlipAngle": 90,
        "MagneticFieldStrength": 3,
        "Manufacturer": "Siemens",
        "PostLabelingDelay": 1.8,
        "LabelingDuration": 1.8,
    }
    (perf / f"{prefix}_asl.json").write_text(json.dumps(asl))
    if with_m0:
        (perf / f"{prefix}_m0scan.json").write_text(
            json.dumps(
                {
                    "EchoTime": 0.012,
                    "RepetitionTimePreparation": 4.0,
                    "M0Type": "Separate",
                }
            )
        )
    (perf / f"{prefix}_aslcontext.tsv").write_text(
        "volume_type\n" + "\n".join(["label", "control"] * 3) + "\n"
    )
    nib.save(
        nib.Nifti1Image(np.zeros((4, 4, slices, 2), dtype=np.float32), np.eye(4)),
        str(perf / f"{prefix}_asl.nii.gz"),
    )


class TestRunnerHarness:
    """Verify discovery, ordering, comparison, and golden-update end to end."""

    def test_discovery_keys_on_inputs_not_golden(self, tmp_path: Path) -> None:
        """A folder is a case as soon as it has inputs, before any golden exists.

        This is what lets --update-expected bootstrap a brand-new case.
        """
        case = _make_example(tmp_path)
        assert discover_examples(tmp_path) == [case]  # found with no expected yet
        assert not (case / "expected_output.json").exists()
        write_expected(case, run_example(case))
        assert discover_examples(tmp_path) == [case]  # still found after golden

    def test_discovery_handles_missing_dir(self, tmp_path: Path) -> None:
        """A non-existent base yields no cases (drives the empty-skip path)."""
        assert discover_examples(tmp_path / "nope") == []

    def test_build_inputs_orders_asl_before_m0(self, tmp_path: Path) -> None:
        """asl.json precedes m0scan.json precedes tsv (load-bearing order)."""
        case = _make_example(tmp_path)
        files, nifti = build_inputs(case)
        names = [Path(f).name for f in files]
        assert names == ["sub-X_asl.json", "sub-X_m0scan.json", "sub-X_aslcontext.tsv"]
        assert nifti.endswith("sub-X_asl.nii.gz")

    def test_multi_run_pairs_per_acquisition(self, tmp_path: Path) -> None:
        """Two runs in one perf dir interleave asl->m0->tsv per run, not by type."""
        case = tmp_path / "case"
        perf = case / "sub-X" / "perf"
        _write_acq(perf, "sub-X_run-1")
        _write_acq(perf, "sub-X_run-2")
        files, _ = build_inputs(case)
        names = [Path(f).name for f in files]
        assert names == [
            "sub-X_run-1_asl.json",
            "sub-X_run-1_m0scan.json",
            "sub-X_run-1_aslcontext.tsv",
            "sub-X_run-2_asl.json",
            "sub-X_run-2_m0scan.json",
            "sub-X_run-2_aslcontext.tsv",
        ]

    def test_multi_session_and_subject_supported(self, tmp_path: Path) -> None:
        """Sessions and subjects each contribute their perf acquisition, in order."""
        case = tmp_path / "case"
        _write_acq(case / "sub-A" / "ses-01" / "perf", "sub-A_ses-01")
        _write_acq(case / "sub-A" / "ses-02" / "perf", "sub-A_ses-02")
        _write_acq(case / "sub-B" / "perf", "sub-B")
        files, _ = build_inputs(case)
        asls = [Path(f).name for f in files if f.endswith("_asl.json")]
        assert asls == [
            "sub-A_ses-01_asl.json",
            "sub-A_ses-02_asl.json",
            "sub-B_asl.json",
        ]

    def test_more_asl_than_m0_reuses_shared(self, tmp_path: Path) -> None:
        """Two runs, one shared M0 with no run prefix: both reuse it (+ warning)."""
        case = tmp_path / "case"
        perf = case / "sub-X" / "perf"
        _write_acq(perf, "sub-X_run-1")
        _write_acq(perf, "sub-X_run-2")
        (perf / "sub-X_run-1_m0scan.json").unlink()
        (perf / "sub-X_run-2_m0scan.json").unlink()
        (perf / "sub-X_m0scan.json").write_text(
            json.dumps(
                {
                    "EchoTime": 0.012,
                    "RepetitionTimePreparation": 4.0,
                    "M0Type": "Separate",
                }
            )
        )  # shared M0, no run- entity
        with pytest.warns(PairingFallbackWarning):
            files, _ = build_inputs(case)
        m0s = [Path(f).name for f in files if f.endswith("_m0scan.json")]
        assert m0s == ["sub-X_m0scan.json", "sub-X_m0scan.json"]  # reused for both
        assert run_example(case)["nifti_slice_number"] == 18  # still runs

    def test_more_m0_than_asl_uses_latest(self, tmp_path: Path) -> None:
        """One ASL, two non-matching M0s: the latest (by time, then name) is used."""
        case = tmp_path / "case"
        perf = case / "sub-X" / "perf"
        _write_acq(perf, "sub-X")
        (perf / "sub-X_m0scan.json").unlink()  # drop the prefix-matching M0
        (perf / "aaa_m0scan.json").write_text(json.dumps({"M0Type": "Separate"}))
        (perf / "zzz_m0scan.json").write_text(json.dumps({"M0Type": "Separate"}))
        with pytest.warns(PairingFallbackWarning):
            files, _ = build_inputs(case)
        m0s = [Path(f).name for f in files if f.endswith("_m0scan.json")]
        # equal mtimes in tmp -> filename tie-break -> 'zzz' wins
        assert m0s == ["zzz_m0scan.json"]

    def test_differing_slice_counts_use_first_deterministically(
        self, tmp_path: Path
    ) -> None:
        """Acquisitions may differ in geometry (e.g. All_five); first is representative."""
        case = tmp_path / "case"
        _write_acq(case / "sub-A" / "perf", "sub-A", slices=18)
        _write_acq(case / "sub-B" / "perf", "sub-B", slices=32)
        _, nifti = build_inputs(case)
        assert nifti.endswith("sub-A_asl.nii.gz")  # sub-A sorts first
        assert run_example(case)["nifti_slice_number"] == 18

    def test_missing_nifti_fails_loudly(self, tmp_path: Path) -> None:
        """No ASL NIfTI anywhere is a hard, clear failure."""
        case = _make_example(tmp_path)
        for n in (case).rglob("*asl.nii.gz"):
            n.unlink()
        with pytest.raises(ExampleStructureError, match="no .*asl.nii"):
            build_inputs(case)

    def test_run_returns_21_key_contract(self, tmp_path: Path) -> None:
        """The tool returns exactly 21 keys for a valid example."""
        case = _make_example(tmp_path)
        report = run_example(case)
        assert len(report) == 21

    def test_roundtrip_matches_and_drift_fails(self, tmp_path: Path) -> None:
        """A freshly-written golden matches; a corrupted one is detected."""
        case = _make_example(tmp_path)
        write_expected(case, run_example(case))
        expected = load_expected(case)
        actual = normalize(run_example(case))
        assert actual == expected  # golden round-trips

        corrupted = dict(expected)
        corrupted["nifti_slice_number"] = expected["nifti_slice_number"] + 1
        write_expected(case, corrupted)
        assert normalize(run_example(case)) != load_expected(case)  # drift caught

    def test_slice_count_tracks_nifti(self, tmp_path: Path) -> None:
        """nifti_slice_number reflects the synthesized slice axis."""
        case = _make_example(tmp_path, slices=27)
        assert run_example(case)["nifti_slice_number"] == 27
