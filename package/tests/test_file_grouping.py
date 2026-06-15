"""Tests for _group_files and FileReader behavior."""

import json
from pathlib import Path
from typing import Callable

import pytest

from pyaslreport.modalities.asl.processor import ASLProcessor, ProcessingContext

# ---------- _group_files: NIfTI mode (exact suffix matching) ----------


class TestGroupFilesNiftiMode:
    def test_groups_asl_with_tsv_and_m0(
        self, make_processor: Callable[..., ASLProcessor], tmp_path: Path
    ) -> None:
        """A canonical BIDS triple groups together correctly."""
        asl_json = tmp_path / "sub-01_asl.json"
        asl_json.write_text(json.dumps({"M0Type": "Separate"}))
        tsv = tmp_path / "sub-01_aslcontext.tsv"
        tsv.write_text("volume_type\ncontrol\nlabel\n")
        m0_json = tmp_path / "sub-01_m0scan.json"
        m0_json.write_text(json.dumps({"EchoTime": 0.012}))

        proc = make_processor(files=[str(asl_json), str(tsv), str(m0_json)])
        groups = proc._group_files("nifti")

        assert len(groups) == 1
        g = groups[0]
        assert g["asl_json"][0] == "sub-01_asl.json"
        assert g["tsv"][0] == "sub-01_aslcontext.tsv"
        assert g["m0_json"][0] == "sub-01_m0scan.json"

    def test_two_sessions_produce_two_groups(
        self, make_processor: Callable[..., ASLProcessor], tmp_path: Path
    ) -> None:
        """Two BIDS sessions in the same directory produce two groups."""
        # Build the list explicitly. Do NOT scan tmp_path with iterdir(): the
        # make_processor -> minimal_nifti_path fixture also writes asl.nii.gz
        # into tmp_path, and _group_files rejects unknown extensions.
        files: list[str] = []
        for i in [1, 2]:
            asl_json = tmp_path / f"sub-0{i}_asl.json"
            asl_json.write_text(json.dumps({"M0Type": "Separate"}))
            tsv = tmp_path / f"sub-0{i}_aslcontext.tsv"
            tsv.write_text("volume_type\ncontrol\nlabel\n")
            files.extend([str(asl_json), str(tsv)])
        proc = make_processor(files=files)
        groups = proc._group_files("nifti")
        assert len(groups) == 2

    def test_unsupported_extension_raises(
        self, make_processor: Callable[..., ASLProcessor], tmp_path: Path
    ) -> None:
        """An unsupported extension raises ValueError during grouping."""
        bad = tmp_path / "weird.xml"
        bad.write_text("<x/>")
        proc = make_processor(files=[str(bad)])
        with pytest.raises(ValueError, match="Unsupported file format"):
            proc._group_files("nifti")


# ---------- _group_files: DICOM mode (substring matching for m0) ----------


class TestGroupFilesDicomMode:
    def test_dicom_mode_uses_substring_for_m0(
        self, make_processor: Callable[..., ASLProcessor], tmp_path: Path
    ) -> None:
        """In DICOM mode, any filename containing 'm0' counts as M0."""
        asl_json = tmp_path / "scan_dump.json"
        asl_json.write_text(json.dumps({"M0Type": "Separate"}))
        m0_json = tmp_path / "scan_m0_dump.json"
        m0_json.write_text(json.dumps({"EchoTime": 0.012}))

        proc = make_processor(files=[str(asl_json), str(m0_json)])
        groups = proc._group_files("dicom")
        assert len(groups) == 1
        assert groups[0]["asl_json"][0] == "scan_dump.json"
        assert groups[0]["m0_json"][0] == "scan_m0_dump.json"


# ---------- _validate_tsv_data: missing TSV behavior ----------


class TestMissingTSV:
    def test_missing_tsv_in_nifti_mode_errors(
        self,
        make_processor: Callable[..., ASLProcessor],
        make_context: Callable[..., ProcessingContext],
    ) -> None:
        """Missing aslcontext.tsv in NIfTI mode produces a missing-file error."""
        proc = make_processor()
        ctx = make_context()
        group = {
            "asl_json": ("asl.json", {"M0Type": "Absent"}),
            "m0_json": None,
            "tsv": None,
        }
        proc._validate_tsv_data(group, ctx, "asl.json", group["asl_json"][1], "nifti")
        assert any("aslcontext.tsv" in e and "missing" in e for e in ctx.errors)

    def test_missing_tsv_in_dicom_mode_falls_through_to_dicom_repetitions(
        self,
        make_processor: Callable[..., ASLProcessor],
        make_context: Callable[..., ProcessingContext],
    ) -> None:
        """Missing TSV in DICOM mode delegates to _analyze_dicom_repetitions."""
        proc = make_processor()
        ctx = make_context()
        asl_data = {"lRepetitions": 10}
        group = {"asl_json": ("asl.json", asl_data), "m0_json": None, "tsv": None}
        proc._validate_tsv_data(group, ctx, "asl.json", asl_data, "dicom")
        # _analyze_dicom_repetitions sets total_acquired_pairs from lRepetitions/2
        assert ctx.total_acquired_pairs == 5
        # No TSV-missing error in DICOM mode
        assert not any("aslcontext.tsv" in e for e in ctx.errors)


# ---------- FileReader: TSV header enforcement ----------


class TestFileReaderTSVHeader:
    def test_valid_header_returns_data(self, tmp_path: Path) -> None:
        """A 'volume_type' header with rows returns the rows as a list."""
        from pyaslreport.io.readers.file_reader import FileReader

        f = tmp_path / "valid.tsv"
        f.write_text("volume_type\ncontrol\nlabel\n")
        result = FileReader.read(str(f))
        assert result == ["control", "label"]

    def test_invalid_header_raises(self, tmp_path: Path) -> None:
        """A header that isn't exactly 'volume_type' raises RuntimeError.

        NOTE: FileReader.read re-wraps the inner error as
        'Error reading file: Invalid TSV header, ...'. The substring match below
        still matches; do NOT anchor this regex with '^'.
        """
        from pyaslreport.io.readers.file_reader import FileReader

        f = tmp_path / "bad.tsv"
        f.write_text("volume_types\ncontrol\nlabel\n")  # plural, wrong
        with pytest.raises(RuntimeError, match="Invalid TSV header"):
            FileReader.read(str(f))

    def test_empty_file_returns_none(self, tmp_path: Path) -> None:
        """A truly empty TSV returns None rather than raising."""
        from pyaslreport.io.readers.file_reader import FileReader

        f = tmp_path / "empty.tsv"
        f.write_text("")
        assert FileReader.read(str(f)) is None
