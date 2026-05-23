import pytest
from unittest.mock import patch, MagicMock
from pyaslreport.main import get_bids_metadata

# filepath: /home/ibrahim/MyPc/Projects/GSoC/ASL-Parameter-Generator/package/src/pyaslreport/test_main.py

def test_get_bids_metadata_success():
    data = {"modality": "asl", "dicom_dir": "/fake/dir"}
    fake_header = MagicMock()
    fake_sequence = MagicMock()
    fake_sequence.extract_bids_metadata.return_value = ("meta", "context")
    with patch("pyaslreport.main.get_dicom_header", return_value=fake_header), \
         patch("pyaslreport.main.get_sequence", return_value=fake_sequence):
        result = get_bids_metadata(data)
        assert result == ("meta", "context")
        fake_sequence.extract_bids_metadata.assert_called_once()

def test_get_bids_metadata_no_dicom_dir():
    data = {"modality": "asl"}
    with pytest.raises(TypeError):
        get_bids_metadata(data)

def test_get_bids_metadata_no_sequence():
    data = {"modality": "asl", "dicom_dir": "/fake/dir"}
    fake_header = MagicMock()
    with patch("pyaslreport.main.get_dicom_header", return_value=fake_header), \
         patch("pyaslreport.main.get_sequence", return_value=None):
        with pytest.raises(ValueError) as exc:
            get_bids_metadata(data)
        assert "No matching sequence found" in str(exc.value)

def test_get_bids_metadata_invalid_modality():
    data = {"modality": None, "dicom_dir": "/fake/dir"}
    fake_header = MagicMock()
    with patch("pyaslreport.main.get_dicom_header", return_value=fake_header), \
         patch("pyaslreport.main.get_sequence", return_value=None):
        with pytest.raises(ValueError):
            get_bids_metadata(data)