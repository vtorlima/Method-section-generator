import pytest
from pyaslreport.modalities.asl.processor import ASLProcessor
from pyaslreport.modalities.asl.constants import DURATION_OF_EACH_RFBLOCK
from pyaslreport.utils.unit_conversion_utils import UnitConverterUtils
from pyaslreport.utils.math_utils import MathUtils


def make_processor():
    return ASLProcessor({"files": [], "nifti_file": None, "dcm_files": []})
