import os
from pyaslreport.core.config import config

from pyaslreport.io.readers.file_reader import FileReader

class ASLUtils:

    @staticmethod
    def determine_pld_type(session):
        # Check if any of the specified keys contain arrays with different unique values
        for key in ['PostLabelingDelay', 'EchoTime', 'LabelingDuration']:
            if key in session and isinstance(session[key], list):
                unique_values = set(session[key])
                if len(unique_values) > 1:
                    return "multi-PLD"
        return "single-PLD"

    @staticmethod
    def extract_params(data):
        return {
            "EchoTime": data.get("EchoTime"),
            "FlipAngle": data.get("FlipAngle"),
            "MagneticFieldStrength": data.get("MagneticFieldStrength"),
            "MRAcquisitionType": data.get("MRAcquisitionType"),
            "PulseSequenceType": data.get("PulseSequenceType")
        }

    @staticmethod
    def compare_params(params_asl, params_m0, asl_filename, m0_filename):
        consistency_schema = config['schemas']['consistency_schema']

        errors = []
        warnings = []
        for param, asl_value in params_asl.items():
            m0_value = params_m0.get(param)
            schema = consistency_schema.get(param)

            if not schema:
                continue

            validation_type = schema.get('validation_type')
            warning_variation = schema.get('warning_variation', 1e-5)
            error_variation = schema.get('error_variation', 1e-4)

            if validation_type == "string":
                if asl_value != m0_value:
                    errors.append(
                        f"Discrepancy in '{param}' for ASL file '{asl_filename}' and M0 file '{m0_filename}': "
                        f"ASL value = {asl_value}, M0 value = {m0_value}")
            elif validation_type == "floatOrArray":
                if isinstance(asl_value, (int, float)) and isinstance(m0_value, (int, float)):
                    difference = abs(asl_value - m0_value)
                    difference_formatted = f"{difference:.2f}"
                    if difference > error_variation:
                        errors.append(
                            f"ERROR: Discrepancy in '{param}' for ASL file '{asl_filename}' and M0 file '{m0_filename}': "
                            f"ASL value = {asl_value}, M0 value = {m0_value}, difference = {difference_formatted}, exceeds error threshold {error_variation}")
                    elif difference > warning_variation:
                        warnings.append(
                            f"WARNING: Discrepancy in '{param}' for ASL file '{asl_filename}' and M0 file '{m0_filename}': "
                            f"ASL value = {asl_value}, M0 value = {m0_value}, difference = {difference_formatted}, exceeds warning threshold {warning_variation}")
        return errors, warnings

    @staticmethod
    def analyze_volume_types(volume_types):
        first_non_m0type = next((vt for vt in volume_types if vt in {'control', 'label', 'deltam'}), None)
        pattern = "pattern error"
        control_label_pairs = 0
        label_control_pairs = 0

        if first_non_m0type == 'control':
            pattern = 'control-label'
        elif first_non_m0type == 'label':
            pattern = 'label-control'
        elif first_non_m0type == 'deltam':
            pattern = 'deltam'
            deltam_count = volume_types.count('deltam')
            return pattern, deltam_count
        i = 0
        while i < len(volume_types):
            if volume_types[i] == 'control' and i + 1 < len(volume_types) and volume_types[
                i + 1] == 'label':
                control_label_pairs += 1
                i += 2
            elif volume_types[i] == 'label' and i + 1 < len(volume_types) and volume_types[
                i + 1] == 'control':
                label_control_pairs += 1
                i += 2
            else:
                i += 1
        if pattern == 'control-label':
            return pattern, control_label_pairs
        else:
            return pattern, label_control_pairs

    @staticmethod
    def ensure_keys_and_append(dictionary, key, value):
        if value and key not in dictionary:
            dictionary[key] = []
            dictionary[key].append(value)

    @staticmethod
    def extract_concise_error(issue_dict):
        report = []

        for field, issues in issue_dict.items():
            for issue in issues:
                if isinstance(issue, dict):
                    for sub_issue, details in issue.items():
                        if isinstance(details, list):
                            details_str = ', '.join(map(str, details))
                            report.append(f'{sub_issue} for "{field}": {details_str}')
                        else:
                            report.append(f'{sub_issue} for "{field}": {details}')

        return '\n'.join(report)

    @staticmethod
    def condense_and_reformat_discrepancies(error_list):

        if not error_list:
            return [], []
        
        condensed_errors = {}
        param_names = []

        for error in error_list:
            if "Discrepancy in '" in error:
                # Extract the key part of the error message
                start_idx = error.index("Discrepancy in '")
                end_idx = error.index("'", start_idx + len("Discrepancy in '"))
                param_name = error[start_idx + len("Discrepancy in '"):end_idx]

                # Reformat the message in the desired format
                reformatted_error = f"{param_name} (M0): Discrepancy between ASL JSON and M0 JSON"

                # If the parameter is already in the dictionary, skip adding it again
                if param_name not in condensed_errors:
                    condensed_errors[param_name] = reformatted_error
                    # Collect the param_name
                    param_names.append(param_name)
            else:
                # If the message doesn't contain "Discrepancy", add it as is
                condensed_errors[error] = error

        return list(condensed_errors.values()), param_names

    @staticmethod
    def determine_m0_tr_and_report(m0_prep_times_collection, all_absent, bs_all_off, discrepancies,
                                   m0_type, inconsistent_params):
        M0_TR = None
        if m0_type == "Estimate":
            return M0_TR, "A single M0 scaling value is provided for CBF quantification"
        if m0_prep_times_collection and all(m0_prep_times_collection):
            if all(abs(x - m0_prep_times_collection[0]) < 1e-5 for x in m0_prep_times_collection):
                M0_TR = m0_prep_times_collection[0]
            else:
                discrepancies.append("Different `RepetitionTimePreparation` parameters for M0")

        if all_absent and bs_all_off:
            report_line_on_M0 = "No m0-scan was acquired, a control image without background suppression was used for M0 estimation."
        elif all_absent and not bs_all_off:
            report_line_on_M0 = "No m0-scan was acquired, but there doesn't always exist a control image without background suppression."
        else:
            if not discrepancies:
                report_line_on_M0 = "M0 was acquired with the same readout and without background suppression."
            else:
                inconsistent_params_str = ", ".join(inconsistent_params)
                report_line_on_M0 = f"There is inconsistency in {inconsistent_params_str} between M0 and ASL scans."

        return M0_TR, report_line_on_M0