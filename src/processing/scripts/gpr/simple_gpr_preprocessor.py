"""
Simple GPR Pre-processor - Segments raw survey data and saves as an NPZ file.

This script implements the basic pre-processing workflow for GPR data as outlined
in the user's initial notes. It is designed to be a straightforward tool for 
converting raw survey files into a segmented, project-ready format.

Workflow:
1. Load raw survey data from a specified directory using a case date.
2. Segment the data based on survey waypoints.
3. Apply simple processing: interpolation and time-cutting.
4. Save the resulting segmented data object to a single .npz file.
"""

from pathlib import Path
from typing import Dict, Any, Optional, Callable
import pandas as pd
from loguru import logger

from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError, ScriptMetadata

try:
    # Attempt to import the necessary GPR data loading utilities.
    from src.processing.scripts.gpr.utils.LoadGPRdata import LoadRawSurveyData
    GPR_LOADER_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import GPR utilities for SimpleGPRPreprocessor: {e}. The script will be unavailable.", exc_info=True)
    # Create a fallback class if the import fails to ensure the UI can still load the script list.
    class LoadRawSurveyData:
        def __init__(self, *args, **kwargs):
            raise ImportError("LoadGPRdata utility not found. Please check GPR script dependencies.")
    GPR_LOADER_AVAILABLE = False


class SimpleGPRPreprocessor(ScriptInterface):
    """
    A simple GPR pre-processor that segments raw survey data and saves it as a single .npz file.
    This is a basic pre-processing step to prepare raw survey data for further analysis.
    """

    @property
    def name(self) -> str:
        return "Simple GPR Pre-processor"

    @property
    def description(self) -> str:
        return ("Loads raw GPR survey data (log, csv, sgy), creates interpolated and time-cut "
                "segments, and saves the result as a single project .npz file.")

    def get_metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            description="Loads raw GPR survey data, segments it by survey waypoints, and saves the result as a single project .npz file for further processing.",
            flags=["preprocessing", "field-use", "export"],
            typical_use_case="Initial step for converting raw GPR survey files into a segmented, project-ready format. Use this before running analysis scripts.",
            field_compatible=True,
            estimated_runtime="1-3 minutes"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Defines the parameters for the Simple GPR Pre-processor."""
        return {
            'data_input': {
                'survey_directory': {
                    'value': '',
                    'type': 'directory',
                    'description': 'Directory containing the raw survey files (e.g., project_Juvfonne/raw_data)'
                }
            },
            'segmentation_settings': {
                'trace_spacing': {
                    'value': 0.05,
                    'type': 'float',
                    'min': 0.01,
                    'max': 1.0,
                    'description': 'Trace spacing for interpolation in meters (default: 0.05m).'
                },
                'time_cut': {
                    'value': 70,
                    'type': 'int',
                    'min': -1,
                    'max': 200,
                    'description': 'Time cut for data profile in nanoseconds (default: 70ns, -1 for no cut).'
                }
            }
        }

    def validate_data(self, data: pd.DataFrame) -> bool:
        """This script does not use an input DataFrame, so validation is always true."""
        return True

    def execute(self, data: pd.DataFrame, params: Dict[str, Any],
                progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        
        if not GPR_LOADER_AVAILABLE:
            raise ProcessingError("GPR processing dependencies (LoadGPRdata) are not available.")

        result = ProcessingResult(success=False, script_id=self.name)

        try:
            logger.info("Starting Simple GPR Pre-processing...")
            if progress_callback:
                progress_callback(0, "Initializing...")

            # --- 1. Extract and Validate Parameters ---
            logger.info("Extracting and validating parameters...")
            input_params = params.get('data_input', {})
            survey_directory = input_params.get('survey_directory', {}).get('value')

            # Hardcoded and derived parameters
            case_date = ''  # Will be auto-detected
            seg_params = params.get('segmentation_settings', {})
            trace_spacing = seg_params.get('trace_spacing', {}).get('value', 0.05)
            time_cut = seg_params.get('time_cut', {}).get('value', 70)
            output_suffix = '_raw'

            if not survey_directory:
                raise ValueError("Survey directory must be specified.")

            survey_dir_path = Path(survey_directory)
            if not survey_dir_path.is_dir():
                raise FileNotFoundError(f"Survey directory not found: {survey_directory}")

            # --- 2. Determine and Create Output Directory ---
            project_root_path = survey_dir_path.parent
            output_dir_path = project_root_path / 'processed'
            output_dir_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"Survey Directory: {survey_dir_path}")
            logger.info(f"Output Directory (auto-derived): {output_dir_path}")

            # --- 3. Auto-detect case_date ---
            logger.info("Auto-detecting case date...")
            if progress_callback: progress_callback(10, "Auto-detecting case date...")
            
            log_files = list(survey_dir_path.glob('*-system.log'))
            if log_files:
                case_date = log_files[0].stem.replace('-system', '')
                logger.info(f"Auto-detected case date: {case_date}")
            else:
                raise FileNotFoundError("Could not auto-detect case date from *-system.log files in survey directory.")

            if progress_callback: progress_callback(20, f"Processing survey: {case_date}")

            # --- 4. Initialize LoadRawSurveyData ---
            survey_path_str = str(survey_dir_path)
            if not survey_path_str.endswith('/'):
                survey_path_str += '/'

            logger.info(f"Loading raw survey data for case '{case_date}' from '{survey_path_str}'")
            raw_survey_obj = LoadRawSurveyData(survey_path_str, case_date)
            
            # --- 5. Create Segments with Simple Processing ---
            if progress_callback: progress_callback(40, "Creating segments...")
            logger.info(f"Creating segments with trace_spacing={trace_spacing}m and time_cut={time_cut}ns.")
            raw_survey_obj.create_segments(trace_spacing=trace_spacing, timeCut=time_cut)

            # --- 6. Retrieve Project Data ---
            if progress_callback: progress_callback(60, "Retrieving project data...")
            gprProjectData_obj = raw_survey_obj.get_project_data()
            logger.info(f"Retrieved project data with {gprProjectData_obj.numberOfSegments} segments.")

            # --- 7. Save Processed Data to NPZ File ---
            if progress_callback: progress_callback(80, "Saving processed data...")
            output_filename_base = f"{case_date}{output_suffix}"
            output_filepath = output_dir_path / output_filename_base
            
            logger.info(f"Saving data to {output_filepath}.npz")
            gprProjectData_obj.to_filepath(str(output_filepath))

            # --- 8. Finalize Result Object ---
            output_npz_file = str(output_filepath) + '.npz'
            result.success = True
            result.add_output_file(
                file_path=output_npz_file,
                file_type='gpr_project_data',
                description=f'Segmented GPR data for survey {case_date}'
            )
            result.message = f"Successfully processed and saved GPR data to {output_npz_file}"
            logger.success(f"Processing complete. Output at: {output_npz_file}")
            
            if progress_callback:
                progress_callback(100, "Processing complete.")

        except (ValueError, FileNotFoundError) as e:
            logger.error(f"Configuration error: {e}")
            result.error_message = f"Configuration error: {str(e)}"
        except Exception as e:
            logger.error(f"An unexpected error occurred during GPR pre-processing: {e}", exc_info=True)
            result.error_message = f"An unexpected error occurred: {str(e)}"
        
        return result

# IMPORTANT: Export the class for discovery by the framework
SCRIPT_CLASS = SimpleGPRPreprocessor 