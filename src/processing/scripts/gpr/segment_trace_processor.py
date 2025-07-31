"""
GPR Segment Trace Processor - Processes a single trace from a GPR data segment.

This script allows for detailed analysis of a single GPR trace by applying a
user-selected sequence of processing steps. It loads a pre-processed .npz data file,
extracts a specific trace, processes it, and generates diagnostic plots.

Workflow:
1.  Load a ProjectGPRdata object from a .npz file.
2.  Select a specific segment and trace number.
3.  Choose a series of processing steps from the UI.
4.  Process the single trace using the selected steps.
5.  Generate and save two plots:
    - A profile of the entire segment for context.
    - A power spectrum comparison of the raw vs. processed trace.
6.  Outputs are saved as .mplplot files in project/processed/gpr/ following framework structure.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
import datetime
from typing import Dict, Any, Optional, Callable
from loguru import logger
from rpy2.rinterface_lib.embedded import RRuntimeError

from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError, ScriptMetadata

try:
    from src.processing.scripts.gpr.utils.LoadGPRdata import ProjectGPRdata
    from src.processing.scripts.gpr.utils.GPRdataSegmentProcessor import GPRdataSegmentProcessor
    from src.processing.scripts.gpr.utils.gpr_configs import ProcessingConfig
    GPR_UTILS_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import GPR utilities for SegmentTraceProcessor: {e}. The script will be unavailable.", exc_info=True)
    ProjectGPRdata = ProcessingConfig = GPRdataSegmentProcessor = object
    GPR_UTILS_AVAILABLE = False

# Plotting helper function from user notes
def plotting_profile_geo(input_data, dist_vector, dt, resample, aspect, title, perc):
    m, n = input_data.shape
    num_ticks = 10
    tick_indices = np.linspace(0, n - 1, num_ticks, dtype=int)
    d_vec = np.array(dist_vector)
    tick_values = np.round(d_vec[tick_indices]).astype(int)
    
    fig, ax = plt.subplots(figsize=(25, 7))
    im = ax.matshow(input_data, extent=[0, len(input_data[0]), len(input_data)*dt/resample, 0], 
                    aspect=aspect, cmap='seismic', 
                    vmin=np.amin(input_data)/perc, vmax=np.amax(input_data)/perc)
    ax.set_xticks(tick_indices)
    ax.set_xticklabels(tick_values)
    fig.colorbar(im, ax=ax)
    ax.set_ylabel('Time (ns)')
    ax.set_xlabel('Distance (m)') 
    ax.set_title(title)
    return fig

class SegmentTraceProcessor(ScriptInterface):
    """Processes a single trace from a GPR data segment and generates diagnostic plots."""

    @property
    def name(self) -> str:
        return "GPR Segment Trace Processor"

    @property
    def description(self) -> str:
        return ("Processes a single GPR trace from a selected segment within an .npz file. "
                "Allows custom processing chains and outputs diagnostic plots.")

    def get_metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            description="Processes a single GPR trace from a selected segment. Allows custom processing chains and outputs diagnostic plots for detailed analysis.",
            flags=["analysis", "qc", "visualization"],
            typical_use_case="Detailed analysis or debugging of a specific GPR trace. Useful for quality control and parameter tuning.",
            field_compatible=True,
            estimated_runtime="< 1 minute"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Defines the parameters for the processor."""
        # A default list of available methods.
        available_methods = [
            "resample_profile",
            "dc_shift_correction_rgpr",
            "t0_correction",
            "t0_correction_rgpr",
            "lowpass_filter",
            "hipass_filter",
            "bandpass_filter",
            "bandpass_filter_rgpr",
            "background_removal",
            "background_removal_rgpr",
            "horizontal_filter",
            "horizontal_filter_rgpr",
            "dewow",
            "amplitude_correction",
            "amplitude_correction_rgpr",
            "power_gain_rgpr",
            "mixed_phase_deconvolution_rgpr",
            "kirchoff_migration_rgpr"
        ]
        
        params = {
            'target_selection': {
                'segment_number': {
                    'value': 1,
                    'type': 'int',
                    'min': 1,
                    'description': 'The segment number to analyze.'
                },
                'trace_number': {
                    'value': 1,
                    'type': 'int',
                    'min': 1,
                    'description': 'The trace number within the segment to process.'
                }
            },
            'processing_steps': {}
        }

        # Dynamically create checkboxes for each processing step
        for method in available_methods:
            params['processing_steps'][method] = {
                'value': False,
                'type': 'bool',
                'description': f'Enable {method.replace("_", " ").title()}'
            }
        return params

    def validate_data(self, data: pd.DataFrame) -> bool:
        # GPR trace processor works with .npz files via input_file_path
        # DataFrame can be empty since we use the file path directly
        return True

    def _create_output_directory(self, input_file_path: Optional[str], working_directory: Optional[str] = None) -> Path:
        """Create output directory in project/processed/gpr/ following framework structure"""    
        if input_file_path:
            base_filename = Path(input_file_path).stem
            project_dir = Path(working_directory) if working_directory else Path(input_file_path).parent
            
            # Navigate up to find project root if we're in a subdirectory
            while project_dir != project_dir.parent:
                if (project_dir / "processed").exists() or len(list(project_dir.glob("*.uxo"))) > 0:
                    break
                project_dir = project_dir.parent
            
            # Create project/processed/gpr/filename_analysis_timestamp structure
            output_dir = project_dir / "processed" / "gpr" / f"{base_filename}_analysis_{datetime.datetime.now().strftime('%Y%m%d_%H%M%S')}"
        else:
            # Fallback: use temp directory
            import tempfile
            temp_dir = tempfile.mkdtemp(prefix="gpr_processing_")
            output_dir = Path(temp_dir)
        
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def execute(self, data: pd.DataFrame, params: Dict[str, Any],
                progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        
        if not GPR_UTILS_AVAILABLE:
            raise ProcessingError("GPR processing utilities are not available. Please check dependencies.")

        result = ProcessingResult(success=False, processing_script=self.name)
        
        try:
            # --- 1. Extract Parameters ---
            if progress_callback: progress_callback(5, "Loading parameters...")
            # The input file is now passed directly from the framework
            npz_file = input_file_path
            
            if not npz_file or not Path(npz_file).exists():
                raise FileNotFoundError("A valid input .npz file is required but was not provided by the framework.")

            segment_num = params.get('target_selection', {}).get('segment_number', {}).get('value')
            trace_num = params.get('target_selection', {}).get('trace_number', {}).get('value')
            
            # Create output directory following framework structure
            output_dir = self._create_output_directory(input_file_path, self.get_project_working_directory())
            
            # --- 2. Load Data ---
            if progress_callback: progress_callback(15, f"Loading data from {Path(npz_file).name}...")
            # The script should always load from the file path for this workflow.
            proj_data = ProjectGPRdata.from_filepath(npz_file)
            
            segment_data, dist_vec, _ = proj_data.getDataForSegment(segment_num)
            
            if trace_num > segment_data.shape[1]:
                raise ValueError(f"Trace number {trace_num} is out of bounds for segment {segment_num} "
                                 f"which has {segment_data.shape[1]} traces.")
            
            target_trace = segment_data[:, trace_num - 1]

            # --- 3. Configure Processing ---
            if progress_callback: progress_callback(30, "Configuring processing pipeline...")
            processing_steps_params = params.get('processing_steps', {})
            processing_order = [step for step, config in processing_steps_params.items() if config.get('value')]
            
            if not processing_order:
                raise ValueError("No processing steps selected.")

            process_config = ProcessingConfig()
            process_config.processing_order = processing_order
            logger.info(f"Processing order set to: {processing_order}")

            # --- 4. Process Trace ---
            if progress_callback: progress_callback(50, f"Processing trace {trace_num}...")
            
            # Validate trace data before processing
            logger.info(f"Trace data shape: {target_trace.shape}")
            logger.info(f"Trace data range: [{np.min(target_trace):.6f}, {np.max(target_trace):.6f}]")
            logger.info(f"Processing steps to apply: {processing_order}")
            
            # Check for NaN or infinite values
            if np.any(np.isnan(target_trace)):
                logger.warning("Target trace contains NaN values")
            if np.any(np.isinf(target_trace)):
                logger.warning("Target trace contains infinite values")
                
            try:
                segm_processor = GPRdataSegmentProcessor()
                logger.info("GPR processor initialized successfully")
            except Exception as init_err:
                raise ProcessingError(f"Failed to initialize GPR processor: {init_err}")
            
            # The input 'data' DataFrame is not used, as we load directly from the NPZ file.
            # The original trace data is in 'target_trace'
            processed_trace, intermediate_steps = segm_processor.process_data(target_trace, process_config)
            
            # --- 5. Generate and Save Plots ---
            if progress_callback: progress_callback(70, "Generating plots...")
            dt = proj_data.config.get('dt', 1.0) # Get time step, default to 1.0

            # Plot 1: Full Segment Profile
            profile_fig = plotting_profile_geo(segment_data, dist_vec, dt, 1, 2, f'Segment {segment_num} Profile', 10)
            profile_plot_path = output_dir / f"{Path(npz_file).stem}_seg{segment_num}_profile.mplplot"
            with open(profile_plot_path, 'wb') as f:
                pickle.dump(profile_fig, f)
            result.add_output_file(str(profile_plot_path), 'matplotlib_plot', f'Profile plot for segment {segment_num}')
            plt.close(profile_fig)
            
            # Plot 2: Power Spectrum
            spectrum_fig, ax = plt.subplots(figsize=(10, 6))
            
            # Raw trace spectrum
            n_samples_raw = target_trace.shape[0]
            frequencies_raw = np.fft.fftfreq(n_samples_raw, dt * 1e-9)
            raw_ps = np.abs(np.fft.fft(target_trace))**2
            ax.plot(frequencies_raw, raw_ps / np.max(raw_ps), label='Raw Trace', linewidth=2)

            # Processed trace spectrum
            n_samples_proc = processed_trace.shape[0]
            frequencies_proc = np.fft.fftfreq(n_samples_proc, dt * 1e-9)
            proc_ps = np.abs(np.fft.fft(processed_trace))**2
            ax.plot(frequencies_proc, proc_ps / np.max(proc_ps), label='Processed Trace')

            ax.set_title(f'Power Spectrum Comparison (Trace {trace_num})')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Normalized Power')
            ax.grid(True)
            ax.set_xlim(0, 2e9)
            ax.legend()
            
            spectrum_plot_path = output_dir / f"{Path(npz_file).stem}_trace{trace_num}_spectrum.mplplot"
            with open(spectrum_plot_path, 'wb') as f:
                pickle.dump(spectrum_fig, f)
            result.add_output_file(str(spectrum_plot_path), 'matplotlib_plot', f'Power spectrum plot for trace {trace_num}')
            plt.close(spectrum_fig)
            
            # --- 6. Finalize ---
            if progress_callback: progress_callback(100, "Processing complete.")
            result.success = True
            result.message = (f"Successfully processed trace {trace_num} from segment {segment_num}. "
                              f"Plots saved to {output_dir.name} directory.")
            
            # Add comprehensive metadata
            result.metadata = {
                'processor': 'gpr',
                'script_name': self.name,
                'segment_number': segment_num,
                'trace_number': trace_num,
                'processing_steps': processing_order,
                'processing_steps_count': len(processing_order),
                'output_files_count': len(result.output_files),
                'segment_data_shape': list(segment_data.shape),
                'trace_data_length': len(raw_trace),
                'processed_trace_length': len(processed_trace),
                'processing_config': {
                    'dt': process_config.dt,
                    'visualization': {
                        'plt_aspect': plt_aspect,
                        'plt_perc': plt_perc
                    }
                },
                'parameters': params  # Include all parameters used
            }
            
            logger.success(result.message)

        except RRuntimeError as r_err:
            error_message = f"An R error occurred during trace processing: {r_err}"
            logger.error(error_message, exc_info=True)
            result.error_message = error_message
        except Exception as e:
            logger.error(f"Trace processing failed: {e}", exc_info=True)
            result.error_message = f"Trace processing failed: {str(e)}"

        return result

# Export for framework discovery
SCRIPT_CLASS = SegmentTraceProcessor 