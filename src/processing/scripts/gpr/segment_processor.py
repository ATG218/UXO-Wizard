"""
GPR Segment Processor - Processes all traces from a GPR data segment.

This script processes an entire GPR segment using a user-selected sequence of 
processing steps. It loads a pre-processed .npz data file, extracts a specific 
segment, processes all traces, and generates comprehensive diagnostic outputs.

Workflow:
1.  Load a ProjectGPRdata object from a .npz file.
2.  Select a specific segment number.
3.  Choose a series of processing steps and configure their parameters.
4.  Process all traces in the segment using the selected steps.
5.  Generate and save multiple plots:
    - Raw segment profile
    - Each intermediate processing step
    - Final processed segment profile
    - Power spectrum comparison for a selected trace
6.  Outputs are saved as .mplplot files in the same directory as the input .npz file.
"""

from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import pickle
from typing import Dict, Any, Optional, Callable
from loguru import logger
from rpy2.rinterface_lib.embedded import RRuntimeError

from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError, ScriptMetadata

try:
    from src.processing.scripts.gpr.utils.LoadGPRdata import ProjectGPRdata
    from src.processing.scripts.gpr.utils.GPRdataSegmentProcessor import GPRdataSegmentProcessor
    from src.processing.scripts.gpr.utils.gpr_configs import ProcessingConfig
    GPR_UTILS_AVAILABLE = True
    print("GPR_UTILS_AVAILABLE: True")
except ImportError as e:
    logger.warning(f"Could not import GPR utilities for SegmentProcessor: {e}. The script will be unavailable.", exc_info=True)
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

class SegmentProcessor(ScriptInterface):
    """Processes all traces from a GPR data segment with configurable processing steps."""

    @property
    def name(self) -> str:
        return "GPR Segment Processor"

    @property
    def description(self) -> str:
        return ("Processes all traces from a selected GPR segment within an .npz file. "
                "Allows custom processing chains with configurable parameters and outputs comprehensive diagnostic plots.")

    def get_metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            description="Processes all traces from a selected GPR segment. Allows custom processing chains with configurable parameters and outputs diagnostic plots for each processing step.",
            flags=["processing", "analysis", "visualization", "batch"],
            typical_use_case="Complete processing of GPR segment data with full parameter control and step-by-step visualization.",
            field_compatible=True,
            estimated_runtime="1-5 minutes"
        )

    def get_parameters(self) -> Dict[str, Any]:
        """Defines the parameters for the processor."""
        # Available processing methods
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
                    'value': 2,
                    'type': 'int',
                    'min': 1,
                    'description': 'The segment number to process.'
                },
                'trace_for_spectrum': {
                    'value': 400,
                    'type': 'int',
                    'min': 1,
                    'description': 'Trace number for power spectrum comparison.'
                }
            },
            'processing_steps': {},
            'dc_shift_correction': {
                'DCsample_start': {
                    'value': 250,
                    'type': 'int',
                    'min': 1,
                    'max': 999,
                    'description': 'Start sample for DC shift estimation.'
                },
                'DCsample_end': {
                    'value': 450,
                    'type': 'int',
                    'min': 1,
                    'max': 999,
                    'description': 'End sample for DC shift estimation.'
                }
            },
            't0_correction': {
                'time1': {
                    'value': 5.0,
                    'type': 'float',
                    'min': 0.0,
                    'description': 'Start time for zoom on 1st arrival (ns).'
                },
                'time2': {
                    'value': 17.0,
                    'type': 'float',
                    'min': 0.0,
                    'description': 'End time for zoom on 1st arrival (ns).'
                },
                'threshold_t0': {
                    'value': 0.06,
                    'type': 'float',
                    'min': 0.001,
                    'max': 1.0,
                    'description': 'Threshold value to define t0.'
                }
            },
            'bandpass_filter': {
                'f1_bandpass': {
                    'value': 0.15,
                    'type': 'float',
                    'min': 0.01,
                    'max': 2.0,
                    'description': 'Lowcut frequency in GHz.'
                },
                'f2_bandpass': {
                    'value': 0.55,
                    'type': 'float',
                    'min': 0.01,
                    'max': 2.0,
                    'description': 'Highcut frequency in GHz.'
                },
                'poly_order_bandpass': {
                    'value': 4,
                    'type': 'int',
                    'min': 1,
                    'max': 10,
                    'description': 'Polynomial order for bandpass filter.'
                }
            },
            'amplitude_correction': {
                'alpha_power_gain': {
                    'value': 1.0,
                    'type': 'float',
                    'min': 0.1,
                    'max': 5.0,
                    'description': 'Power order for geometrical spreading correction.'
                },
                'te_power_gain': {
                    'value': 70.0,
                    'type': 'float',
                    'min': 1.0,
                    'max': 200.0,
                    'description': 'Gain applied until te, in ns.'
                },
                'tcst_power_gain': {
                    'value': 3.0,
                    'type': 'float',
                    'min': 0.1,
                    'max': 50.0,
                    'description': 'Time where gain is selected, in ns.'
                },
                'alpha_exp_gain': {
                    'value': 1.0,
                    'type': 'float',
                    'min': 0.1,
                    'max': 5.0,
                    'description': 'Exponential order for intrinsic attenuation correction.'
                },
                't0_exp_gain': {
                    'value': 0.0,
                    'type': 'float',
                    'min': 0.0,
                    'max': 100.0,
                    'description': 'Gain applied from time t0, in ns.'
                },
                'te_exp_gain': {
                    'value': 70.0,
                    'type': 'float',
                    'min': 1.0,
                    'max': 200.0,
                    'description': 'Gain applied until te, in ns.'
                }
            },
            'mixed_phase_deconvolution': {
                'tWin_start': {
                    'value': 10.0,
                    'type': 'float',
                    'min': 0.0,
                    'max': 100.0,
                    'description': 'Start time for deconvolution window (ns).'
                },
                'tWin_end': {
                    'value': 70.0,
                    'type': 'float',
                    'min': 1.0,
                    'max': 200.0,
                    'description': 'End time for deconvolution window (ns).'
                },
                'wtr': {
                    'value': 5,
                    'type': 'int',
                    'min': 1,
                    'max': 20,
                    'description': 'Number of neighboring traces to combine.'
                },
                'mu': {
                    'value': 0.00001,
                    'type': 'float',
                    'min': 0.000001,
                    'max': 0.01,
                    'decimals': 6,
                    'step': 0.000001,
                    'description': 'White noise percentage.'
                }
            },
            'resample_profile': {
                'resample_factor': {
                    'value': 1,
                    'type': 'int',
                    'min': 1,
                    'max': 10,
                    'description': 'Resampling factor for profile data.'
                }
            },
            'lowpass_filter': {
                'cutoff_freq': {
                    'value': 0.8,
                    'type': 'float',
                    'min': 0.01,
                    'max': 2.0,
                    'description': 'Cutoff frequency for lowpass filter (GHz).'
                },
                'filter_order': {
                    'value': 4,
                    'type': 'int',
                    'min': 1,
                    'max': 10,
                    'description': 'Filter order for lowpass filter.'
                }
            },
            'hipass_filter': {
                'cutoff_freq': {
                    'value': 0.1,
                    'type': 'float',
                    'min': 0.01,
                    'max': 2.0,
                    'description': 'Cutoff frequency for highpass filter (GHz).'
                },
                'filter_order': {
                    'value': 4,
                    'type': 'int',
                    'min': 1,
                    'max': 10,
                    'description': 'Filter order for highpass filter.'
                }
            },
            'background_removal': {
                'method': {
                    'value': 'mean',
                    'type': 'choice',
                    'choices': ['mean', 'median', 'linear'],
                    'description': 'Background removal method.'
                },
                'window_size': {
                    'value': 10,
                    'type': 'int',
                    'min': 3,
                    'max': 100,
                    'description': 'Window size for background removal.'
                }
            },
            'background_removal_rgpr': {
                'method': {
                    'value': 'mean',
                    'type': 'choice',
                    'choices': ['mean', 'median', 'linear'],
                    'description': 'RGPR background removal method.'
                }
            },
            'horizontal_filter': {
                'filter_length': {
                    'value': 5,
                    'type': 'int',
                    'min': 3,
                    'max': 50,
                    'description': 'Horizontal filter length (traces).'
                }
            },
            'horizontal_filter_rgpr': {
                'filter_length': {
                    'value': 5,
                    'type': 'int',
                    'min': 3,
                    'max': 50,
                    'description': 'RGPR horizontal filter length (traces).'
                }
            },
            'dewow': {
                'window_size': {
                    'value': 10,
                    'type': 'int',
                    'min': 3,
                    'max': 100,
                    'description': 'Window size for dewow filter.'
                }
            },
            'kirchoff_migration_rgpr': {
                'velocity': {
                    'value': 0.1,
                    'type': 'float',
                    'min': 0.05,
                    'max': 0.3,
                    'description': 'Migration velocity (m/ns).'
                },
                'aperture': {
                    'value': 45.0,
                    'type': 'float',
                    'min': 10.0,
                    'max': 90.0,
                    'description': 'Migration aperture angle (degrees).'
                }
            },
            'visualization': {
                'plt_aspect': {
                    'value': 10,
                    'type': 'int',
                    'min': 1,
                    'max': 50,
                    'description': 'Plot aspect ratio.'
                },
                'plt_perc': {
                    'value': 1,
                    'type': 'int',
                    'min': 1,
                    'max': 10,
                    'description': 'Plot color scale percentage.'
                }
            }
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
        return True

    def execute(self, data: pd.DataFrame, params: Dict[str, Any],
                progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        
        if not GPR_UTILS_AVAILABLE:
            raise ProcessingError("GPR processing utilities are not available. Please check dependencies.")

        result = ProcessingResult(success=False, script_id=self.name)
        
        try:
            # --- 1. Extract Parameters ---
            if progress_callback: progress_callback(5, "Loading parameters...")
            npz_file = input_file_path
            
            if not npz_file or not Path(npz_file).exists():
                raise FileNotFoundError("A valid input .npz file is required but was not provided by the framework.")

            segment_num = params.get('target_selection', {}).get('segment_number', {}).get('value')
            trace_for_spectrum = params.get('target_selection', {}).get('trace_for_spectrum', {}).get('value')
            
            output_dir = Path(npz_file).parent
            
            # Extract processing steps
            processing_steps_params = params.get('processing_steps', {})
            processing_order = [step for step, config in processing_steps_params.items() if config.get('value')]
            
            # Use custom processing order if provided by the UI
            custom_order = params.get('processing_order', {}).get('value', [])
            if custom_order:
                # Filter to only include enabled steps
                enabled_steps = set(step for step, config in processing_steps_params.items() if config.get('value'))
                processing_order = [step for step in custom_order if step in enabled_steps]
                logger.info(f"Using custom processing order: {processing_order}")
            else:
                processing_order = [step for step, config in processing_steps_params.items() if config.get('value')]
                logger.info(f"Using default processing order: {processing_order}")
            
            if not processing_order:
                raise ValueError("No processing steps selected.")

            # Extract visualization parameters
            viz_params = params.get('visualization', {})
            plt_aspect = viz_params.get('plt_aspect', {}).get('value', 10)
            plt_perc = viz_params.get('plt_perc', {}).get('value', 1)
            
            # --- 2. Load Data ---
            if progress_callback: progress_callback(15, f"Loading data from {Path(npz_file).name}...")
            proj_data = ProjectGPRdata.from_filepath(npz_file)
            
            segment_data, dist_vec, df_pos = proj_data.getDataForSegment(segment_num)
            
            if trace_for_spectrum > segment_data.shape[1]:
                trace_for_spectrum = min(trace_for_spectrum, segment_data.shape[1])
                logger.warning(f"Trace number adjusted to {trace_for_spectrum} (max available)")
            
            # --- 3. Configure Processing ---
            if progress_callback: progress_callback(25, "Configuring processing pipeline...")
            
            process_config = ProcessingConfig()
            process_config.processing_order = processing_order
            process_config.dt = proj_data.config.get("dt", 1.0)
            
            # Configure individual processing steps based on parameters
            if 'dc_shift_correction_rgpr' in processing_order:
                dc_params = params.get('dc_shift_correction', {})
                process_config.dc_shift_correction.DCsample_start = dc_params.get('DCsample_start', {}).get('value', 250)
                process_config.dc_shift_correction.DCsample_end = dc_params.get('DCsample_end', {}).get('value', 450)
            
            if any('t0_correction' in step for step in processing_order):
                t0_params = params.get('t0_correction', {})
                process_config.t0_correction.time1 = t0_params.get('time1', {}).get('value', 5.0)
                process_config.t0_correction.time2 = t0_params.get('time2', {}).get('value', 17.0)
                process_config.t0_correction.threshold_t0 = t0_params.get('threshold_t0', {}).get('value', 0.06)
            
            if any('bandpass_filter' in step for step in processing_order):
                bp_params = params.get('bandpass_filter', {})
                process_config.band_pass_filter.f1_bandpass = bp_params.get('f1_bandpass', {}).get('value', 0.15)
                process_config.band_pass_filter.f2_bandpass = bp_params.get('f2_bandpass', {}).get('value', 0.55)
                process_config.band_pass_filter.poly_order_bandpass = bp_params.get('poly_order_bandpass', {}).get('value', 4)
            
            if any('amplitude_correction' in step for step in processing_order):
                amp_params = params.get('amplitude_correction', {})
                process_config.amplitude_correction.alpha_power_gain = amp_params.get('alpha_power_gain', {}).get('value', 1.0)
                process_config.amplitude_correction.te_power_gain = amp_params.get('te_power_gain', {}).get('value', 70.0)
                process_config.amplitude_correction.tcst_power_gain = amp_params.get('tcst_power_gain', {}).get('value', 3.0)
                process_config.amplitude_correction.alpha_exp_gain = amp_params.get('alpha_exp_gain', {}).get('value', 1.0)
                process_config.amplitude_correction.t0_exp_gain = amp_params.get('t0_exp_gain', {}).get('value', 0.0)
                process_config.amplitude_correction.te_exp_gain = amp_params.get('te_exp_gain', {}).get('value', 70.0)
            
            if 'mixed_phase_deconvolution_rgpr' in processing_order:
                mpd_params = params.get('mixed_phase_deconvolution', {})
                process_config.mixed_phase_deconvolution.tWin_start = mpd_params.get('tWin_start', {}).get('value', 10.0)
                process_config.mixed_phase_deconvolution.tWin_end = mpd_params.get('tWin_end', {}).get('value', 70.0)
                process_config.mixed_phase_deconvolution.wtr = mpd_params.get('wtr', {}).get('value', 5)
                process_config.mixed_phase_deconvolution.mu = mpd_params.get('mu', {}).get('value', 0.00001)
                # Inherit bandpass filter parameters for mixed phase deconvolution
                bp_params = params.get('bandpass_filter', {})
                process_config.mixed_phase_deconvolution.f1_bandpass = bp_params.get('f1_bandpass', {}).get('value', 0.15)
                process_config.mixed_phase_deconvolution.f2_bandpass = bp_params.get('f2_bandpass', {}).get('value', 0.55)
                process_config.mixed_phase_deconvolution.poly_order_bandpass = bp_params.get('poly_order_bandpass', {}).get('value', 4)

            if 'power_gain_rgpr' in processing_order:
                # Power gain uses the same parameters as amplitude correction
                amp_params = params.get('amplitude_correction', {})
                process_config.amplitude_correction.alpha_power_gain = amp_params.get('alpha_power_gain', {}).get('value', 1.0)
                process_config.amplitude_correction.te_power_gain = amp_params.get('te_power_gain', {}).get('value', 70.0)
                process_config.amplitude_correction.tcst_power_gain = amp_params.get('tcst_power_gain', {}).get('value', 3.0)

            if 'resample_profile' in processing_order:
                resample_params = params.get('resample_profile', {})
                if hasattr(process_config, 'resample_profile'):
                    process_config.resample_profile.factor = resample_params.get('resample_factor', {}).get('value', 1)
            
            if any('lowpass_filter' in step for step in processing_order):
                lp_params = params.get('lowpass_filter', {})
                if hasattr(process_config, 'lowpass_filter'):
                    process_config.lowpass_filter.cutoff_freq = lp_params.get('cutoff_freq', {}).get('value', 0.8)
                    process_config.lowpass_filter.filter_order = lp_params.get('filter_order', {}).get('value', 4)
            
            if any('hipass_filter' in step for step in processing_order):
                hp_params = params.get('hipass_filter', {})
                if hasattr(process_config, 'hipass_filter'):
                    process_config.hipass_filter.cutoff_freq = hp_params.get('cutoff_freq', {}).get('value', 0.1)
                    process_config.hipass_filter.filter_order = hp_params.get('filter_order', {}).get('value', 4)
            
            if any('background_removal' in step for step in processing_order):
                bg_params = params.get('background_removal', {})
                if hasattr(process_config, 'background_removal'):
                    process_config.background_removal.method = bg_params.get('method', {}).get('value', 'mean')
                    process_config.background_removal.window_size = bg_params.get('window_size', {}).get('value', 10)
                
                # Also configure RGPR background removal
                bg_rgpr_params = params.get('background_removal_rgpr', {})
                if hasattr(process_config, 'background_removal_rgpr'):
                    process_config.background_removal_rgpr.method = bg_rgpr_params.get('method', {}).get('value', 'mean')
            
            if any('horizontal_filter' in step for step in processing_order):
                hf_params = params.get('horizontal_filter', {})
                if hasattr(process_config, 'horizontal_filter'):
                    process_config.horizontal_filter.filter_length = hf_params.get('filter_length', {}).get('value', 5)
                
                # Also configure RGPR horizontal filter
                hf_rgpr_params = params.get('horizontal_filter_rgpr', {})
                if hasattr(process_config, 'horizontal_filter_rgpr'):
                    process_config.horizontal_filter_rgpr.filter_length = hf_rgpr_params.get('filter_length', {}).get('value', 5)
            
            if 'dewow' in processing_order:
                dewow_params = params.get('dewow', {})
                if hasattr(process_config, 'dewow'):
                    process_config.dewow.window_size = dewow_params.get('window_size', {}).get('value', 10)
            
            if 'kirchoff_migration_rgpr' in processing_order:
                km_params = params.get('kirchoff_migration_rgpr', {})
                if hasattr(process_config, 'kirchoff_migration'):
                    process_config.kirchoff_migration.velocity = km_params.get('velocity', {}).get('value', 0.1)
                    process_config.kirchoff_migration.aperture = km_params.get('aperture', {}).get('value', 45.0)

            logger.info(f"Processing order set to: {processing_order}")

            # --- 4. Process Segment ---
            if progress_callback: progress_callback(40, f"Processing segment {segment_num}...")
            
            try:
                segm_processor = GPRdataSegmentProcessor(process_config, proj_data.config_gpr)
                logger.info("GPR segment processor initialized successfully")
            except Exception as init_err:
                raise ProcessingError(f"Failed to initialize GPR processor: {init_err}")
            
            # Log data characteristics before processing
            logger.info(f"Segment data shape: {segment_data.shape}")
            logger.info(f"Segment data range: [{np.min(segment_data):.6f}, {np.max(segment_data):.6f}]")
            logger.info(f"Processing order: {processing_order}")
            
            # Check for problematic data before processing
            if np.any(np.isnan(segment_data)):
                logger.warning("Segment data contains NaN values")
                segment_data = np.nan_to_num(segment_data, nan=0.0)
                
            if np.any(np.isinf(segment_data)):
                logger.warning("Segment data contains infinite values")
                segment_data = np.nan_to_num(segment_data, posinf=0.0, neginf=0.0)
            
            # Process the entire segment with enhanced error handling
            try:
                logger.info("Starting segment processing...")
                
                # Add timeout handling for R-based operations
                if 'mixed_phase_deconvolution_rgpr' in processing_order:
                    logger.warning("Mixed phase deconvolution detected - this may take longer due to R processing")
                    if progress_callback: 
                        progress_callback(45, "Processing mixed phase deconvolution (this may take a while)...")
                
                processed_segment, profile_list = segm_processor.process_data(segment_data, process_config)
                logger.info("Segment processing completed successfully")
                
            except Exception as proc_err:
                logger.error(f"Error during segment processing: {proc_err}")
                
                # If mixed phase deconvolution is causing issues, try without it
                if 'mixed_phase_deconvolution_rgpr' in processing_order:
                    logger.warning("Retrying without mixed_phase_deconvolution_rgpr due to processing error")
                    modified_order = [step for step in processing_order if step != 'mixed_phase_deconvolution_rgpr']
                    process_config.processing_order = modified_order
                    
                    try:
                        processed_segment, profile_list = segm_processor.process_data(segment_data, process_config)
                        logger.info("Segment processing completed successfully without mixed phase deconvolution")
                        processing_order = modified_order  # Update for plot generation
                    except Exception as retry_err:
                        raise ProcessingError(f"Segment processing failed even without mixed phase deconvolution: {retry_err}")
                else:
                    raise ProcessingError(f"Segment processing failed: {proc_err}")
            
            # --- 5. Generate and Save Plots ---
            if progress_callback: progress_callback(60, "Generating profile plots...")
            dt = process_config.dt
            
            # Plot 1: Raw segment profile
            raw_fig = plotting_profile_geo(segment_data, dist_vec, dt, 1, plt_aspect, f'Segment {segment_num} - Raw', plt_perc)
            raw_plot_path = output_dir / f"{Path(npz_file).stem}_seg{segment_num}_raw.mplplot"
            with open(raw_plot_path, 'wb') as f:
                pickle.dump(raw_fig, f)
            result.add_output_file(str(raw_plot_path), 'matplotlib_plot', f'Raw profile for segment {segment_num}')
            plt.close(raw_fig)
            
            # Plot 2-N: Intermediate processing steps
            for i, step_name in enumerate(processing_order):
                if i < len(profile_list):
                    step_fig = plotting_profile_geo(
                        profile_list[i], dist_vec, dt, 1, plt_aspect, 
                        f'Step {i+1}: {step_name.replace("_", " ").title()}', plt_perc
                    )
                    step_plot_path = output_dir / f"{Path(npz_file).stem}_seg{segment_num}_step{i+1}_{step_name}.mplplot"
                    with open(step_plot_path, 'wb') as f:
                        pickle.dump(step_fig, f)
                    result.add_output_file(str(step_plot_path), 'matplotlib_plot', f'Processing step {i+1}: {step_name}')
                    plt.close(step_fig)
            
            # Plot N+1: Power spectrum comparison
            if progress_callback: progress_callback(80, "Generating power spectrum comparison...")
            
            trace_idx = trace_for_spectrum - 1  # Convert to 0-based index
            if trace_idx >= segment_data.shape[1]:
                trace_idx = segment_data.shape[1] - 1
            
            spectrum_fig, ax = plt.subplots(figsize=(10, 6))
            
            # Raw trace spectrum
            raw_trace = segment_data[:, trace_idx]
            n_samples_raw = raw_trace.shape[0]
            frequencies_raw = np.fft.fftfreq(n_samples_raw, dt * 1e-9)
            raw_ps = np.abs(np.fft.fft(raw_trace))**2
            ax.plot(frequencies_raw, raw_ps / np.max(raw_ps), label='Raw Trace', linewidth=3)

            # Processed trace spectrum
            processed_trace = processed_segment[:, trace_idx]
            n_samples_proc = processed_trace.shape[0]
            frequencies_proc = np.fft.fftfreq(n_samples_proc, dt * 1e-9)
            proc_ps = np.abs(np.fft.fft(processed_trace))**2
            ax.plot(frequencies_proc, proc_ps / np.max(proc_ps), label='Processed Trace')

            ax.set_title(f'Power Spectrum Comparison (Trace {trace_for_spectrum})')
            ax.set_xlabel('Frequency (Hz)')
            ax.set_ylabel('Normalized Power')
            ax.grid(True)
            ax.set_xlim(0, 2e9)
            ax.legend()
            
            spectrum_plot_path = output_dir / f"{Path(npz_file).stem}_seg{segment_num}_trace{trace_for_spectrum}_spectrum.mplplot"
            with open(spectrum_plot_path, 'wb') as f:
                pickle.dump(spectrum_fig, f)
            result.add_output_file(str(spectrum_plot_path), 'matplotlib_plot', f'Power spectrum for trace {trace_for_spectrum}')
            plt.close(spectrum_fig)
            
            # --- 6. Finalize ---
            if progress_callback: progress_callback(100, "Processing complete.")
            result.success = True
            result.message = (f"Successfully processed segment {segment_num} with {len(processing_order)} steps. "
                              f"{len(result.output_files)} plots saved to {output_dir.name} directory.")
            logger.success(result.message)

        except RRuntimeError as r_err:
            error_message = f"An R error occurred during segment processing: {r_err}"
            logger.error(error_message, exc_info=True)
            result.error_message = error_message
        except Exception as e:
            logger.error(f"Segment processing failed: {e}", exc_info=True)
            result.error_message = f"Segment processing failed: {str(e)}"

        return result

# Export for framework discovery
SCRIPT_CLASS = SegmentProcessor 