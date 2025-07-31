"""
Example Custom Magnetic Processing Script

This script demonstrates how to create a custom magnetic data processing script
that can be used alongside the built-in UXO Wizard scripts.

Features demonstrated:
- Custom anomaly detection using statistical methods
- Gradient calculation for enhanced anomaly detection
- Custom visualization with multiple subplots
- Proper parameter configuration
- Layer output for map display

Copy this file to scripts/magnetic/ to use it in the magnetic processor.
"""

from src.processing.base import ScriptInterface, ProcessingResult, ScriptMetadata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
from typing import Dict, Any, Optional, Callable

class CustomMagneticAnalysis(ScriptInterface):
    """Custom magnetic anomaly detection with statistical analysis"""
    
    @property
    def name(self) -> str:
        return "Custom Magnetic Analysis"
    
    @property
    def description(self) -> str:
        return "Statistical anomaly detection with gradient analysis for magnetic data"
    
    def get_metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            description=self.description,
            flags=["magnetic", "anomaly", "statistical", "user"],
            typical_use_case="Enhanced anomaly detection with statistical methods",
            field_compatible=True,
            estimated_runtime="2-5 minutes"
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        return {
            'detection_settings': {
                'anomaly_threshold': {
                    'value': 2.5,
                    'type': 'float',
                    'min': 1.0,
                    'max': 5.0,
                    'description': 'Standard deviations for anomaly detection'
                },
                'use_gradient': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Include gradient analysis for enhanced detection'
                },
                'smoothing_window': {
                    'value': 5,
                    'type': 'int',
                    'min': 1,
                    'max': 20,
                    'description': 'Window size for data smoothing'
                }
            },
            'analysis_options': {
                'remove_outliers': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Remove statistical outliers before processing'
                },
                'analysis_method': {
                    'value': 'zscore',
                    'type': 'choice',
                    'choices': ['zscore', 'iqr', 'modified_zscore'],
                    'description': 'Statistical method for anomaly detection'
                }
            },
            'visualization': {
                'create_plots': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Create detailed analysis plots'
                },
                'plot_gradients': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Include gradient plots in visualization'
                }
            }
        }
    
    def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None, 
                input_file_path: Optional[str] = None) -> ProcessingResult:
        
        if progress_callback:
            progress_callback(0, "Starting custom magnetic analysis...")
        
        try:
            # Extract parameters
            anomaly_threshold = params.get('detection_settings', {}).get('anomaly_threshold', {}).get('value', 2.5)
            use_gradient = params.get('detection_settings', {}).get('use_gradient', {}).get('value', True)
            smoothing_window = params.get('detection_settings', {}).get('smoothing_window', {}).get('value', 5)
            remove_outliers = params.get('analysis_options', {}).get('remove_outliers', {}).get('value', True)
            analysis_method = params.get('analysis_options', {}).get('analysis_method', {}).get('value', 'zscore')
            create_plots = params.get('visualization', {}).get('create_plots', {}).get('value', True)
            plot_gradients = params.get('visualization', {}).get('plot_gradients', {}).get('value', True)
            
            # Validate data
            required_columns = ['latitude', 'longitude']
            magnetic_columns = [col for col in data.columns if any(mag_word in col.lower() 
                               for mag_word in ['mag', 'field', 'btotal', 'magnetic'])]
            
            if not magnetic_columns:
                raise ValueError("No magnetic field columns found in data")
            
            # Use the first magnetic column found
            magnetic_col = magnetic_columns[0]
            required_columns.append(magnetic_col)
            
            missing_cols = [col for col in required_columns if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            if progress_callback:
                progress_callback(10, f"Processing magnetic data from column: {magnetic_col}")
            
            # Create working copy
            processed_data = data.copy()
            
            # Remove outliers if requested
            if remove_outliers:
                if progress_callback:
                    progress_callback(15, "Removing statistical outliers...")
                
                Q1 = processed_data[magnetic_col].quantile(0.25)
                Q3 = processed_data[magnetic_col].quantile(0.75)
                IQR = Q3 - Q1
                lower_bound = Q1 - 1.5 * IQR
                upper_bound = Q3 + 1.5 * IQR
                
                outlier_mask = (processed_data[magnetic_col] >= lower_bound) & (processed_data[magnetic_col] <= upper_bound)
                processed_data = processed_data[outlier_mask].copy()
                
                if progress_callback:
                    progress_callback(20, f"Removed {len(data) - len(processed_data)} outliers")
            
            # Apply smoothing
            if smoothing_window > 1:
                if progress_callback:
                    progress_callback(25, f"Applying smoothing with window size {smoothing_window}...")
                
                processed_data['smoothed_field'] = processed_data[magnetic_col].rolling(
                    window=smoothing_window, center=True
                ).mean()
                processed_data['smoothed_field'] = processed_data['smoothed_field'].fillna(processed_data[magnetic_col])
            else:
                processed_data['smoothed_field'] = processed_data[magnetic_col]
            
            if progress_callback:
                progress_callback(35, f"Performing {analysis_method} anomaly detection...")
            
            # Perform anomaly detection
            if analysis_method == 'zscore':
                z_scores = np.abs(stats.zscore(processed_data['smoothed_field']))
                anomaly_mask = z_scores > anomaly_threshold
                processed_data['anomaly_score'] = z_scores
            elif analysis_method == 'iqr':
                Q1 = processed_data['smoothed_field'].quantile(0.25)
                Q3 = processed_data['smoothed_field'].quantile(0.75)
                IQR = Q3 - Q1
                anomaly_mask = (processed_data['smoothed_field'] < (Q1 - anomaly_threshold * IQR)) | \
                              (processed_data['smoothed_field'] > (Q3 + anomaly_threshold * IQR))
                processed_data['anomaly_score'] = np.abs(processed_data['smoothed_field'] - processed_data['smoothed_field'].median()) / (IQR / 1.349)
            elif analysis_method == 'modified_zscore':
                median = processed_data['smoothed_field'].median()
                mad = np.median(np.abs(processed_data['smoothed_field'] - median))
                modified_z_scores = 0.6745 * (processed_data['smoothed_field'] - median) / mad
                anomaly_mask = np.abs(modified_z_scores) > anomaly_threshold
                processed_data['anomaly_score'] = np.abs(modified_z_scores)
            
            processed_data['is_anomaly'] = anomaly_mask
            
            # Calculate gradients if requested
            if use_gradient:
                if progress_callback:
                    progress_callback(50, "Calculating magnetic field gradients...")
                
                # Sort by coordinates for gradient calculation
                processed_data = processed_data.sort_values(['latitude', 'longitude'])
                
                # Calculate gradients (simplified version)
                processed_data['gradient_x'] = np.gradient(processed_data['smoothed_field'])
                processed_data['gradient_magnitude'] = np.abs(processed_data['gradient_x'])
                
                # Enhanced anomaly detection using gradients
                gradient_threshold = processed_data['gradient_magnitude'].quantile(0.95)
                gradient_anomalies = processed_data['gradient_magnitude'] > gradient_threshold
                
                # Combine original anomalies with gradient anomalies
                processed_data['combined_anomaly'] = processed_data['is_anomaly'] | gradient_anomalies
                processed_data['gradient_anomaly'] = gradient_anomalies
            
            if progress_callback:
                progress_callback(70, "Creating analysis results...")
            
            # Create result
            result = ProcessingResult(
                success=True,
                data=processed_data,
                processor_type='magnetic',
                script_id='custom_magnetic_analysis'
            )
            
            # Add main data layer
            result.add_layer_output(
                layer_type='points',
                data=processed_data,
                style_info={
                    'color_field': 'anomaly_score',
                    'use_graduated_colors': True,
                    'color_scheme': ['#0000FF', '#00FFFF', '#00FF00', '#FFFF00', '#FF8000', '#FF0000'],
                    'size': 4
                },
                metadata={
                    'layer_name': 'Custom Magnetic Analysis - All Data',
                    'analysis_method': analysis_method,
                    'threshold': anomaly_threshold,
                    'total_points': len(processed_data),
                    'anomalies_detected': processed_data['is_anomaly'].sum()
                }
            )
            
            # Add anomaly-only layer
            anomaly_data = processed_data[processed_data['is_anomaly']].copy()
            if len(anomaly_data) > 0:
                result.add_layer_output(
                    layer_type='points',
                    data=anomaly_data,
                    style_info={
                        'color': '#FF0000',
                        'size': 8,
                        'opacity': 0.8
                    },
                    metadata={
                        'layer_name': 'Detected Anomalies',
                        'anomaly_count': len(anomaly_data)
                    }
                )
            
            # Add gradient anomalies layer if applicable
            if use_gradient and 'combined_anomaly' in processed_data.columns:
                gradient_anomaly_data = processed_data[processed_data['gradient_anomaly']].copy()
                if len(gradient_anomaly_data) > 0:
                    result.add_layer_output(
                        layer_type='points',
                        data=gradient_anomaly_data,
                        style_info={
                            'color': '#FF8000',
                            'size': 6,
                            'opacity': 0.7
                        },
                        metadata={
                            'layer_name': 'Gradient Anomalies',
                            'gradient_anomaly_count': len(gradient_anomaly_data)
                        }
                    )
            
            # Create visualization
            if create_plots:
                if progress_callback:
                    progress_callback(85, "Creating visualization plots...")
                
                n_plots = 3 if (use_gradient and plot_gradients) else 2
                fig, axes = plt.subplots(n_plots, 1, figsize=(12, 4*n_plots))
                if n_plots == 1:
                    axes = [axes]
                
                # Plot 1: Original vs Smoothed Data
                ax1 = axes[0]
                ax1.plot(processed_data[magnetic_col], alpha=0.6, label='Original', linewidth=1)
                ax1.plot(processed_data['smoothed_field'], alpha=0.8, label='Smoothed', linewidth=1.5)
                ax1.axhline(y=processed_data['smoothed_field'].mean(), color='k', linestyle='--', alpha=0.5, label='Mean')
                ax1.set_title('Magnetic Field Data - Original vs Smoothed')
                ax1.set_ylabel('Magnetic Field (nT)')
                ax1.legend()
                ax1.grid(True, alpha=0.3)
                
                # Plot 2: Anomaly Detection Results
                ax2 = axes[1]
                scatter = ax2.scatter(range(len(processed_data)), processed_data['smoothed_field'], 
                                    c=processed_data['anomaly_score'], cmap='viridis', alpha=0.7, s=20)
                anomaly_indices = processed_data[processed_data['is_anomaly']].index
                if len(anomaly_indices) > 0:
                    ax2.scatter(anomaly_indices, processed_data.loc[anomaly_indices, 'smoothed_field'], 
                              color='red', s=50, alpha=0.8, label=f'Anomalies ({len(anomaly_indices)})')
                ax2.set_title(f'Anomaly Detection Results ({analysis_method.upper()}, threshold={anomaly_threshold})')
                ax2.set_xlabel('Data Point Index')
                ax2.set_ylabel('Magnetic Field (nT)')
                plt.colorbar(scatter, ax=ax2, label='Anomaly Score')
                if len(anomaly_indices) > 0:
                    ax2.legend()
                ax2.grid(True, alpha=0.3)
                
                # Plot 3: Gradient Analysis (if enabled)
                if use_gradient and plot_gradients and n_plots > 2:
                    ax3 = axes[2]
                    ax3.plot(processed_data['gradient_magnitude'], alpha=0.7, color='orange', linewidth=1)
                    gradient_anomaly_indices = processed_data[processed_data.get('gradient_anomaly', False)].index
                    if len(gradient_anomaly_indices) > 0:
                        ax3.scatter(gradient_anomaly_indices, 
                                  processed_data.loc[gradient_anomaly_indices, 'gradient_magnitude'],
                                  color='red', s=30, alpha=0.8, label=f'Gradient Anomalies ({len(gradient_anomaly_indices)})')
                    ax3.set_title('Magnetic Field Gradient Analysis')
                    ax3.set_xlabel('Data Point Index')
                    ax3.set_ylabel('Gradient Magnitude')
                    if len(gradient_anomaly_indices) > 0:
                        ax3.legend()
                    ax3.grid(True, alpha=0.3)
                
                plt.tight_layout()
                result.figure = fig
            
            if progress_callback:
                progress_callback(100, f"Analysis complete! Detected {processed_data['is_anomaly'].sum()} anomalies")
            
            return result
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Custom magnetic analysis failed: {str(e)}",
                processor_type='magnetic'
            )

# Required: Export the script class
SCRIPT_CLASS = CustomMagneticAnalysis