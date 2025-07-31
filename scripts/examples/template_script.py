"""
UXO Wizard User Script Template

Instructions:
1. Copy this file to the appropriate processor folder (magnetic/, gamma/, gpr/, multispectral/)
2. Rename the file and class to something descriptive
3. Implement the required methods
4. Only use packages available in UXO Wizard (see README.md for list)
5. If you need a new package, open an issue at: https://github.com/your-repo/UXO-Wizard/issues

Available packages: 
- Data Processing: pandas, numpy, scipy
- Visualization: matplotlib, plotly, seaborn  
- Geospatial: geopandas, rasterio, fiona
- Machine Learning: scikit-learn
- Other: requests, lxml, openpyxl

Example usage:
- Copy to scripts/magnetic/my_custom_analysis.py
- Rename class to MyCustomAnalysis
- Implement your processing logic
"""

from src.processing.base import ScriptInterface, ProcessingResult, ScriptMetadata
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from typing import Dict, Any, Optional, Callable

class TemplateScript(ScriptInterface):
    """Template for creating custom UXO Wizard processing scripts"""
    
    @property
    def name(self) -> str:
        return "Custom Script Template"
    
    @property
    def description(self) -> str:
        return "Template for creating custom processing scripts"
    
    def get_metadata(self) -> ScriptMetadata:
        return ScriptMetadata(
            description=self.description,
            flags=["template", "example", "user"],
            typical_use_case="Learning script development",
            field_compatible=True,
            estimated_runtime="< 1 minute"
        )
    
    def get_parameters(self) -> Dict[str, Any]:
        """Define the parameters that users can configure in the UI"""
        return {
            'processing_options': {
                'threshold': {
                    'value': 2.0,
                    'type': 'float',
                    'min': 0.1,
                    'max': 10.0,
                    'description': 'Processing threshold value'
                },
                'method': {
                    'value': 'default',
                    'type': 'choice',
                    'choices': ['default', 'advanced', 'fast'],
                    'description': 'Processing method to use'
                },
                'create_visualization': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Create visualization plots'
                }
            },
            'output_options': {
                'output_format': {
                    'value': 'csv',
                    'type': 'choice',
                    'choices': ['csv', 'xlsx', 'json'],
                    'description': 'Output file format'
                }
            }
        }
    
    def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None, 
                input_file_path: Optional[str] = None) -> ProcessingResult:
        """
        Execute the script with provided data and parameters
        
        Args:
            data: Input DataFrame to process
            params: Processing parameters from UI
            progress_callback: Optional callback for progress updates
            input_file_path: Optional path to the original input file
            
        Returns:
            ProcessingResult with success status, output files, and layer data
        """
        
        if progress_callback:
            progress_callback(0, "Starting custom processing...")
        
        try:
            # Extract parameters
            threshold = params.get('processing_options', {}).get('threshold', {}).get('value', 2.0)
            method = params.get('processing_options', {}).get('method', {}).get('value', 'default')
            create_viz = params.get('processing_options', {}).get('create_visualization', {}).get('value', True)
            output_format = params.get('output_options', {}).get('output_format', {}).get('value', 'csv')
            
            if progress_callback:
                progress_callback(10, f"Processing with threshold {threshold} using {method} method...")
            
            # Example processing logic
            processed_data = data.copy()
            
            # Validate required columns (example for magnetic data)
            required_cols = ['latitude', 'longitude']  # Adjust based on your processor type
            missing_cols = [col for col in required_cols if col not in data.columns]
            if missing_cols:
                raise ValueError(f"Missing required columns: {missing_cols}")
            
            if progress_callback:
                progress_callback(25, "Applying processing algorithm...")
            
            # Example: Add a processed column based on threshold
            if 'magnetic_field' in data.columns:
                processed_data['processed_field'] = np.where(
                    data['magnetic_field'] > threshold,
                    data['magnetic_field'] * 1.1,  # Enhance strong signals
                    data['magnetic_field'] * 0.9   # Reduce weak signals
                )
            
            if progress_callback:
                progress_callback(50, "Processing complete, generating outputs...")
            
            # Create result object
            result = ProcessingResult(
                success=True,
                data=processed_data,
                processor_type='custom',  # Will be set by the processor
                script_id='template_script'
            )
            
            # Add layer output for map display
            result.add_layer_output(
                layer_type='points',
                data=processed_data,
                style_info={
                    'color': '#FF6600',
                    'size': 5,
                    'color_field': 'processed_field' if 'processed_field' in processed_data.columns else None
                },
                metadata={
                    'script_name': self.name,
                    'threshold_used': threshold,
                    'method_used': method,
                    'total_points': len(processed_data)
                }
            )
            
            if progress_callback:
                progress_callback(75, "Creating output files...")
            
            # Save output file (optional - the pipeline can handle this automatically)
            if self.project_manager:
                working_dir = self.get_project_working_directory()
                if working_dir:
                    import os
                    output_filename = f"custom_processed_data.{output_format}"
                    output_path = os.path.join(working_dir, output_filename)
                    
                    if output_format == 'csv':
                        processed_data.to_csv(output_path, index=False)
                    elif output_format == 'xlsx':
                        processed_data.to_excel(output_path, index=False)
                    elif output_format == 'json':
                        processed_data.to_json(output_path, orient='records', indent=2)
                    
                    result.add_output_file(
                        file_path=output_path,
                        file_type=output_format,
                        description=f"Custom processed data in {output_format.upper()} format"
                    )
            
            # Create visualization if requested
            if create_viz and progress_callback:
                progress_callback(85, "Creating visualization...")
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                if 'latitude' in processed_data.columns and 'longitude' in processed_data.columns:
                    scatter = ax.scatter(
                        processed_data['longitude'], 
                        processed_data['latitude'],
                        c=processed_data.get('processed_field', processed_data.iloc[:, -1]),
                        cmap='viridis',
                        alpha=0.7,
                        s=20
                    )
                    ax.set_xlabel('Longitude')
                    ax.set_ylabel('Latitude')
                    ax.set_title(f'Custom Processing Results (Threshold: {threshold})')
                    plt.colorbar(scatter, label='Processed Values')
                else:
                    # Fallback visualization
                    processed_data.plot(ax=ax)
                    ax.set_title('Custom Processing Results')
                
                plt.tight_layout()
                result.figure = fig
            
            if progress_callback:
                progress_callback(100, "Processing complete!")
            
            return result
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Custom script failed: {str(e)}",
                processor_type='custom'
            )

# Required: Export the script class
SCRIPT_CLASS = TemplateScript