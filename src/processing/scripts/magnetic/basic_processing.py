"""
Basic magnetic data processing script
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, Callable

from src.processing.base import ScriptInterface, ProcessingResult, ProcessingError


class BasicMagneticProcessing(ScriptInterface):
    """Basic magnetic data processing with simple column operations"""
    
    @property
    def name(self) -> str:
        return "Basic Processing"
    
    @property
    def description(self) -> str:
        return "Basic magnetic data processing with column duplication and simple statistics"
    
    def get_parameters(self) -> Dict[str, Any]:
        """Return script-specific parameters"""
        return {
            'processing_options': {
                'create_processed_column': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Create processed data column'
                },
                'calculate_statistics': {
                    'value': True,
                    'type': 'bool',
                    'description': 'Calculate basic statistics'
                }
            },
            'output_options': {
                'generate_visualization': {
                    'value': False,
                    'type': 'bool',
                    'description': 'Generate visualization files'
                },
                'layer_type': {
                    'value': 'point_data',
                    'type': 'choice',
                    'choices': ['point_data', 'grid_visualization', 'flight_lines'],
                    'description': 'Type of layer to generate'
                }
            }
        }
    
    def validate_data(self, data: pd.DataFrame) -> bool:
        """Validate magnetic data"""
        if data.empty:
            raise ProcessingError("No data provided")
        
        if len(data.columns) < 2:
            raise ProcessingError("Data must have at least 2 columns")
        
        return True
    
    def execute(self, data: pd.DataFrame, params: Dict[str, Any], 
                progress_callback: Optional[Callable] = None, input_file_path: Optional[str] = None) -> ProcessingResult:
        """Execute basic magnetic processing"""
        try:
            if progress_callback:
                progress_callback(0, "Starting basic magnetic processing...")
            
            # Create result
            result = ProcessingResult(success=True)
            result_data = data.copy()
            
            # Get processing options
            create_processed = params.get('processing_options', {}).get('create_processed_column', {}).get('value', True)
            calc_stats = params.get('processing_options', {}).get('calculate_statistics', {}).get('value', True)
            gen_viz = params.get('output_options', {}).get('generate_visualization', {}).get('value', False)
            layer_type = params.get('output_options', {}).get('layer_type', {}).get('value', 'point_data')
            
            if progress_callback:
                progress_callback(20, "Processing magnetic data...")
            
            # Create processed column if requested
            if create_processed:
                # Find first numeric column
                numeric_cols = result_data.select_dtypes(include=[np.number]).columns
                if len(numeric_cols) > 0:
                    first_col = numeric_cols[0]
                    result_data[f'{first_col}_processed'] = result_data[first_col]
                    
            if progress_callback:
                progress_callback(50, "Calculating statistics...")
            
            # Calculate statistics if requested
            if calc_stats:
                numeric_cols = result_data.select_dtypes(include=[np.number]).columns
                for col in numeric_cols:
                    result_data[f'{col}_mean'] = result_data[col].mean()
                    result_data[f'{col}_std'] = result_data[col].std()
            
            if progress_callback:
                progress_callback(80, "Generating outputs...")
            
            # Set processed data
            result.data = result_data
            
            # Add basic metadata
            result.metadata = {
                'processor': 'magnetic',
                'script': 'basic_processing',
                'data_shape': result_data.shape,
                'columns_processed': len(result_data.columns),
                'parameters': params
            }
            
            # Generate visualization layers if requested
            if gen_viz and result_data is not None and not result_data.empty:
                self._generate_basic_layers(result_data, result, layer_type)
            elif result_data is not None and not result_data.empty:
                # Always generate at least one basic layer
                coord_cols = self._detect_coordinates(result_data)
                if coord_cols:
                    result.add_layer_output(
                        layer_type=layer_type,
                        data=result_data,
                        style_info={
                            'color': 'blue',
                            'size': 3,
                            'opacity': 0.7
                        },
                        metadata={
                            'description': 'Basic magnetic data processing output',
                            'coordinate_columns': coord_cols,
                            'total_points': len(result_data),
                            'data_type': 'basic_processing'
                        }
                    )
            
            # Generate visualization file if requested
            if gen_viz:
                # For now, just add a placeholder - in real implementation would generate actual files
                result.add_output_file(
                    file_path="magnetic_basic_visualization.png",
                    file_type="png",
                    description="Basic magnetic data visualization",
                    metadata={'generated_by': 'basic_processing'}
                )
            
            if progress_callback:
                progress_callback(100, "Basic processing complete!")
            
            return result
            
        except Exception as e:
            return ProcessingResult(
                success=False,
                error_message=f"Basic processing failed: {str(e)}"
            )
    
    def _detect_coordinates(self, data: pd.DataFrame) -> Dict[str, str]:
        """Detect coordinate columns in the data"""
        coord_patterns = {
            'latitude': ['lat', 'latitude', 'y', 'northing'],
            'longitude': ['lon', 'lng', 'longitude', 'x', 'easting']
        }
        
        detected = {}
        for coord_type, patterns in coord_patterns.items():
            for col in data.columns:
                col_lower = col.lower()
                if any(pattern in col_lower for pattern in patterns):
                    detected[coord_type] = col
                    break
        
        return detected
    
    def _generate_basic_layers(self, data: pd.DataFrame, result: ProcessingResult, layer_type: str):
        """
        Demonstrate multiple layer generation patterns for basic processing
        This shows other processor developers how to create multiple layers
        """
        coord_cols = self._detect_coordinates(data)
        
        if not coord_cols:
            return
        
        # 1. Main data layer - demonstrates point layer with coordinate detection
        result.add_layer_output(
            layer_type=layer_type,
            data=data,
            style_info={
                'color': '#0066CC',
                'size': 4,
                'opacity': 0.8,
                'use_graduated_colors': False
            },
            metadata={
                'description': 'Basic processed magnetic data',
                'coordinate_columns': coord_cols,
                'total_points': len(data),
                'data_type': 'basic_main_data'
            }
        )
        
        # 2. Numeric data layer - demonstrates graduated colors based on first numeric column
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        if len(numeric_cols) > 0:
            first_numeric = numeric_cols[0]
            
            result.add_layer_output(
                layer_type='points',
                data=data,
                style_info={
                    'color_field': first_numeric,
                    'use_graduated_colors': True,
                    'color_scheme': 'viridis',
                    'size': 5,
                    'opacity': 0.9
                },
                metadata={
                    'description': f'Data colored by {first_numeric}',
                    'coordinate_columns': coord_cols,
                    'color_field': first_numeric,
                    'total_points': len(data),
                    'data_type': 'colored_by_value'
                }
            )
        
        # 3. Subsampled layer - demonstrates data reduction for performance
        if len(data) > 100:
            # Create subsampled version for overview
            step = max(1, len(data) // 50)  # Reduce to ~50 points
            subsampled = data.iloc[::step].copy()
            
            result.add_layer_output(
                layer_type='points',
                data=subsampled,
                style_info={
                    'color': '#FF6600',
                    'size': 6,
                    'opacity': 0.7,
                    'show_labels': True
                },
                metadata={
                    'description': f'Subsampled overview ({len(subsampled)} points)',
                    'coordinate_columns': coord_cols,
                    'original_points': len(data),
                    'subsampled_points': len(subsampled),
                    'data_type': 'subsampled_overview'
                }
            )


# Export the script class
SCRIPT_CLASS = BasicMagneticProcessing