"""
Automatic Configuration for SGTool Python
=========================================

Automated processing configuration that takes CSV directories,
runs all filters, and creates comprehensive interactive maps.
"""

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Union
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class MagneticFieldConfig:
    """Configuration for magnetic field parameters."""
    inclination: float = 70.0  # degrees
    declination: float = 2.0   # degrees
    auto_detect: bool = True   # Try to detect from data location


@dataclass
class ProcessingConfig:
    """Configuration for data processing parameters."""
    grid_resolution: int = 300
    max_kriging_points: int = 15000
    interpolation_method: str = 'minimum_curvature'  # 'minimum_curvature', 'kriging', 'linear', 'cubic'
    enable_all_filters: bool = True
    custom_filters: Optional[List[str]] = None
    enable_boundary_masking: bool = True
    boundary_method: str = 'convex_hull'  # 'convex_hull', 'alpha_shape', 'distance'
    boundary_buffer_distance: Optional[float] = None


@dataclass
class VisualizationConfig:
    """Configuration for visualization parameters."""
    create_interactive_map: bool = True
    include_flight_paths: bool = True
    include_contours: bool = True
    max_flight_path_points: int = 5000
    mapbox_token: Optional[str] = None
    default_opacity: float = 0.7


@dataclass
class OutputConfig:
    """Configuration for output parameters."""
    save_grids_as_geotiff: bool = True
    save_grids_as_csv: bool = True
    save_grids_as_numpy: bool = True
    create_summary_report: bool = True
    output_directory_name: str = "sgtool_auto_results"


@dataclass
class AutoProcessingConfig:
    """Complete automatic processing configuration."""
    magnetic_field: MagneticFieldConfig
    processing: ProcessingConfig
    visualization: VisualizationConfig
    output: OutputConfig
    
    @classmethod
    def create_default(cls) -> 'AutoProcessingConfig':
        """Create default configuration."""
        return cls(
            magnetic_field=MagneticFieldConfig(),
            processing=ProcessingConfig(),
            visualization=VisualizationConfig(),
            output=OutputConfig()
        )
    
    @classmethod
    def from_file(cls, config_path: Union[str, Path]) -> 'AutoProcessingConfig':
        """Load configuration from JSON file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            logger.warning(f"Config file not found: {config_path}, using defaults")
            return cls.create_default()
        
        try:
            with open(config_path, 'r') as f:
                config_dict = json.load(f)
            
            return cls(
                magnetic_field=MagneticFieldConfig(**config_dict.get('magnetic_field', {})),
                processing=ProcessingConfig(**config_dict.get('processing', {})),
                visualization=VisualizationConfig(**config_dict.get('visualization', {})),
                output=OutputConfig(**config_dict.get('output', {}))
            )
        except Exception as e:
            logger.error(f"Failed to load config from {config_path}: {e}")
            return cls.create_default()
    
    def save_to_file(self, config_path: Union[str, Path]) -> None:
        """Save configuration to JSON file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(config_path, 'w') as f:
            json.dump(asdict(self), f, indent=2)
        
        logger.info(f"Configuration saved to: {config_path}")
    
    def get_all_filters(self) -> List[str]:
        """Get list of all available filters."""
        if self.processing.custom_filters:
            return self.processing.custom_filters
        
        if self.processing.enable_all_filters:
            return [
                'rtp',                    # Reduction to Pole
                'rte',                    # Reduction to Equator
                'upward_continuation',    # Upward continuation
                'vertical_integration',   # Vertical integration (pseudogravity)
                'thg',                    # Total Horizontal Gradient
                'analytic_signal',        # Analytic Signal
                'tilt_angle',            # Tilt Angle
                'high_pass',             # High-pass filter
                'low_pass',              # Low-pass filter
                'remove_regional'        # Remove regional trend
            ]
        else:
            # Default essential filters
            return ['rtp', 'thg', 'analytic_signal', 'tilt_angle']


class AutoProcessor:
    """
    Automatic processor that handles complete workflow from CSV directory to maps.
    """
    
    def __init__(self, config: Optional[AutoProcessingConfig] = None):
        """
        Initialize auto processor.
        
        Parameters:
            config (Optional[AutoProcessingConfig]): Processing configuration
        """
        self.config = config or AutoProcessingConfig.create_default()
        self.target_field = None  # Can be set to override magnetic field selection
        
    def detect_magnetic_parameters(self, csv_directory: Union[str, Path]) -> Dict[str, float]:
        """
        Attempt to auto-detect magnetic field parameters from data location.
        
        Parameters:
            csv_directory (Union[str, Path]): Directory containing CSV files
            
        Returns:
            Dict[str, float]: Detected magnetic parameters
        """
        # This is a simplified implementation
        # In practice, you'd use IGRF models based on lat/lon and date
        
        # Default values for northern regions (can be enhanced)
        detected_params = {
            'inclination': 70.0,
            'declination': 2.0
        }
        
        try:
            from ..io.csv_reader import CSVReader
            
            # Try to load sample data to get location info
            reader = CSVReader()
            dataframes = reader.read_csv_directory(csv_directory)
            
            if dataframes:
                combined = reader.combine_dataframes(dataframes)
                
                # Get center coordinates
                center_x = combined['x'].mean()
                center_y = combined['y'].mean()
                
                # Simple heuristic based on coordinates
                if center_x > 1000:  # Likely UTM
                    # For northern hemisphere UTM zones
                    if center_y > 7000000:  # Northern regions
                        detected_params['inclination'] = 75.0
                        detected_params['declination'] = 1.0
                    elif center_y > 5000000:  # Mid-latitude
                        detected_params['inclination'] = 65.0
                        detected_params['declination'] = 3.0
                else:
                    # Geographic coordinates
                    if center_y > 60:  # High latitude
                        detected_params['inclination'] = 80.0
                        detected_params['declination'] = 0.0
                    elif center_y > 45:  # Mid latitude
                        detected_params['inclination'] = 70.0
                        detected_params['declination'] = 2.0
                
                logger.info(f"Auto-detected magnetic parameters: I={detected_params['inclination']:.1f}°, D={detected_params['declination']:.1f}°")
        
        except Exception as e:
            logger.warning(f"Failed to auto-detect magnetic parameters: {e}")
        
        return detected_params
    
    def run_automatic_processing(self, input_directory: Union[str, Path],
                                output_directory: Optional[Union[str, Path]] = None) -> Dict:
        """
        Run complete automatic processing workflow.
        
        Parameters:
            input_directory (Union[str, Path]): Directory containing CSV files
            output_directory (Optional[Union[str, Path]]): Output directory
            
        Returns:
            Dict: Processing results and file paths
        """
        from ..pipeline.batch_processor import BatchProcessor
        
        input_directory = Path(input_directory)
        
        if output_directory is None:
            output_directory = input_directory / self.config.output.output_directory_name
        
        logger.info("="*60)
        logger.info("SGTool Python - Automatic Processing")
        logger.info("="*60)
        logger.info(f"Input directory: {input_directory}")
        logger.info(f"Output directory: {output_directory}")
        
        # Auto-detect magnetic parameters if enabled
        if self.config.magnetic_field.auto_detect:
            detected_params = self.detect_magnetic_parameters(input_directory)
            self.config.magnetic_field.inclination = detected_params['inclination']
            self.config.magnetic_field.declination = detected_params['declination']
        
        # Initialize batch processor
        processor = BatchProcessor(
            input_directory=input_directory,
            output_directory=output_directory,
            grid_resolution=self.config.processing.grid_resolution,
            max_kriging_points=self.config.processing.max_kriging_points
        )
        
        # Set target field if specified
        if self.target_field:
            processor.csv_reader.target_field = self.target_field
        
        # Configure boundary masking
        processor.enable_boundary_masking = self.config.processing.enable_boundary_masking
        processor.boundary_method = self.config.processing.boundary_method
        processor.boundary_buffer_distance = self.config.processing.boundary_buffer_distance
        
        # Set magnetic parameters
        processor.magnetic_params = {
            'inclination': self.config.magnetic_field.inclination,
            'declination': self.config.magnetic_field.declination
        }
        
        # Configure visualization
        if self.config.visualization.mapbox_token:
            processor.interactive_maps.mapbox_token = self.config.visualization.mapbox_token
        
        # Get filters to apply
        filters_to_apply = self.config.get_all_filters()
        
        logger.info(f"Applying {len(filters_to_apply)} filters: {', '.join(filters_to_apply)}")
        logger.info(f"Grid resolution: {self.config.processing.grid_resolution}x{self.config.processing.grid_resolution}")
        logger.info(f"Interpolation: {self.config.processing.interpolation_method}")
        logger.info(f"Magnetic parameters: I={self.config.magnetic_field.inclination:.1f}°, D={self.config.magnetic_field.declination:.1f}°")
        
        # Run processing
        results = processor.run(
            filters=filters_to_apply,
            interpolation_method=self.config.processing.interpolation_method,
            create_map=self.config.visualization.create_interactive_map
        )
        
        # Save configuration used
        config_file = Path(output_directory) / "processing_config.json"
        self.config.save_to_file(config_file)
        
        # Create summary report if enabled
        if self.config.output.create_summary_report:
            self._create_summary_report(results, output_directory)
        
        logger.info("="*60)
        logger.info("AUTOMATIC PROCESSING COMPLETED!")
        logger.info("="*60)
        logger.info(f"Total points processed: {results['total_points']:,}")
        logger.info(f"Filters applied: {len(results['filters_applied'])}")
        logger.info(f"Output directory: {results['output_directory']}")
        
        if results.get('interactive_map'):
            logger.info(f"Interactive map: {results['interactive_map']}")
        
        return results
    
    def _create_summary_report(self, results: Dict, output_directory: Union[str, Path]) -> None:
        """Create a summary report of the processing results."""
        output_directory = Path(output_directory)
        report_file = output_directory / "processing_summary.md"
        
        with open(report_file, 'w') as f:
            f.write("# SGTool Python Processing Summary\n\n")
            f.write(f"**Processing Date:** {results.get('timestamp', 'Unknown')}\n\n")
            
            f.write("## Input Data\n")
            f.write(f"- **Source Directory:** {results['input_directory']}\n")
            f.write(f"- **Total Points:** {results['total_points']:,}\n")
            f.write(f"- **Grid Resolution:** {results['grid_resolution']}×{results['grid_resolution']}\n")
            f.write(f"- **Interpolation Method:** {results['interpolation_method']}\n\n")
            
            f.write("## Processing Parameters\n")
            f.write(f"- **Magnetic Inclination:** {self.config.magnetic_field.inclination:.1f}°\n")
            f.write(f"- **Magnetic Declination:** {self.config.magnetic_field.declination:.1f}°\n")
            f.write(f"- **Grid Spacing:** {results['grid_spacing'][0]:.2f}m × {results['grid_spacing'][1]:.2f}m\n\n")
            
            f.write("## Applied Filters\n")
            for filter_name in results['filters_applied']:
                f.write(f"- {filter_name}\n")
            f.write("\n")
            
            f.write("## Output Files\n")
            for file_type, file_path in results['saved_files'].items():
                if '_csv' in file_type:
                    filter_name = file_type.replace('_csv', '')
                    f.write(f"- **{filter_name}:** {Path(file_path).name}\n")
            
            if results.get('interactive_map'):
                f.write(f"- **Interactive Map:** {Path(results['interactive_map']).name}\n")
            
            f.write("\n## Filter Descriptions\n")
            filter_descriptions = {
                'rtp': 'Reduction to Pole - Transforms magnetic data to what it would look like at the magnetic pole',
                'rte': 'Reduction to Equator - Transforms magnetic data to what it would look like at the magnetic equator',
                'thg': 'Total Horizontal Gradient - Highlights edges and boundaries in magnetic data',
                'analytic_signal': 'Analytic Signal - Provides amplitude of magnetic anomalies independent of field direction',
                'tilt_angle': 'Tilt Angle - Enhances edges and provides structural information',
                'upward_continuation': 'Upward Continuation - Smooths data by simulating measurement at higher altitude',
                'vertical_integration': 'Vertical Integration - Converts magnetic data to pseudogravity',
                'high_pass': 'High-Pass Filter - Removes long-wavelength regional trends',
                'low_pass': 'Low-Pass Filter - Smooths data by removing short-wavelength noise',
                'remove_regional': 'Remove Regional - Removes polynomial regional trends from data'
            }
            
            for filter_name in results['filters_applied']:
                if filter_name in filter_descriptions:
                    f.write(f"### {filter_name}\n{filter_descriptions[filter_name]}\n\n")
        
        logger.info(f"Summary report created: {report_file}")


def create_example_config(output_path: Union[str, Path] = "sgtool_config.json") -> None:
    """Create an example configuration file."""
    config = AutoProcessingConfig.create_default()
    
    # Set some example values
    config.magnetic_field.inclination = 70.0
    config.magnetic_field.declination = 2.0
    config.magnetic_field.auto_detect = True
    
    config.processing.grid_resolution = 300
    config.processing.enable_all_filters = True
    
    config.visualization.create_interactive_map = True
    config.visualization.include_flight_paths = True
    config.visualization.mapbox_token = "your_mapbox_token_here"
    
    config.output.save_grids_as_geotiff = True
    config.output.create_summary_report = True
    
    config.save_to_file(output_path)
    print(f"Example configuration saved to: {output_path}")


if __name__ == '__main__':
    # Create example configuration
    create_example_config()