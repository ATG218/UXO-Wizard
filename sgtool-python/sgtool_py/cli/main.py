"""
Command Line Interface for SGTool Python
========================================

Main CLI for geophysical data processing with SGTool algorithms.
"""

import click
import logging
import sys
from pathlib import Path

from ..pipeline.batch_processor import BatchProcessor
from ..core.geophysical_processor import GeophysicalProcessor
from ..core.frequency_filters import FrequencyFilters
from ..core.gradient_filters import GradientFilters
from ..config.auto_config import AutoProcessor, AutoProcessingConfig, create_example_config


def setup_logging(verbose: bool = False):
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


@click.group()
@click.option('--verbose', '-v', is_flag=True, help='Enable verbose logging')
def main(verbose):
    """SGTool Python - Geophysical Processing Toolkit"""
    setup_logging(verbose)


@main.command()
@click.argument('input_directory', type=click.Path(exists=True, path_type=Path))
@click.option('--output-dir', '-o', type=click.Path(path_type=Path), 
              help='Output directory (default: input_dir/sgtool_results)')
@click.option('--filters', '-f', 
              default='rtp,thg,analytic_signal,tilt_angle',
              help='Comma-separated list of filters to apply')
@click.option('--resolution', '-r', default=300, 
              help='Grid resolution for interpolation')
@click.option('--max-kriging-points', default=15000,
              help='Maximum points for kriging interpolation')
@click.option('--interpolation', default='minimum_curvature',
              type=click.Choice(['minimum_curvature', 'kriging', 'linear', 'cubic']),
              help='Interpolation method')
@click.option('--inclination', default=70.0, 
              help='Magnetic inclination in degrees')
@click.option('--declination', default=2.0,
              help='Magnetic declination in degrees')
@click.option('--no-map', is_flag=True,
              help='Skip creating interactive map')
def process_directory(input_directory, output_dir, filters, resolution, 
                     max_kriging_points, interpolation, inclination, 
                     declination, no_map):
    """
    Process a directory of CSV files with SGTool filters.
    
    INPUT_DIRECTORY: Directory containing CSV files from magbase/flight_path_segmenter
    """
    try:
        # Parse filters
        filter_list = [f.strip() for f in filters.split(',')]
        
        # Initialize processor
        processor = BatchProcessor(
            input_directory=input_directory,
            output_directory=output_dir,
            grid_resolution=resolution,
            max_kriging_points=max_kriging_points
        )
        
        # Set magnetic parameters
        processor.magnetic_params = {
            'inclination': inclination,
            'declination': declination
        }
        
        # Run processing
        click.echo(f"Processing directory: {input_directory}")
        click.echo(f"Filters: {', '.join(filter_list)}")
        click.echo(f"Grid resolution: {resolution}x{resolution}")
        
        results = processor.run(
            filters=filter_list,
            interpolation_method=interpolation,
            create_map=not no_map
        )
        
        # Display results
        click.echo("\n" + "="*50)
        click.echo("Processing completed successfully!")
        click.echo("="*50)
        click.echo(f"Total points processed: {results['total_points']:,}")
        click.echo(f"Grid resolution: {results['grid_resolution']}x{results['grid_resolution']}")
        click.echo(f"Interpolation method: {results['interpolation_method']}")
        click.echo(f"Filters applied: {len(results['filters_applied'])}")
        click.echo(f"Output directory: {results['output_directory']}")
        
        if results['interactive_map']:
            click.echo(f"Interactive map: {results['interactive_map']}")
        
        click.echo(f"\nSaved {len(results['saved_files'])//2} result files")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_file', type=click.Path(exists=True, path_type=Path))
@click.option('--filter-name', '-f', required=True,
              type=click.Choice(['rtp', 'rte', 'upward_continuation', 'thg', 
                               'analytic_signal', 'tilt_angle', 'high_pass', 
                               'low_pass', 'remove_regional']),
              help='Filter to apply')
@click.option('--output', '-o', type=click.Path(path_type=Path),
              help='Output file path')
@click.option('--inclination', default=70.0,
              help='Magnetic inclination in degrees (for RTP/RTE)')
@click.option('--declination', default=2.0,
              help='Magnetic declination in degrees (for RTP/RTE)')
@click.option('--dx', default=1.0, help='Grid spacing in x-direction')
@click.option('--dy', default=1.0, help='Grid spacing in y-direction')
def filter(input_file, filter_name, output, inclination, declination, dx, dy):
    """
    Apply a single SGTool filter to gridded data.
    
    INPUT_FILE: Numpy (.npy) file containing 2D grid data
    """
    try:
        import numpy as np
        
        # Load data
        click.echo(f"Loading data from: {input_file}")
        data = np.load(input_file)
        
        if data.ndim != 2:
            raise ValueError("Input file must contain 2D grid data")
        
        # Initialize appropriate processor
        if filter_name in ['rtp', 'rte', 'upward_continuation', 'remove_regional']:
            processor = GeophysicalProcessor(dx, dy)
        elif filter_name in ['high_pass', 'low_pass']:
            processor = FrequencyFilters(dx, dy)
        elif filter_name in ['thg', 'analytic_signal', 'tilt_angle']:
            processor = GradientFilters(dx, dy)
        
        # Apply filter
        click.echo(f"Applying filter: {filter_name}")
        
        if filter_name == 'rtp':
            result = processor.reduction_to_pole(data, inclination, declination)
        elif filter_name == 'rte':
            result = processor.reduction_to_equator(data, inclination, declination)
        elif filter_name == 'upward_continuation':
            result = processor.upward_continuation(data, 100.0)
        elif filter_name == 'thg':
            result = processor.total_horizontal_gradient(data)
        elif filter_name == 'analytic_signal':
            result = processor.analytic_signal(data)
        elif filter_name == 'tilt_angle':
            result = processor.tilt_angle_degrees(data)
        elif filter_name == 'high_pass':
            wavelength = min(dx, dy) * 50
            result = processor.high_pass_filter(data, wavelength)
        elif filter_name == 'low_pass':
            wavelength = min(dx, dy) * 10
            result = processor.low_pass_filter(data, wavelength)
        elif filter_name == 'remove_regional':
            result = processor.remove_regional_trend(data, order=1)
        
        # Save result
        if output is None:
            output = input_file.parent / f"{input_file.stem}_{filter_name}.npy"
        
        np.save(output, result)
        click.echo(f"Result saved to: {output}")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('input_directory', type=click.Path(exists=True, path_type=Path))
def info(input_directory):
    """
    Display information about CSV files in directory.
    """
    try:
        from ..io.csv_reader import CSVReader
        
        reader = CSVReader()
        click.echo(f"Analyzing directory: {input_directory}")
        
        # Read files
        dataframes = reader.read_csv_directory(input_directory)
        
        if not dataframes:
            click.echo("No valid CSV files found.")
            return
        
        # Get summary
        combined = reader.combine_dataframes(dataframes)
        summary = reader.get_data_summary(combined)
        
        # Display info
        click.echo("\n" + "="*50)
        click.echo("Dataset Information")
        click.echo("="*50)
        click.echo(f"Total files: {summary['unique_files']}")
        click.echo(f"Total points: {summary['total_points']:,}")
        click.echo(f"Coordinate system: {summary['coordinate_system']}")
        click.echo(f"X range: {summary['x_range'][0]:.2f} to {summary['x_range'][1]:.2f}")
        click.echo(f"Y range: {summary['y_range'][0]:.2f} to {summary['y_range'][1]:.2f}")
        click.echo(f"Magnetic field range: {summary['magnetic_field_range'][0]:.2f} to {summary['magnetic_field_range'][1]:.2f} nT")
        
        # Statistics
        stats = summary['magnetic_field_stats']
        click.echo(f"\nMagnetic Field Statistics:")
        click.echo(f"  Mean: {stats['mean']:.2f} nT")
        click.echo(f"  Std:  {stats['std']:.2f} nT")
        click.echo(f"  Min:  {stats['min']:.2f} nT")
        click.echo(f"  Max:  {stats['max']:.2f} nT")
        
    except Exception as e:
        click.echo(f"Error: {e}", err=True)
        sys.exit(1)


@main.command()
def list_filters():
    """List available SGTool filters."""
    filters = {
        'Frequency Domain': [
            'rtp - Reduction to Pole',
            'rte - Reduction to Equator', 
            'upward_continuation - Upward continuation',
            'high_pass - High-pass filter',
            'low_pass - Low-pass filter'
        ],
        'Gradient Based': [
            'thg - Total Horizontal Gradient',
            'analytic_signal - Analytic Signal',
            'tilt_angle - Tilt Angle (degrees)'
        ],
        'Processing': [
            'remove_regional - Remove regional trend'
        ]
    }
    
    click.echo("Available SGTool Filters:")
    click.echo("=" * 40)
    
    for category, filter_list in filters.items():
        click.echo(f"\n{category}:")
        for f in filter_list:
            click.echo(f"  {f}")


@main.command()
@click.argument('input_directory', type=click.Path(exists=True, path_type=Path))
@click.option('--config', '-c', type=click.Path(path_type=Path),
              help='Configuration file (JSON format)')
@click.option('--output-dir', '-o', type=click.Path(path_type=Path),
              help='Output directory (default: input_dir/sgtool_auto_results)')
@click.option('--target-field', '-t', type=str,
              help='Specific magnetic field to use (e.g., "R1 [nT]", "R2 [nT]", "Btotal1 [nT]")')
def auto_process(input_directory, config, output_dir, target_field):
    """
    Automatically process CSV directory with all filters and create maps.
    
    This command runs the complete automated workflow:
    - Auto-detects magnetic field parameters
    - Applies all SGTool filters
    - Creates interactive maps
    - Generates summary report
    
    INPUT_DIRECTORY: Directory containing CSV files from magbase/flight_path_segmenter
    """
    try:
        # Load configuration
        if config:
            auto_config = AutoProcessingConfig.from_file(config)
            click.echo(f"Using configuration from: {config}")
        else:
            auto_config = AutoProcessingConfig.create_default()
            click.echo("Using default configuration")
        
        # Override output directory if specified
        if output_dir:
            auto_config.output.output_directory_name = output_dir.name
        
        # Override target field if specified
        if target_field:
            # We'll need to pass this to the CSV reader
            click.echo(f"Using specified magnetic field: {target_field}")
        
        # Initialize auto processor
        auto_processor = AutoProcessor(auto_config)
        
        # Set target field if specified
        if target_field:
            auto_processor.target_field = target_field
        
        # Run automatic processing
        click.echo(f"Starting automatic processing of: {input_directory}")
        click.echo("This will apply ALL available SGTool filters and create interactive maps...")
        
        results = auto_processor.run_automatic_processing(
            input_directory=input_directory,
            output_directory=output_dir
        )
        
        # Display results
        click.echo("\n" + "="*60)
        click.echo("🎉 AUTOMATIC PROCESSING COMPLETED! 🎉")
        click.echo("="*60)
        click.echo(f"📊 Total points processed: {results['total_points']:,}")
        click.echo(f"🔧 Filters applied: {len(results['filters_applied'])}")
        click.echo(f"📁 Output directory: {results['output_directory']}")
        
        if results.get('interactive_map'):
            click.echo(f"🗺️  Interactive map: {results['interactive_map']}")
        
        click.echo(f"\n✅ Generated {len(results['saved_files'])//2} processed grids")
        click.echo("✅ Created comprehensive summary report")
        click.echo("✅ All files ready for analysis!")
        
    except Exception as e:
        click.echo(f"❌ Error: {e}", err=True)
        sys.exit(1)


@main.command()
@click.argument('output_path', type=click.Path(path_type=Path), 
                default='sgtool_config.json')
def create_config(output_path):
    """
    Create an example configuration file for auto-processing.
    
    OUTPUT_PATH: Path where to save the configuration file (default: sgtool_config.json)
    """
    try:
        create_example_config(output_path)
        click.echo(f"✅ Example configuration created: {output_path}")
        click.echo("\nEdit the configuration file to customize:")
        click.echo("  - Magnetic field parameters")
        click.echo("  - Processing options")
        click.echo("  - Visualization settings")
        click.echo("  - Output formats")
        click.echo(f"\nThen run: sgtool-py auto-process /path/to/data --config {output_path}")
        
    except Exception as e:
        click.echo(f"❌ Error creating config: {e}", err=True)
        sys.exit(1)


if __name__ == '__main__':
    main()