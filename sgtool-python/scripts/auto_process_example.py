#!/usr/bin/env python3
"""
SGTool Python Auto-Processing Example
====================================

Complete automated processing example that takes a directory of CSVs,
runs all filters, and creates comprehensive interactive maps.
"""

import sys
import logging
from pathlib import Path

# Add the parent directory to the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgtool_py.config.auto_config import AutoProcessor, AutoProcessingConfig


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main auto-processing example."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # =================================================================
    # CONFIGURATION - UPDATE THESE PATHS FOR YOUR DATA
    # =================================================================
    
    # Input directory containing CSV files from magbase/flight_path_segmenter
    INPUT_DIRECTORY = "/Users/aleksandergarbuz/Documents/SINTEF/data/test_data"  # UPDATE THIS
    
    # Output directory (optional - will be created automatically)
    OUTPUT_DIRECTORY = None  # Will create 'sgtool_auto_results' in input directory
    
    # Optional: Path to configuration file (JSON format)
    CONFIG_FILE = None  # Set to path if you have a custom config
    
    # =================================================================
    # PROCESSING CONFIGURATION
    # =================================================================
    
    if CONFIG_FILE and Path(CONFIG_FILE).exists():
        # Load from configuration file
        config = AutoProcessingConfig.from_file(CONFIG_FILE)
        logger.info(f"Loaded configuration from: {CONFIG_FILE}")
    else:
        # Create configuration programmatically
        config = AutoProcessingConfig.create_default()
        
        # Customize magnetic field parameters
        config.magnetic_field.inclination = 70.0     # Magnetic inclination (degrees)
        config.magnetic_field.declination = 2.0      # Magnetic declination (degrees)
        config.magnetic_field.auto_detect = True     # Try to auto-detect from data location
        
        # Customize processing parameters
        config.processing.grid_resolution = 300      # Grid resolution (300x300)
        config.processing.max_kriging_points = 15000 # Max points for kriging
        config.processing.interpolation_method = 'minimum_curvature'  # 'minimum_curvature', 'kriging', 'linear', 'cubic'
        config.processing.enable_all_filters = True  # Apply ALL available filters
        
        # Customize visualization
        config.visualization.create_interactive_map = True
        config.visualization.include_flight_paths = True
        config.visualization.include_contours = True
        config.visualization.max_flight_path_points = 5000
        config.visualization.mapbox_token = None     # Add your Mapbox token for satellite imagery
        
        # Customize output
        config.output.save_grids_as_geotiff = True   # Save as GeoTIFF files
        config.output.save_grids_as_csv = True       # Save as CSV files
        config.output.save_grids_as_numpy = True     # Save as NumPy arrays
        config.output.create_summary_report = True   # Generate summary report
        
        logger.info("Using programmatic configuration")
    
    try:
        # =================================================================
        # RUN AUTOMATIC PROCESSING
        # =================================================================
        
        logger.info("🚀 Starting SGTool Python Auto-Processing")
        logger.info("="*60)
        
        # Check if input directory exists
        if not Path(INPUT_DIRECTORY).exists():
            logger.error(f"❌ Input directory not found: {INPUT_DIRECTORY}")
            logger.info("Please update the INPUT_DIRECTORY path in this script")
            return 1
        
        # Initialize auto processor
        auto_processor = AutoProcessor(config)
        
        # Display what will be processed
        filters = config.get_all_filters()
        logger.info(f"📁 Input directory: {INPUT_DIRECTORY}")
        logger.info(f"🔧 Filters to apply: {', '.join(filters)}")
        logger.info(f"📐 Grid resolution: {config.processing.grid_resolution}x{config.processing.grid_resolution}")
        logger.info(f"🧮 Interpolation: {config.processing.interpolation_method}")
        logger.info(f"🧭 Auto-detect magnetic params: {config.magnetic_field.auto_detect}")
        
        # Run the complete automated workflow
        results = auto_processor.run_automatic_processing(
            input_directory=INPUT_DIRECTORY,
            output_directory=OUTPUT_DIRECTORY
        )
        
        # =================================================================
        # DISPLAY RESULTS
        # =================================================================
        
        logger.info("🎉 SUCCESS! Auto-processing completed!")
        logger.info("="*60)
        logger.info("📊 PROCESSING SUMMARY:")
        logger.info(f"   • Total data points: {results['total_points']:,}")
        logger.info(f"   • Grid resolution: {results['grid_resolution']}×{results['grid_resolution']}")
        logger.info(f"   • Interpolation method: {results['interpolation_method']}")
        logger.info(f"   • Filters applied: {len(results['filters_applied'])}")
        logger.info(f"   • Output directory: {results['output_directory']}")
        
        logger.info("\n📁 GENERATED FILES:")
        # Count different file types
        csv_files = [f for f in results['saved_files'].keys() if '_csv' in f]
        npy_files = [f for f in results['saved_files'].keys() if '_npy' in f]
        
        logger.info(f"   • {len(csv_files)} CSV grid files")
        logger.info(f"   • {len(npy_files)} NumPy array files")
        logger.info("   • 1 Processing configuration file")
        logger.info("   • 1 Summary report (Markdown)")
        
        if results.get('interactive_map'):
            logger.info(f"   • 1 Interactive map: {Path(results['interactive_map']).name}")
        
        logger.info("\n🔧 APPLIED FILTERS:")
        for filter_name in results['filters_applied']:
            logger.info(f"   ✓ {filter_name}")
        
        logger.info("\n🗺️  VISUALIZATION:")
        if results.get('interactive_map'):
            logger.info(f"   • Interactive Folium map created")
            logger.info(f"   • All filter results as toggleable layers")
            logger.info(f"   • Flight path overlay with magnetic field coloring")
            logger.info(f"   • Multiple basemap options")
            logger.info(f"   • Open in browser: {results['interactive_map']}")
        
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("   1. Open the interactive map in your browser")
        logger.info("   2. Toggle between different filter results")
        logger.info("   3. Analyze geological features and anomalies")
        logger.info("   4. Use CSV/GeoTIFF files for further analysis")
        logger.info("   5. Read the summary report for processing details")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Auto-processing failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())