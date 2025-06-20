#!/usr/bin/env python3
"""
SGTool Python Pipeline Runner
============================

Main script for running the complete geophysical processing pipeline.
"""

import sys
import logging
from pathlib import Path

# Add the parent directory to the path so we can import sgtool_py
sys.path.insert(0, str(Path(__file__).parent.parent))

from sgtool_py.pipeline.batch_processor import BatchProcessor


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def main():
    """Main pipeline runner."""
    setup_logging()
    logger = logging.getLogger(__name__)
    
    # Configuration - modify these paths for your data
    INPUT_DIRECTORY = "/Users/aleksandergarbuz/Documents/SINTEF/data/test_data"  # Update this path
    OUTPUT_DIRECTORY = None  # Will use input_dir/sgtool_results
    
    # Processing parameters
    FILTERS = ['rtp', 'thg', 'analytic_signal', 'tilt_angle', 'remove_regional']
    GRID_RESOLUTION = 300
    INTERPOLATION_METHOD = 'minimum_curvature'
    
    # Magnetic field parameters (adjust for your survey area)
    MAGNETIC_INCLINATION = 70.0  # degrees
    MAGNETIC_DECLINATION = 2.0   # degrees
    
    try:
        logger.info("="*60)
        logger.info("SGTool Python Processing Pipeline")
        logger.info("="*60)
        
        # Check if input directory exists
        if not Path(INPUT_DIRECTORY).exists():
            logger.error(f"Input directory not found: {INPUT_DIRECTORY}")
            logger.info("Please update the INPUT_DIRECTORY path in this script")
            return 1
        
        # Initialize processor
        processor = BatchProcessor(
            input_directory=INPUT_DIRECTORY,
            output_directory=OUTPUT_DIRECTORY,
            grid_resolution=GRID_RESOLUTION
        )
        
        # Set magnetic parameters
        processor.magnetic_params = {
            'inclination': MAGNETIC_INCLINATION,
            'declination': MAGNETIC_DECLINATION
        }
        
        # Run processing
        logger.info(f"Input directory: {INPUT_DIRECTORY}")
        logger.info(f"Filters to apply: {', '.join(FILTERS)}")
        logger.info(f"Grid resolution: {GRID_RESOLUTION}x{GRID_RESOLUTION}")
        logger.info(f"Interpolation method: {INTERPOLATION_METHOD}")
        logger.info(f"Magnetic parameters: I={MAGNETIC_INCLINATION}°, D={MAGNETIC_DECLINATION}°")
        
        results = processor.run(
            filters=FILTERS,
            interpolation_method=INTERPOLATION_METHOD,
            create_map=True
        )
        
        # Display results
        logger.info("="*60)
        logger.info("PROCESSING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"Total points processed: {results['total_points']:,}")
        logger.info(f"Grid resolution: {results['grid_resolution']}x{results['grid_resolution']}")
        logger.info(f"Filters applied: {len(results['filters_applied'])}")
        logger.info(f"Output directory: {results['output_directory']}")
        
        if results['interactive_map']:
            logger.info(f"Interactive map created: {results['interactive_map']}")
        
        logger.info(f"Saved {len(results['saved_files'])//2} processed grids")
        
        # List output files
        logger.info("\nOutput files:")
        for name, path in results['saved_files'].items():
            if name.endswith('_csv'):
                logger.info(f"  {name.replace('_csv', '')}: {path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()
        return 1


if __name__ == '__main__':
    sys.exit(main())