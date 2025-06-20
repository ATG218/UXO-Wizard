"""
SGTool Python - Geophysical Processing Toolkit

A Python adaptation of SGTool for processing magnetic survey data.
"""

__version__ = "0.1.0"
__author__ = "Aleksander Garbuz"

from .pipeline.batch_processor import BatchProcessor
from .core.geophysical_processor import GeophysicalProcessor

__all__ = ["BatchProcessor", "GeophysicalProcessor"]