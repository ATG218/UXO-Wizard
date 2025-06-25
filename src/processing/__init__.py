"""
Processing module for UXO Wizard Desktop Suite
Handles GPR, Magnetic, Gamma, and Multispectral data processing
"""

from .base import BaseProcessor, ProcessingResult, ProcessingError
from .magnetic import MagneticProcessor
from .gpr import GPRProcessor
from .gamma import GammaProcessor
from .multispectral import MultispectralProcessor
from .pipeline import ProcessingPipeline

__all__ = [
    'BaseProcessor',
    'ProcessingResult',
    'ProcessingError',
    'MagneticProcessor',
    'GPRProcessor', 
    'GammaProcessor',
    'MultispectralProcessor',
    'ProcessingPipeline'
] 