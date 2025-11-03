"""
Package initialization for Video Meme Compositor source modules.
"""

# Import main classes for easy access
from .config import Config, load_config
from .frame_extractor import FrameExtractor, FramePair
from .band_processor import BandProcessor, BandGeometry
from .mask_generator import MaskGenerator, MaskSet

__version__ = "1.0.0"
__author__ = "Video Meme Compositor"

__all__ = [
    'Config', 
    'load_config',
    'FrameExtractor', 
    'FramePair',
    'BandProcessor', 
    'BandGeometry',
    'MaskGenerator', 
    'MaskSet'
]