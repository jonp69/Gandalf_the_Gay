"""
Configuration management for Video Meme Compositor.

This module handles loading and validation of configuration parameters
according to the specifications in RUNBOOK.md and context.md.
"""

import os
import re
from pathlib import Path
from typing import Dict, Any, Optional, Union
import yaml
import configparser
from dataclasses import dataclass, field


def parse_smart_timestamp(timestamp_str: str) -> float:
    """
    Smart timestamp parser that handles multiple formats:
    
    Supported formats:
    - Seconds only: "45.5", "123", "45"
    - MM:SS: "1:23.5", "2:15", "0:45.2"  
    - HH:MM:SS: "1:23:45.5", "0:02:15", "2:01:30"
    - Mixed: "1h23m45.5s", "2m15s", "45s"
    
    Args:
        timestamp_str: Timestamp string in any supported format
        
    Returns:
        Timestamp in seconds as float
    """
    timestamp_str = str(timestamp_str).strip()
    
    # Handle pure numeric values (seconds)
    try:
        return float(timestamp_str)
    except ValueError:
        pass
    
    # Handle HH:MM:SS or MM:SS formats
    if ':' in timestamp_str:
        parts = timestamp_str.split(':')
        
        if len(parts) == 2:  # MM:SS
            minutes = float(parts[0])
            seconds = float(parts[1])
            return minutes * 60 + seconds
            
        elif len(parts) == 3:  # HH:MM:SS
            hours = float(parts[0])
            minutes = float(parts[1])
            seconds = float(parts[2])
            return hours * 3600 + minutes * 60 + seconds
    
    # Handle text formats like "1h23m45.5s", "2m15s", "45s"
    # Extract hours, minutes, seconds using regex
    hour_match = re.search(r'(\d+(?:\.\d+)?)h', timestamp_str.lower())
    min_match = re.search(r'(\d+(?:\.\d+)?)m', timestamp_str.lower())
    sec_match = re.search(r'(\d+(?:\.\d+)?)s', timestamp_str.lower())
    
    total_seconds = 0.0
    
    if hour_match:
        total_seconds += float(hour_match.group(1)) * 3600
    if min_match:
        total_seconds += float(min_match.group(1)) * 60
    if sec_match:
        total_seconds += float(sec_match.group(1))
    
    if total_seconds > 0:
        return total_seconds
    
    # If nothing worked, raise an error with helpful message
    raise ValueError(
        f"Could not parse timestamp '{timestamp_str}'. "
        f"Supported formats: '45.5' (seconds), '1:23.5' (MM:SS), "
        f"'1:23:45' (HH:MM:SS), '1h23m45s' (text format)"
    )


@dataclass
class Config:
    """Configuration class for Video Meme Compositor."""
    
    # Input files
    widescreen_source: Optional[Path] = None
    dvd_source: Optional[Path] = None
    face_asset: Path = Path("El_xox_sillyface.jpg")
    reference_frame: Optional[Path] = None
    aligned_4x3_source: Optional[Path] = None
    
    # Output directories
    output_dir: Path = Path("output")
    debug_dir: Path = Path("debug")
    
    # Processing parameters
    start_time: float = 0.0  # seconds
    duration: float = 10.0   # seconds
    output_width: int = 3840
    output_height: int = 2160
    
    # Reference frame timestamp hints for faster sync
    widescreen_reference_time: Optional[float] = None  # seconds
    dvd_reference_time: Optional[float] = None  # seconds
    
    # Band upscaling (from manual measurement: 540 -> 2160)
    band_upscale_factor: int = 4
    measured_center_sample: tuple = (960, 540)
    
    # AI model settings
    upscaler_model: str = "realesrgan"
    segmentation_model: str = "sam"
    inpainting_model: str = "stable-diffusion"
    
    # Processing options
    use_fp16: bool = True
    tile_size: int = 512
    mask_feather_radius: int = 12
    
    # Export settings
    mp4_crf: int = 20
    gif_fps: int = 15
    gif_max_width: int = 800
    
    # Device settings
    device: str = "cuda"  # or "cpu"
    max_vram_gb: float = 4.0  # GTX 1050 Mobile constraint
    
    def __post_init__(self):
        """Post-initialization validation and path conversion."""
        # Convert string paths to Path objects
        if isinstance(self.widescreen_source, str):
            self.widescreen_source = Path(self.widescreen_source)
        if isinstance(self.dvd_source, str):
            self.dvd_source = Path(self.dvd_source)
        if isinstance(self.face_asset, str):
            self.face_asset = Path(self.face_asset)
        if isinstance(self.reference_frame, str):
            self.reference_frame = Path(self.reference_frame)
        if isinstance(self.aligned_4x3_source, str):
            self.aligned_4x3_source = Path(self.aligned_4x3_source)
        if isinstance(self.output_dir, str):
            self.output_dir = Path(self.output_dir)
        if isinstance(self.debug_dir, str):
            self.debug_dir = Path(self.debug_dir)
            
        # Validate critical parameters
        self.validate()
        
    def validate(self):
        """Validate configuration parameters."""
        # Check band upscale factor matches manual measurement
        expected_height = self.measured_center_sample[1] * self.band_upscale_factor
        if expected_height != 2160:
            raise ValueError(
                f"Band upscale factor {self.band_upscale_factor} with center sample "
                f"{self.measured_center_sample} does not produce expected height 2160. "
                f"Got: {expected_height}"
            )
            
        # Validate output dimensions
        if self.output_width <= 0 or self.output_height <= 0:
            raise ValueError("Output dimensions must be positive")
            
        # Check device availability
        if self.device == "cuda":
            try:
                import torch
                if not torch.cuda.is_available():
                    self.device = "cpu"
                    print("Warning: CUDA not available, falling back to CPU")
            except ImportError:
                self.device = "cpu"
                print("Warning: PyTorch not installed, using CPU")
                
    def get_canonical_center(self) -> tuple:
        """Get the canonical output center pixel coordinates."""
        return (self.output_width // 2, self.output_height // 2)
        
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary for serialization."""
        return {
            "widescreen_source": str(self.widescreen_source) if self.widescreen_source else None,
            "dvd_source": str(self.dvd_source) if self.dvd_source else None,
            "face_asset": str(self.face_asset),
            "reference_frame": str(self.reference_frame) if self.reference_frame else None,
            "aligned_4x3_source": str(self.aligned_4x3_source) if self.aligned_4x3_source else None,
            "output_dir": str(self.output_dir),
            "debug_dir": str(self.debug_dir),
            "start_time": self.start_time,
            "duration": self.duration,
            "output_width": self.output_width,
            "output_height": self.output_height,
            "band_upscale_factor": self.band_upscale_factor,
            "measured_center_sample": self.measured_center_sample,
            "upscaler_model": self.upscaler_model,
            "segmentation_model": self.segmentation_model,
            "inpainting_model": self.inpainting_model,
            "use_fp16": self.use_fp16,
            "tile_size": self.tile_size,
            "mask_feather_radius": self.mask_feather_radius,
            "mp4_crf": self.mp4_crf,
            "gif_fps": self.gif_fps,
            "gif_max_width": self.gif_max_width,
            "device": self.device,
            "max_vram_gb": self.max_vram_gb
        }


def load_config(config_path: str) -> Config:
    """Load configuration from YAML file and resources.txt file."""
    config_file = Path(config_path)
    
    # First try to load from resources.txt
    resources_file = Path("resources.txt")
    if resources_file.exists():
        print(f"Loading configuration from resources.txt")
        return load_config_from_resources("resources.txt", config_path)
    
    # Fall back to YAML config
    if not config_file.exists():
        # Create default config file
        default_config = Config()
        save_config(default_config, config_path)
        print(f"Created default configuration file: {config_path}")
        print("Please edit the configuration file with your input paths and run again.")
        print("TIP: You can also create a resources.txt file for easier file management.")
        return default_config
        
    with open(config_file, 'r') as f:
        config_dict = yaml.safe_load(f)
        
    # Create config object from loaded data
    config = Config(**config_dict)
    return config


def load_config_from_resources(resources_path: str, config_path: str) -> Config:
    """Load configuration from resources.txt file."""
    resources = configparser.ConfigParser()
    resources.read(resources_path)
    
    # Extract paths from resources file
    config_dict = {}
    
    # Video sources
    if 'VIDEO_SOURCES' in resources:
        video_section = resources['VIDEO_SOURCES']
        if 'widescreen_source' in video_section:
            config_dict['widescreen_source'] = video_section['widescreen_source']
        if 'dvd_source' in video_section:
            config_dict['dvd_source'] = video_section['dvd_source']
        if 'aligned_4x3_source' in video_section:
            config_dict['aligned_4x3_source'] = video_section['aligned_4x3_source']
    
    # Face assets
    if 'FACE_ASSETS' in resources:
        face_section = resources['FACE_ASSETS']
        if 'face_asset' in face_section:
            config_dict['face_asset'] = face_section['face_asset']
    
    # Reference frames
    if 'REFERENCE_FRAMES' in resources:
        ref_section = resources['REFERENCE_FRAMES']
        if 'reference_frame' in ref_section:
            config_dict['reference_frame'] = ref_section['reference_frame']
        # Add timestamp hints for faster sync (with smart parsing)
        if 'widescreen_reference_time' in ref_section:
            try:
                config_dict['widescreen_reference_time'] = parse_smart_timestamp(ref_section['widescreen_reference_time'])
                print(f"Parsed widescreen timestamp: {ref_section['widescreen_reference_time']} → {config_dict['widescreen_reference_time']:.1f}s")
            except ValueError as e:
                print(f"Warning: Could not parse widescreen_reference_time '{ref_section['widescreen_reference_time']}': {e}")
        if 'dvd_reference_time' in ref_section:
            try:
                config_dict['dvd_reference_time'] = parse_smart_timestamp(ref_section['dvd_reference_time'])
                print(f"Parsed DVD timestamp: {ref_section['dvd_reference_time']} → {config_dict['dvd_reference_time']:.1f}s")
            except ValueError as e:
                print(f"Warning: Could not parse dvd_reference_time '{ref_section['dvd_reference_time']}': {e}")
    
    # Output directories
    if 'OUTPUT' in resources:
        output_section = resources['OUTPUT']
        if 'output_directory' in output_section:
            config_dict['output_dir'] = output_section['output_directory']
        if 'debug_directory' in output_section:
            config_dict['debug_dir'] = output_section['debug_directory']
    
    # Load default config first, then override with resources
    default_config = Config()
    
    # Merge with any existing YAML config
    yaml_config_file = Path(config_path)
    if yaml_config_file.exists():
        with open(yaml_config_file, 'r') as f:
            yaml_dict = yaml.safe_load(f) or {}
        # YAML config takes precedence over defaults, resources.txt takes precedence over YAML
        for key, value in yaml_dict.items():
            if key not in config_dict:  # Only use YAML if not specified in resources
                config_dict[key] = value
    
    # Apply to default config
    for key, value in config_dict.items():
        if hasattr(default_config, key):
            setattr(default_config, key, value)
    
    # Validate and convert paths in post_init
    config = Config(**config_dict)
    return config


def save_config(config: Config, config_path: str):
    """Save configuration to YAML file."""
    config_dict = config.to_dict()
    
    with open(config_path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def create_example_config() -> str:
    """Create an example configuration file."""
    example_config = """# Video Meme Compositor Configuration
# Edit these paths to match your input files

# Required input files
widescreen_source: "path/to/widescreen_video.mp4"  # High-detail source (e.g., 3840x1610)
dvd_source: "path/to/dvd_video.mp4"               # DVD source (720x540) with extra bands
face_asset: "El_xox_sillyface.jpg"                # Face insertion asset

# Optional inputs
reference_frame: null                              # External reference frame for sync
aligned_4x3_source: null                          # 4:3 source for cross-checks

# Output settings
output_dir: "output"
debug_dir: "debug"

# Processing parameters
start_time: 0.0      # Start time in seconds
duration: 10.0       # Duration to process in seconds
output_width: 3840   # Final output width
output_height: 2160  # Final output height

# Band processing (from manual measurement)
band_upscale_factor: 4           # Integer scale factor for DVD bands
measured_center_sample: [960, 540]  # Manual measurement results

# AI model settings
upscaler_model: "realesrgan"     # Real-ESRGAN for band upscaling
segmentation_model: "sam"        # Segment Anything Model
inpainting_model: "stable-diffusion"  # For background generation

# Performance settings
use_fp16: true       # Use FP16 for VRAM efficiency
tile_size: 512       # Tile size for upscaling (VRAM constraint)
device: "cuda"       # "cuda" or "cpu"
max_vram_gb: 4.0     # GTX 1050 Mobile constraint

# Mask settings
mask_feather_radius: 12  # Feathering radius in pixels (8-20)

# Export settings
mp4_crf: 20          # H.264 quality (18-22 recommended)
gif_fps: 15          # GIF frame rate
gif_max_width: 800   # GIF width limit
"""
    
    return example_config


if __name__ == "__main__":
    # Generate example config when run directly
    example = create_example_config()
    with open("config_example.yaml", "w") as f:
        f.write(example)
    print("Example configuration saved to config_example.yaml")