"""
Band geometry calculator and processor for Video Meme Compositor.

This module implements the core band-only upscaling logic specified in RUNBOOK.md:
- Calculate DVD band geometry from representative frame
- Extract top/bottom bands while preserving center core
- Apply 4x integer upscaling to bands only
- Precise center-pixel mapping with integer transforms

Based on manual measurement: 960×540 center sample → 4x upscaling factor
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional
from dataclasses import dataclass
import torch
from PIL import Image

from .config import Config
from .frame_extractor import FramePair

logger = logging.getLogger(__name__)


@dataclass
class BandGeometry:
    """Represents the calculated band geometry for DVD processing."""
    # DVD center coordinates
    dvd_center_x: int
    dvd_center_y: int
    
    # Band boundaries (y-coordinates in DVD frame)
    top_band_start: int
    top_band_end: int
    core_start: int
    core_end: int
    bottom_band_start: int
    bottom_band_end: int
    
    # Upscaled dimensions
    upscaled_top_height: int
    upscaled_bottom_height: int
    upscaled_width: int
    
    # Transform parameters for canonical mapping
    canonical_center_x: int
    canonical_center_y: int
    paste_x: int
    paste_y: int


class BandProcessor:
    """Handles DVD band geometry calculation and upscaling processing."""
    
    def __init__(self, config: Config):
        self.config = config
        self.band_geometry = None
        self.upscaler_model = None
        
        # Initialize upscaler
        self._load_upscaler()
        
    def _load_upscaler(self):
        """Load the upscaling model."""
        try:
            if self.config.upscaler_model == "realesrgan":
                from basicsr.archs.rrdbnet_arch import RRDBNet
                from realesrgan import RealESRGANer
                
                # Use Real-ESRGAN 4x model
                model = RRDBNet(
                    num_in_ch=3, num_out_ch=3, num_feat=64, 
                    num_block=23, num_grow_ch=32, scale=4
                )
                
                model_path = self._get_model_path("RealESRGAN_x4plus.pth")
                
                self.upscaler_model = RealESRGANer(
                    scale=4,
                    model_path=str(model_path),
                    model=model,
                    tile=self.config.tile_size,
                    tile_pad=10,
                    pre_pad=0,
                    half=self.config.use_fp16,
                    gpu_id=0 if self.config.device == "cuda" else None
                )
                
                logger.info(f"Loaded Real-ESRGAN upscaler (4x, tile_size={self.config.tile_size})")
                
            else:
                raise ValueError(f"Unsupported upscaler model: {self.config.upscaler_model}")
                
        except Exception as e:
            logger.error(f"Failed to load upscaler: {e}")
            logger.warning("Falling back to basic interpolation upscaling")
            self.upscaler_model = None
            
    def _get_model_path(self, model_name: str) -> Path:
        """Get path to upscaler model file."""
        # Check common model locations
        model_paths = [
            Path("models") / model_name,
            Path.home() / ".cache" / "realesrgan" / model_name,
            Path("weights") / model_name
        ]
        
        for path in model_paths:
            if path.exists():
                return path
                
        # If not found, create models directory
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        logger.warning(f"Model {model_name} not found. Please download it to {models_dir}/")
        logger.warning("Download from: https://github.com/xinntao/Real-ESRGAN/releases")
        
        return models_dir / model_name
        
    def calculate_band_geometry(self, representative_pair: FramePair) -> BandGeometry:
        """
        Calculate DVD band geometry from a representative frame pair.
        
        This determines where the DVD extra bands are located relative to
        the widescreen content, based on the manual measurement in config.
        
        Args:
            representative_pair: Frame pair to analyze
            
        Returns:
            BandGeometry object with calculated parameters
        """
        logger.info("Calculating DVD band geometry")
        
        dvd_frame = representative_pair.dvd_frame
        ws_frame = representative_pair.widescreen_frame
        
        # Get frame dimensions
        dvd_h, dvd_w = dvd_frame.shape[:2]
        ws_h, ws_w = ws_frame.shape[:2]
        
        # Calculate DVD center coordinates
        dvd_center_x = dvd_w // 2
        dvd_center_y = dvd_h // 2
        
        # Get canonical output center
        canonical_center_x, canonical_center_y = self.config.get_canonical_center()
        
        logger.info(f"DVD dimensions: {dvd_w}×{dvd_h}, center: ({dvd_center_x}, {dvd_center_y})")
        logger.info(f"Widescreen dimensions: {ws_w}×{ws_h}")
        logger.info(f"Canonical center: ({canonical_center_x}, {canonical_center_y})")
        
        # Use manual measurement to determine core region
        measured_w, measured_h = self.config.measured_center_sample
        
        # Calculate core region boundaries in DVD frame
        # The core region is what overlaps with widescreen content
        core_half_w = measured_w // 2
        core_half_h = measured_h // 2
        
        # Map measured sample to DVD coordinates
        # Since DVD is 720×540 and measured sample is 960×540,
        # we need to scale the width component
        dvd_core_half_w = int(core_half_w * (dvd_w / measured_w))
        dvd_core_half_h = core_half_h  # Height should match exactly
        
        # Core region boundaries
        core_start_y = dvd_center_y - dvd_core_half_h
        core_end_y = dvd_center_y + dvd_core_half_h
        
        # Band regions (everything outside the core)
        top_band_start = 0
        top_band_end = core_start_y
        bottom_band_start = core_end_y
        bottom_band_end = dvd_h
        
        # Calculate upscaled dimensions
        upscale_factor = self.config.band_upscale_factor
        upscaled_width = dvd_w * upscale_factor
        upscaled_top_height = (top_band_end - top_band_start) * upscale_factor
        upscaled_bottom_height = (bottom_band_end - bottom_band_start) * upscale_factor
        
        # Calculate paste coordinates for canonical center mapping
        # The upscaled DVD center should map to canonical center
        upscaled_dvd_center_x = dvd_center_x * upscale_factor
        upscaled_dvd_center_y = dvd_center_y * upscale_factor
        
        paste_x = canonical_center_x - upscaled_dvd_center_x
        paste_y = canonical_center_y - upscaled_dvd_center_y
        
        geometry = BandGeometry(
            dvd_center_x=dvd_center_x,
            dvd_center_y=dvd_center_y,
            top_band_start=top_band_start,
            top_band_end=top_band_end,
            core_start=core_start_y,
            core_end=core_end_y,
            bottom_band_start=bottom_band_start,
            bottom_band_end=bottom_band_end,
            upscaled_top_height=upscaled_top_height,
            upscaled_bottom_height=upscaled_bottom_height,
            upscaled_width=upscaled_width,
            canonical_center_x=canonical_center_x,
            canonical_center_y=canonical_center_y,
            paste_x=paste_x,
            paste_y=paste_y
        )
        
        self.band_geometry = geometry
        
        logger.info(f"Band geometry calculated:")
        logger.info(f"  Top band: {top_band_start}-{top_band_end} (h={top_band_end-top_band_start})")
        logger.info(f"  Core: {core_start_y}-{core_end_y} (h={core_end_y-core_start_y})")
        logger.info(f"  Bottom band: {bottom_band_start}-{bottom_band_end} (h={bottom_band_end-bottom_band_start})")
        logger.info(f"  Upscaled dimensions: {upscaled_width}×{upscaled_top_height+core_end_y-core_start_y+upscaled_bottom_height}")
        logger.info(f"  Paste position: ({paste_x}, {paste_y})")
        
        return geometry
        
    def process_frame_pair(self, frame_pair: FramePair, geometry: BandGeometry) -> Dict[str, Any]:
        """
        Process a frame pair using the calculated band geometry.
        
        This extracts DVD bands, upscales them, and creates a composite
        with the widescreen content according to pixel priority rules.
        
        Args:
            frame_pair: Frame pair to process
            geometry: Pre-calculated band geometry
            
        Returns:
            Dictionary containing processed frame data
        """
        logger.debug(f"Processing frame pair {frame_pair.frame_index}")
        
        dvd_frame = frame_pair.dvd_frame
        ws_frame = frame_pair.widescreen_frame
        
        # Extract DVD regions
        top_band = dvd_frame[geometry.top_band_start:geometry.top_band_end, :]
        core_region = dvd_frame[geometry.core_start:geometry.core_end, :]
        bottom_band = dvd_frame[geometry.bottom_band_start:geometry.bottom_band_end, :]
        
        # Upscale only the bands (core is preserved at original resolution)
        upscaled_top = self._upscale_region(top_band, geometry.upscaled_width, geometry.upscaled_top_height)
        upscaled_bottom = self._upscale_region(bottom_band, geometry.upscaled_width, geometry.upscaled_bottom_height)
        
        # Create composite following pixel priority rules
        composite = self._create_composite(
            ws_frame, 
            upscaled_top, 
            core_region, 
            upscaled_bottom, 
            geometry
        )
        
        return {
            'composite': composite,
            'upscaled_top': upscaled_top,
            'upscaled_bottom': upscaled_bottom,
            'core_region': core_region,
            'widescreen_frame': ws_frame,
            'dvd_frame': dvd_frame,
            'geometry': geometry,
            'frame_index': frame_pair.frame_index
        }
        
    def _upscale_region(self, region: np.ndarray, target_width: int, target_height: int) -> np.ndarray:
        """
        Upscale a region using the loaded upscaler model.
        
        Args:
            region: Input region to upscale
            target_width: Target width
            target_height: Target height
            
        Returns:
            Upscaled region
        """
        if region.size == 0:
            return np.zeros((target_height, target_width, 3), dtype=np.uint8)
            
        if self.upscaler_model is not None:
            try:
                # Use Real-ESRGAN for upscaling
                upscaled, _ = self.upscaler_model.enhance(region, outscale=self.config.band_upscale_factor)
                
                # Resize to exact target dimensions if needed
                if upscaled.shape[:2] != (target_height, target_width):
                    upscaled = cv2.resize(upscaled, (target_width, target_height), interpolation=cv2.INTER_LANCZOS4)
                    
                return upscaled
                
            except Exception as e:
                logger.warning(f"Real-ESRGAN upscaling failed: {e}, falling back to interpolation")
                
        # Fallback to basic interpolation
        upscaled = cv2.resize(
            region, 
            (target_width, target_height), 
            interpolation=cv2.INTER_LANCZOS4
        )
        
        return upscaled
        
    def _create_composite(self, 
                         ws_frame: np.ndarray, 
                         upscaled_top: np.ndarray, 
                         core_region: np.ndarray, 
                         upscaled_bottom: np.ndarray, 
                         geometry: BandGeometry) -> np.ndarray:
        """
        Create composite frame following pixel priority rules.
        
        Priority order:
        1. Widescreen original (real pixels)
        2. Upscaled DVD bands (real upscaled pixels)
        3. Generated background (filled later)
        
        Args:
            ws_frame: Widescreen frame
            upscaled_top: Upscaled top band
            core_region: DVD core region (not upscaled)
            upscaled_bottom: Upscaled bottom band
            geometry: Band geometry
            
        Returns:
            Composite frame
        """
        # Create output canvas
        canvas = np.zeros(
            (self.config.output_height, self.config.output_width, 3), 
            dtype=np.uint8
        )
        
        # Calculate widescreen placement (center-pixel mapping)
        ws_h, ws_w = ws_frame.shape[:2]
        ws_center_x, ws_center_y = ws_w // 2, ws_h // 2
        
        # Map widescreen center to canonical center
        ws_paste_x = geometry.canonical_center_x - ws_center_x
        ws_paste_y = geometry.canonical_center_y - ws_center_y
        
        # Ensure widescreen fits in canvas
        ws_x1 = max(0, ws_paste_x)
        ws_y1 = max(0, ws_paste_y)
        ws_x2 = min(self.config.output_width, ws_paste_x + ws_w)
        ws_y2 = min(self.config.output_height, ws_paste_y + ws_h)
        
        # Corresponding source coordinates
        src_x1 = ws_x1 - ws_paste_x
        src_y1 = ws_y1 - ws_paste_y
        src_x2 = src_x1 + (ws_x2 - ws_x1)
        src_y2 = src_y1 + (ws_y2 - ws_y1)
        
        # Place widescreen content (highest priority)
        if ws_x2 > ws_x1 and ws_y2 > ws_y1:
            canvas[ws_y1:ws_y2, ws_x1:ws_x2] = ws_frame[src_y1:src_y2, src_x1:src_x2]
            
        # Place upscaled bands where there are gaps
        self._place_upscaled_bands(canvas, upscaled_top, upscaled_bottom, geometry)
        
        return canvas
        
    def _place_upscaled_bands(self, 
                             canvas: np.ndarray, 
                             upscaled_top: np.ndarray, 
                             upscaled_bottom: np.ndarray, 
                             geometry: BandGeometry):
        """
        Place upscaled bands on canvas, avoiding widescreen areas.
        
        Args:
            canvas: Output canvas (modified in place)
            upscaled_top: Upscaled top band
            upscaled_bottom: Upscaled bottom band
            geometry: Band geometry
        """
        # Calculate band placement positions
        upscale_factor = self.config.band_upscale_factor
        
        # Top band placement
        if upscaled_top.size > 0:
            top_h, top_w = upscaled_top.shape[:2]
            
            # Center horizontally, place at calculated Y position
            top_x = geometry.canonical_center_x - top_w // 2
            top_y = geometry.paste_y + geometry.top_band_start * upscale_factor
            
            # Ensure within canvas bounds
            if (top_y >= 0 and top_y + top_h <= self.config.output_height and
                top_x >= 0 and top_x + top_w <= self.config.output_width):
                
                # Only place where canvas is currently black (no widescreen content)
                mask = np.all(canvas[top_y:top_y+top_h, top_x:top_x+top_w] == 0, axis=2)
                canvas[top_y:top_y+top_h, top_x:top_x+top_w][mask] = upscaled_top[mask]
                
        # Bottom band placement
        if upscaled_bottom.size > 0:
            bottom_h, bottom_w = upscaled_bottom.shape[:2]
            
            # Center horizontally, place at calculated Y position
            bottom_x = geometry.canonical_center_x - bottom_w // 2
            bottom_y = geometry.paste_y + geometry.bottom_band_start * upscale_factor
            
            # Ensure within canvas bounds
            if (bottom_y >= 0 and bottom_y + bottom_h <= self.config.output_height and
                bottom_x >= 0 and bottom_x + bottom_w <= self.config.output_width):
                
                # Only place where canvas is currently black (no widescreen content)
                mask = np.all(canvas[bottom_y:bottom_y+bottom_h, bottom_x:bottom_x+bottom_w] == 0, axis=2)
                canvas[bottom_y:bottom_y+bottom_h, bottom_x:bottom_x+bottom_w][mask] = upscaled_bottom[mask]
                
    def validate_geometry(self, geometry: BandGeometry) -> bool:
        """
        Validate calculated band geometry.
        
        Args:
            geometry: Geometry to validate
            
        Returns:
            True if geometry is valid
        """
        # Check that upscale factor matches configuration
        expected_factor = self.config.band_upscale_factor
        
        # Check DVD center mapping
        center_x, center_y = self.config.get_canonical_center()
        if geometry.canonical_center_x != center_x or geometry.canonical_center_y != center_y:
            logger.error("Canonical center mismatch in geometry")
            return False
            
        # Check band boundaries
        if (geometry.top_band_start >= geometry.top_band_end or
            geometry.core_start >= geometry.core_end or
            geometry.bottom_band_start >= geometry.bottom_band_end):
            logger.error("Invalid band boundaries")
            return False
            
        # Check upscaled dimensions
        expected_top_h = (geometry.top_band_end - geometry.top_band_start) * expected_factor
        expected_bottom_h = (geometry.bottom_band_end - geometry.bottom_band_start) * expected_factor
        
        if (geometry.upscaled_top_height != expected_top_h or
            geometry.upscaled_bottom_height != expected_bottom_h):
            logger.error("Upscaled dimensions don't match expected factor")
            return False
            
        logger.info("Band geometry validation passed")
        return True
        
    def save_debug_bands(self, processed_frame: Dict[str, Any]):
        """
        Save debug images of extracted and upscaled bands.
        
        Args:
            processed_frame: Processed frame data
        """
        debug_dir = self.config.debug_dir / "bands"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        frame_idx = processed_frame['frame_index']
        
        # Save upscaled bands
        if processed_frame['upscaled_top'].size > 0:
            top_path = debug_dir / f"frame_{frame_idx:04d}_top_band.jpg"
            cv2.imwrite(str(top_path), processed_frame['upscaled_top'])
            
        if processed_frame['upscaled_bottom'].size > 0:
            bottom_path = debug_dir / f"frame_{frame_idx:04d}_bottom_band.jpg"
            cv2.imwrite(str(bottom_path), processed_frame['upscaled_bottom'])
            
        # Save composite
        composite_path = debug_dir / f"frame_{frame_idx:04d}_composite.jpg"
        cv2.imwrite(str(composite_path), processed_frame['composite'])
        
        logger.debug(f"Debug bands saved for frame {frame_idx}")
        
    def cleanup(self):
        """Clean up resources."""
        if self.upscaler_model is not None:
            # Clear CUDA cache if using GPU
            if self.config.device == "cuda":
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
            self.upscaler_model = None