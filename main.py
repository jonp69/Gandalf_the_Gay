#!/usr/bin/env python3
"""
Video Meme Compositor - Main Application

A sophisticated video meme compositor that processes DVD and widescreen video sources
to create enhanced memes with face insertions and AI-generated backgrounds.

This application implements the pipeline described in context.md and RUNBOOK.md:
- Band-only upscaling (4x integer scaling for DVD extra bands)
- Single-frame AI background inpainting  
- Precise center-pixel mapping with integer transforms
- Face insertion with tracking and blending
"""

import os
import sys
import argparse
import logging
import json
from pathlib import Path
from typing import Dict, Any, Optional
import yaml

# Import our modules
from src.config import Config, load_config
from src.frame_extractor import FrameExtractor
from src.band_processor import BandProcessor
from src.mask_generator import MaskGenerator
from src.background_generator import BackgroundGenerator
from src.face_processor import FaceProcessor
from src.compositor import Compositor

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('video_compositor.log')
    ]
)
logger = logging.getLogger(__name__)


class VideoMemeCompositor:
    """Main application class for the Video Meme Compositor."""
    
    def __init__(self, config_path: str):
        """Initialize the compositor with configuration."""
        self.config = load_config(config_path)
        self.setup_directories()
        
        # Initialize components
        self.frame_extractor = FrameExtractor(self.config)
        self.band_processor = BandProcessor(self.config)
        self.mask_generator = MaskGenerator(self.config)
        self.background_generator = BackgroundGenerator(self.config)
        self.face_processor = FaceProcessor(self.config) 
        self.compositor = Compositor(self.config)
        
        logger.info("Video Meme Compositor initialized")
        logger.info(f"Configuration loaded from: {config_path}")
        
    def setup_directories(self):
        """Create output directories if they don't exist."""
        dirs_to_create = [
            self.config.output_dir,
            self.config.debug_dir,
            self.config.output_dir / "masks",
            self.config.output_dir / "bands", 
            self.config.output_dir / "composites",
            self.config.output_dir / "generated",
            self.config.debug_dir
        ]
        
        for dir_path in dirs_to_create:
            dir_path.mkdir(parents=True, exist_ok=True)
            
        logger.info(f"Output directories created: {self.config.output_dir}")
        
    def validate_inputs(self) -> bool:
        """Validate that required input files exist."""
        required_files = []
        
        # Check widescreen source
        if self.config.widescreen_source:
            required_files.append(self.config.widescreen_source)
        else:
            logger.error("No widescreen source specified")
            return False
            
        # Check DVD source  
        if self.config.dvd_source:
            required_files.append(self.config.dvd_source)
        else:
            logger.error("No DVD source specified")
            return False
            
        # Check face asset
        if self.config.face_asset:
            required_files.append(self.config.face_asset)
        else:
            logger.error("No face asset specified")
            return False
        
        missing_files = []
        for file_path in required_files:
            if file_path and not file_path.exists():
                missing_files.append(str(file_path))
                
        if missing_files:
            logger.error(f"Missing required input files: {missing_files}")
            logger.error("Please check your resources.txt or config.yaml file")
            logger.error("TIP: You can place your files anywhere and specify paths in resources.txt")
            return False
            
        # Validate optional reference frame
        if self.config.reference_frame and not self.config.reference_frame.exists():
            logger.warning(f"Reference frame not found: {self.config.reference_frame}")
            logger.warning("Reference frame can be from ANY source (DVD, widescreen, or external)")
            logger.warning("Proceeding without reference frame synchronization")
            self.config.reference_frame = None
            
        logger.info("All required input files validated")
        
        # Log file sources for clarity
        logger.info(f"Widescreen source: {self.config.widescreen_source}")
        logger.info(f"DVD source: {self.config.dvd_source}")
        logger.info(f"Face asset: {self.config.face_asset}")
        if self.config.reference_frame:
            logger.info(f"Reference frame: {self.config.reference_frame}")
        if self.config.aligned_4x3_source:
            logger.info(f"4:3 source: {self.config.aligned_4x3_source}")
            
        return True
        
    def run_pipeline(self):
        """Execute the complete video meme compositor pipeline."""
        logger.info("Starting Video Meme Compositor pipeline")
        
        try:
            # Step 1: Pre-match and synchronization
            logger.info("Step 1: Frame extraction and synchronization")
            matched_pairs = self.frame_extractor.extract_and_match_frames()
            
            if not matched_pairs:
                raise ValueError("No matched frame pairs found")
                
            logger.info(f"Found {len(matched_pairs)} matched frame pairs")
            
            # Step 2: Determine DVD band geometry (representative frame)
            logger.info("Step 2: Calculating DVD band geometry")
            representative_pair = matched_pairs[0]  # Use first pair as representative
            band_geometry = self.band_processor.calculate_band_geometry(representative_pair)
            
            # Step 3: Process each matched pair 
            logger.info("Step 3: Processing frames with band upscaling")
            processed_frames = []
            
            for i, pair in enumerate(matched_pairs):
                logger.info(f"Processing frame pair {i+1}/{len(matched_pairs)}")
                
                # Extract and upscale DVD bands only
                upscaled_bands = self.band_processor.process_frame_pair(pair, band_geometry)
                processed_frames.append(upscaled_bands)
                
            # Step 4: Generate masks for segmentation (once per shot)
            logger.info("Step 4: Generating segmentation masks")
            first_composite = processed_frames[0]['composite']  
            masks = self.mask_generator.generate_masks(first_composite)
            
            # Step 5: Single background generation pass
            logger.info("Step 5: Generating AI background (single frame)")
            background_key = self.background_generator.generate_background(
                processed_frames[0]['composite'], 
                masks
            )
            
            # Step 6-9: Per-frame processing and composition
            logger.info("Step 6-9: Final composition per frame")
            final_frames = []
            
            for i, frame_data in enumerate(processed_frames):
                logger.info(f"Compositing final frame {i+1}/{len(processed_frames)}")
                
                # Propagate masks to current frame
                frame_masks = self.mask_generator.propagate_masks(
                    masks, 
                    frame_data['composite'],
                    i
                )
                
                # Process foreground (recoloring, effects, face preparation)
                processed_fg = self.face_processor.process_foreground(
                    frame_data['composite'],
                    frame_masks,
                    i
                )
                
                # Insert face with tracking
                face_inserted = self.face_processor.insert_face(
                    processed_fg,
                    frame_masks['face'],
                    i
                )
                
                # Final composite assembly
                final_frame = self.compositor.composite_final_frame(
                    background_key,
                    face_inserted, 
                    frame_masks,
                    i
                )
                
                final_frames.append(final_frame)
                
            # Step 10: Export final video and GIF
            logger.info("Step 10: Exporting final video")
            self.compositor.export_video(final_frames)
            self.compositor.export_gif(final_frames)
            
            logger.info("Video Meme Compositor pipeline completed successfully!")
            
        except Exception as e:
            logger.error(f"Pipeline failed: {str(e)}")
            raise
            
    def generate_debug_report(self):
        """Generate debug report with processing statistics."""
        debug_report = {
            "config": self.config.to_dict(),
            "timestamp": str(Path.ctime(Path.now())),
            "input_files": {
                "widescreen_source": str(self.config.widescreen_source),
                "dvd_source": str(self.config.dvd_source), 
                "face_asset": str(self.config.face_asset)
            },
            "pipeline_status": "completed"
        }
        
        debug_file = self.config.debug_dir / "processing_report.json"
        with open(debug_file, 'w') as f:
            json.dump(debug_report, f, indent=2)
            
        logger.info(f"Debug report saved: {debug_file}")


def main():
    """Main entry point for the Video Meme Compositor."""
    parser = argparse.ArgumentParser(
        description="Video Meme Compositor - Create enhanced video memes with AI"
    )
    parser.add_argument(
        "--config", 
        type=str,
        default="config.yaml",
        help="Path to configuration file (default: config.yaml)"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true", 
        help="Only validate inputs without processing"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    args = parser.parse_args()
    
    if args.debug:
        logging.getLogger().setLevel(logging.DEBUG)
        
    try:
        # Initialize compositor
        compositor = VideoMemeCompositor(args.config)
        
        # Validate inputs
        if not compositor.validate_inputs():
            sys.exit(1)
            
        if args.validate_only:
            logger.info("Input validation completed successfully")
            return
            
        # Run the pipeline
        compositor.run_pipeline()
        
        # Generate debug report
        compositor.generate_debug_report()
        
    except KeyboardInterrupt:
        logger.info("Processing interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Fatal error: {str(e)}")
        sys.exit(1)


if __name__ == "__main__":
    main()