"""
Frame extraction and synchronization module for Video Meme Compositor.

This module handles:
1. Pre-match synchronization using downscaled/cropped copies
2. Frame matching across the requested duration
3. Temporal alignment between widescreen and DVD sources

Based on RUNBOOK.md specifications for exact frame matching and sync.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional
from dataclasses import dataclass
import imageio

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class FramePair:
    """Represents a matched pair of frames from different sources."""
    widescreen_frame: np.ndarray
    dvd_frame: np.ndarray
    widescreen_timestamp: float
    dvd_timestamp: float
    frame_index: int


class FrameExtractor:
    """Handles frame extraction and synchronization between video sources."""
    
    def __init__(self, config: Config):
        self.config = config
        self.widescreen_reader = None
        self.dvd_reader = None
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.cleanup()
        
    def cleanup(self):
        """Clean up video readers."""
        if self.widescreen_reader is not None:
            self.widescreen_reader.close()
        if self.dvd_reader is not None:
            self.dvd_reader.close()
            
    def extract_and_match_frames(self) -> List[FramePair]:
        """
        Extract and match frames from both video sources.
        
        Returns:
            List of FramePair objects representing synchronized frames
        """
        logger.info("Starting frame extraction and matching")
        
        # Open video sources
        self._open_video_sources()
        
        # Step 1: Pre-match using reference frame if available
        start_offset = self._determine_start_offset()
        
        # Step 2: Extract frame pairs for the requested duration
        matched_pairs = self._extract_frame_pairs(start_offset)
        
        logger.info(f"Extracted {len(matched_pairs)} matched frame pairs")
        return matched_pairs
        
    def _open_video_sources(self):
        """Open video readers for both sources."""
        try:
            self.widescreen_reader = imageio.get_reader(
                str(self.config.widescreen_source), 
                'ffmpeg'
            )
            self.dvd_reader = imageio.get_reader(
                str(self.config.dvd_source), 
                'ffmpeg'
            )
            
            # Get video metadata
            ws_meta = self.widescreen_reader.get_meta_data()
            dvd_meta = self.dvd_reader.get_meta_data()
            
            logger.info(f"Widescreen: {ws_meta.get('size', 'unknown')} @ {ws_meta.get('fps', 'unknown')} fps")
            logger.info(f"DVD: {dvd_meta.get('size', 'unknown')} @ {dvd_meta.get('fps', 'unknown')} fps")
            
        except Exception as e:
            logger.error(f"Failed to open video sources: {e}")
            raise
            
    def _determine_start_offset(self) -> float:
        """
        Determine the start time offset between sources.
        
        Uses reference frame if available, otherwise assumes synchronized start.
        
        Returns:
            Time offset in seconds
        """
        if self.config.reference_frame and self.config.reference_frame.exists():
            logger.info("Using reference frame for synchronization")
            return self._sync_with_reference_frame()
        else:
            logger.info("No reference frame provided, assuming synchronized start")
            return self.config.start_time
            
    def _sync_with_reference_frame(self) -> float:
        """
        Synchronize using external reference frame.
        
        The reference frame can be from ANY source:
        - Extracted from either video source
        - From a completely different video  
        - A screenshot from any 720×540 or similar source
        - Any image that helps establish timing alignment
        
        Returns:
            Best match timestamp offset
        """
        # Load reference frame
        ref_frame = cv2.imread(str(self.config.reference_frame))
        if ref_frame is None:
            logger.warning("Could not load reference frame, using default start time")
            return self.config.start_time
            
        logger.info(f"Using reference frame: {self.config.reference_frame}")
        logger.info("Reference frame can be from any source - DVD, widescreen, or external")
        
        # Try to match reference frame against both video sources
        widescreen_match_time = self._find_best_match_in_video(ref_frame, self.widescreen_reader, "widescreen")
        dvd_match_time = self._find_best_match_in_video(ref_frame, self.dvd_reader, "DVD")
        
        # Use the best match from either source
        if widescreen_match_time is not None and dvd_match_time is not None:
            # Both found matches - use the one with higher confidence
            # For now, prefer DVD match since reference is likely from DVD/720×540 source
            logger.info(f"Found matches in both sources - DVD: {dvd_match_time:.2f}s, Widescreen: {widescreen_match_time:.2f}s")
            return dvd_match_time
        elif dvd_match_time is not None:
            logger.info(f"Found reference frame match in DVD source at {dvd_match_time:.2f}s")
            return dvd_match_time
        elif widescreen_match_time is not None:
            logger.info(f"Found reference frame match in widescreen source at {widescreen_match_time:.2f}s")
            return widescreen_match_time
        else:
            logger.warning("No good matches found for reference frame, using default start time")
            return self.config.start_time
            
    def _find_best_match_in_video(self, ref_frame: np.ndarray, video_reader, source_name: str) -> Optional[float]:
        """
        Find the best matching frame in a video source.
        
        Args:
            ref_frame: Reference frame to match
            video_reader: Video reader object
            source_name: Name for logging
            
        Returns:
            Timestamp of best match, or None if no good match found
        """
        logger.info(f"Searching for reference frame match in {source_name} source")
        
        try:
            video_meta = video_reader.get_meta_data()
            fps = video_meta.get('fps', 30.0)
            duration = video_meta.get('duration', 60.0)
            
            # Sample frames every 0.5 seconds to find the best match
            sample_interval = 0.5
            best_match_score = 0
            best_match_time = None
            
            # Prepare reference frame for matching
            ref_gray = cv2.cvtColor(ref_frame, cv2.COLOR_BGR2GRAY)
            ref_resized = cv2.resize(ref_gray, (320, 240))  # Small size for fast matching
            
            for t in np.arange(0, min(duration, 120), sample_interval):  # Search first 2 minutes
                try:
                    frame_idx = int(t * fps)
                    if frame_idx >= len(video_reader):
                        break
                        
                    frame = video_reader.get_data(frame_idx)
                    
                    # Convert to grayscale and resize for matching
                    if len(frame.shape) == 3:
                        frame_gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
                    else:
                        frame_gray = frame
                        
                    frame_resized = cv2.resize(frame_gray, (320, 240))
                    
                    # Calculate similarity using template matching
                    result = cv2.matchTemplate(frame_resized, ref_resized, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, _ = cv2.minMaxLoc(result)
                    
                    if max_val > best_match_score:
                        best_match_score = max_val
                        best_match_time = t
                        
                except Exception as e:
                    logger.debug(f"Error matching frame at {t:.2f}s: {e}")
                    continue
            
            # Only return match if confidence is high enough
            if best_match_score > 0.6:  # Threshold for good match
                logger.info(f"Best match in {source_name}: {best_match_time:.2f}s (confidence: {best_match_score:.3f})")
                return best_match_time
            else:
                logger.info(f"No confident match found in {source_name} (best: {best_match_score:.3f})")
                return None
                
        except Exception as e:
            logger.warning(f"Error searching {source_name} for reference frame: {e}")
            return None
        
    def _extract_frame_pairs(self, start_offset: float) -> List[FramePair]:
        """
        Extract synchronized frame pairs for the requested duration.
        
        Args:
            start_offset: Starting time offset in seconds
            
        Returns:
            List of matched frame pairs
        """
        matched_pairs = []
        
        # Get video metadata for frame rate calculation
        ws_meta = self.widescreen_reader.get_meta_data()
        dvd_meta = self.dvd_reader.get_meta_data()
        
        ws_fps = ws_meta.get('fps', 30.0)
        dvd_fps = dvd_meta.get('fps', 30.0)
        
        # Calculate frame indices for start and end
        ws_start_frame = int(start_offset * ws_fps)
        dvd_start_frame = int(start_offset * dvd_fps)
        
        ws_end_frame = int((start_offset + self.config.duration) * ws_fps)
        dvd_end_frame = int((start_offset + self.config.duration) * dvd_fps)
        
        logger.info(f"Widescreen frames: {ws_start_frame} to {ws_end_frame}")
        logger.info(f"DVD frames: {dvd_start_frame} to {dvd_end_frame}")
        
        # Extract frames with temporal matching
        frame_index = 0
        ws_frame_idx = ws_start_frame
        dvd_frame_idx = dvd_start_frame
        
        while (ws_frame_idx < ws_end_frame and 
               dvd_frame_idx < dvd_end_frame and
               ws_frame_idx < len(self.widescreen_reader) and
               dvd_frame_idx < len(self.dvd_reader)):
            
            try:
                # Extract frames
                ws_frame = self.widescreen_reader.get_data(ws_frame_idx)
                dvd_frame = self.dvd_reader.get_data(dvd_frame_idx)
                
                # Convert to RGB if needed (imageio returns RGB by default)
                if len(ws_frame.shape) == 3 and ws_frame.shape[2] == 3:
                    ws_frame = cv2.cvtColor(ws_frame, cv2.COLOR_RGB2BGR)
                if len(dvd_frame.shape) == 3 and dvd_frame.shape[2] == 3:
                    dvd_frame = cv2.cvtColor(dvd_frame, cv2.COLOR_RGB2BGR)
                
                # Calculate timestamps
                ws_timestamp = ws_frame_idx / ws_fps
                dvd_timestamp = dvd_frame_idx / dvd_fps
                
                # Create frame pair
                pair = FramePair(
                    widescreen_frame=ws_frame,
                    dvd_frame=dvd_frame,
                    widescreen_timestamp=ws_timestamp,
                    dvd_timestamp=dvd_timestamp,
                    frame_index=frame_index
                )
                
                matched_pairs.append(pair)
                
                # Advance frame indices (simple 1:1 matching for now)
                # TODO: Implement more sophisticated temporal matching if needed
                ws_frame_idx += 1
                dvd_frame_idx += 1
                frame_index += 1
                
                if frame_index % 30 == 0:  # Log every 30 frames (~1 second)
                    logger.debug(f"Extracted {frame_index} frame pairs")
                    
            except Exception as e:
                logger.warning(f"Error extracting frame pair at indices {ws_frame_idx}/{dvd_frame_idx}: {e}")
                break
                
        logger.info(f"Successfully extracted {len(matched_pairs)} frame pairs")
        return matched_pairs
        
    def create_sync_test_crops(self, frame_pair: FramePair) -> Tuple[np.ndarray, np.ndarray]:
        """
        Create downscaled and cropped versions of frames for synchronization testing.
        
        This creates temporary crops that contain only real overlapping pixels
        for phase correlation or SSIM-based matching.
        
        Args:
            frame_pair: Frame pair to process
            
        Returns:
            Tuple of (cropped_widescreen, cropped_dvd)
        """
        ws_frame = frame_pair.widescreen_frame
        dvd_frame = frame_pair.dvd_frame
        
        # Get frame dimensions
        ws_h, ws_w = ws_frame.shape[:2]
        dvd_h, dvd_w = dvd_frame.shape[:2]
        
        # Calculate center regions that should overlap
        # This is a simplified version - in practice you'd use the band geometry
        # calculated from the measured center sample
        
        # For sync testing, crop to a central region that should be present in both
        crop_ratio = 0.6  # Use central 60% for matching
        
        # Widescreen crop
        ws_crop_h = int(ws_h * crop_ratio)
        ws_crop_w = int(ws_w * crop_ratio)
        ws_y = (ws_h - ws_crop_h) // 2
        ws_x = (ws_w - ws_crop_w) // 2
        ws_crop = ws_frame[ws_y:ws_y+ws_crop_h, ws_x:ws_x+ws_crop_w]
        
        # DVD crop (map to equivalent region)
        dvd_crop_h = int(dvd_h * crop_ratio)
        dvd_crop_w = int(dvd_w * crop_ratio)
        dvd_y = (dvd_h - dvd_crop_h) // 2
        dvd_x = (dvd_w - dvd_crop_w) // 2
        dvd_crop = dvd_frame[dvd_y:dvd_y+dvd_crop_h, dvd_x:dvd_x+dvd_crop_w]
        
        # Resize to common size for comparison
        target_size = (480, 360)  # Small size for fast comparison
        ws_crop_resized = cv2.resize(ws_crop, target_size)
        dvd_crop_resized = cv2.resize(dvd_crop, target_size)
        
        return ws_crop_resized, dvd_crop_resized
        
    def validate_frame_quality(self, frame_pair: FramePair) -> bool:
        """
        Validate that extracted frames are of sufficient quality.
        
        Args:
            frame_pair: Frame pair to validate
            
        Returns:
            True if frames pass quality checks
        """
        # Check for minimum dimensions
        ws_h, ws_w = frame_pair.widescreen_frame.shape[:2]
        dvd_h, dvd_w = frame_pair.dvd_frame.shape[:2]
        
        if ws_h < 100 or ws_w < 100 or dvd_h < 100 or dvd_w < 100:
            logger.warning(f"Frame {frame_pair.frame_index}: Dimensions too small")
            return False
            
        # Check for completely black frames
        ws_mean = np.mean(frame_pair.widescreen_frame)
        dvd_mean = np.mean(frame_pair.dvd_frame)
        
        if ws_mean < 5 or dvd_mean < 5:
            logger.warning(f"Frame {frame_pair.frame_index}: Frame appears to be black")
            return False
            
        return True
        
    def save_debug_frames(self, matched_pairs: List[FramePair], save_count: int = 5):
        """
        Save debug frames for visual inspection.
        
        Args:
            matched_pairs: List of frame pairs
            save_count: Number of frame pairs to save
        """
        debug_dir = self.config.debug_dir / "frames"
        debug_dir.mkdir(parents=True, exist_ok=True)
        
        step = max(1, len(matched_pairs) // save_count)
        
        for i in range(0, len(matched_pairs), step):
            if i >= save_count:
                break
                
            pair = matched_pairs[i]
            
            # Save widescreen frame
            ws_path = debug_dir / f"frame_{i:04d}_widescreen.jpg"
            cv2.imwrite(str(ws_path), pair.widescreen_frame)
            
            # Save DVD frame
            dvd_path = debug_dir / f"frame_{i:04d}_dvd.jpg"
            cv2.imwrite(str(dvd_path), pair.dvd_frame)
            
            # Save sync test crops
            ws_crop, dvd_crop = self.create_sync_test_crops(pair)
            crop_ws_path = debug_dir / f"frame_{i:04d}_sync_ws.jpg"
            crop_dvd_path = debug_dir / f"frame_{i:04d}_sync_dvd.jpg"
            cv2.imwrite(str(crop_ws_path), ws_crop)
            cv2.imwrite(str(crop_dvd_path), dvd_crop)
            
        logger.info(f"Debug frames saved to {debug_dir}")