"""
Mask generation and segmentation system for Video Meme Compositor.

This module handles:
- Foreground/background segmentation using SAM or similar models
- Face detection and masking
- Mask propagation across frames using optical flow
- Mask refinement and feathering

Based on RUNBOOK.md specifications for single-shot mask generation
with per-frame propagation for stability.
"""

import cv2
import numpy as np
import logging
from pathlib import Path
from typing import Dict, Tuple, Any, Optional, List
from dataclasses import dataclass
import torch
from PIL import Image

from .config import Config

logger = logging.getLogger(__name__)


@dataclass
class MaskSet:
    """Container for all masks generated for a frame."""
    foreground: np.ndarray  # Float mask [0..1] for wizard/subject
    face: np.ndarray        # Binary mask for face region
    background: np.ndarray  # Float mask [0..1] for background (1 - foreground)
    
    # Feathered versions
    foreground_feathered: np.ndarray
    face_feathered: np.ndarray
    background_feathered: np.ndarray
    
    # Metadata
    frame_index: int
    confidence: float
    

class MaskGenerator:
    """Handles segmentation and mask generation for video frames."""
    
    def __init__(self, config: Config):
        self.config = config
        self.segmentation_model = None
        self.face_detector = None
        self.reference_masks = None
        
        # Optical flow tracker for mask propagation
        self.flow_tracker = None
        self.previous_frame = None
        
        # Initialize models
        self._load_segmentation_model()
        self._load_face_detector()
        
    def _load_segmentation_model(self):
        """Load the segmentation model (SAM or alternative)."""
        try:
            if self.config.segmentation_model == "sam":
                # Try to load Segment Anything Model
                try:
                    from segment_anything import sam_model_registry, SamPredictor
                    
                    model_path = self._get_sam_model_path()
                    if model_path.exists():
                        sam = sam_model_registry["vit_h"](checkpoint=str(model_path))
                        sam.to(device=self.config.device)
                        self.segmentation_model = SamPredictor(sam)
                        logger.info("Loaded SAM model for segmentation")
                    else:
                        raise FileNotFoundError("SAM model not found")
                        
                except ImportError:
                    logger.warning("SAM not available, trying U2Net")
                    self._load_u2net()
                    
            elif self.config.segmentation_model == "u2net":
                self._load_u2net()
            else:
                raise ValueError(f"Unsupported segmentation model: {self.config.segmentation_model}")
                
        except Exception as e:
            logger.error(f"Failed to load segmentation model: {e}")
            logger.warning("Falling back to basic color-based segmentation")
            self.segmentation_model = None
            
    def _load_u2net(self):
        """Load U2Net as fallback segmentation model."""
        try:
            from rembg import BackgroundRemover
            self.segmentation_model = BackgroundRemover(model_name='u2net')
            logger.info("Loaded U2Net model for segmentation")
        except ImportError:
            logger.warning("U2Net/rembg not available")
            self.segmentation_model = None
            
    def _get_sam_model_path(self) -> Path:
        """Get path to SAM model checkpoint."""
        model_paths = [
            Path("models/sam_vit_h_4b8939.pth"),
            Path.home() / ".cache" / "sam" / "sam_vit_h_4b8939.pth",
            Path("weights/sam_vit_h_4b8939.pth")
        ]
        
        for path in model_paths:
            if path.exists():
                return path
                
        # Create models directory and inform user
        models_dir = Path("models")
        models_dir.mkdir(exist_ok=True)
        
        logger.warning("SAM model not found. Please download sam_vit_h_4b8939.pth to models/")
        logger.warning("Download from: https://github.com/facebookresearch/segment-anything")
        
        return models_dir / "sam_vit_h_4b8939.pth"
        
    def _load_face_detector(self):
        """Load face detection model."""
        try:
            # Use OpenCV's DNN face detector
            prototxt_path = Path("models/deploy.prototxt")
            model_path = Path("models/res10_300x300_ssd_iter_140000.caffemodel")
            
            if prototxt_path.exists() and model_path.exists():
                self.face_detector = cv2.dnn.readNetFromCaffe(
                    str(prototxt_path), 
                    str(model_path)
                )
                logger.info("Loaded OpenCV DNN face detector")
            else:
                # Fall back to Haar cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_detector = cv2.CascadeClassifier(cascade_path)
                logger.info("Loaded Haar cascade face detector")
                
        except Exception as e:
            logger.error(f"Failed to load face detector: {e}")
            self.face_detector = None
            
    def generate_masks(self, composite_frame: np.ndarray) -> MaskSet:
        """
        Generate initial masks for the shot using the composite frame.
        
        This is run once per shot to establish the reference masks.
        
        Args:
            composite_frame: Composite frame (real widescreen + upscaled bands)
            
        Returns:
            MaskSet containing all generated masks
        """
        logger.info("Generating initial segmentation masks")
        
        frame_h, frame_w = composite_frame.shape[:2]
        
        # Generate foreground mask
        foreground_mask = self._generate_foreground_mask(composite_frame)
        
        # Generate face mask (subset of foreground)
        face_mask = self._generate_face_mask(composite_frame, foreground_mask)
        
        # Background mask is inverse of foreground
        background_mask = 1.0 - foreground_mask
        
        # Apply feathering
        foreground_feathered = self._feather_mask(foreground_mask)
        face_feathered = self._feather_mask(face_mask.astype(np.float32))
        background_feathered = self._feather_mask(background_mask)
        
        mask_set = MaskSet(
            foreground=foreground_mask,
            face=face_mask,
            background=background_mask,
            foreground_feathered=foreground_feathered,
            face_feathered=face_feathered,
            background_feathered=background_feathered,
            frame_index=0,
            confidence=self._calculate_mask_confidence(foreground_mask)
        )
        
        # Store as reference for propagation
        self.reference_masks = mask_set
        self.previous_frame = composite_frame.copy()
        
        # Save debug masks
        self._save_debug_masks(mask_set, composite_frame)
        
        logger.info(f"Generated masks with confidence: {mask_set.confidence:.3f}")
        return mask_set
        
    def _generate_foreground_mask(self, frame: np.ndarray) -> np.ndarray:
        """
        Generate foreground mask using the loaded segmentation model.
        
        Args:
            frame: Input frame
            
        Returns:
            Float foreground mask [0..1]
        """
        if self.segmentation_model is None:
            return self._generate_foreground_mask_fallback(frame)
            
        try:
            if hasattr(self.segmentation_model, 'predict'):  # SAM
                return self._generate_sam_mask(frame)
            elif hasattr(self.segmentation_model, 'remove'):  # U2Net/rembg
                return self._generate_u2net_mask(frame)
            else:
                return self._generate_foreground_mask_fallback(frame)
                
        except Exception as e:
            logger.warning(f"Segmentation model failed: {e}, using fallback")
            return self._generate_foreground_mask_fallback(frame)
            
    def _generate_sam_mask(self, frame: np.ndarray) -> np.ndarray:
        """Generate mask using SAM with automatic point prompts."""
        # Convert BGR to RGB for SAM
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Set image for SAM
        self.segmentation_model.set_image(rgb_frame)
        
        # Use center point as positive prompt
        h, w = frame.shape[:2]
        center_point = np.array([[w//2, h//2]])
        center_label = np.array([1])
        
        # Generate masks
        masks, scores, logits = self.segmentation_model.predict(
            point_coords=center_point,
            point_labels=center_label,
            multimask_output=True
        )
        
        # Select best mask
        best_idx = np.argmax(scores)
        mask = masks[best_idx].astype(np.float32)
        
        return mask
        
    def _generate_u2net_mask(self, frame: np.ndarray) -> np.ndarray:
        """Generate mask using U2Net/rembg."""
        # Convert to PIL Image
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Remove background
        result = self.segmentation_model.remove(pil_image)
        
        # Extract alpha channel as mask
        if result.mode == 'RGBA':
            mask = np.array(result)[:, :, 3].astype(np.float32) / 255.0
        else:
            # Convert to grayscale and threshold
            gray = np.array(result.convert('L')).astype(np.float32) / 255.0
            mask = (gray > 0.1).astype(np.float32)
            
        return mask
        
    def _generate_foreground_mask_fallback(self, frame: np.ndarray) -> np.ndarray:
        """
        Fallback foreground mask generation using color-based methods.
        
        This uses basic computer vision techniques when advanced models aren't available.
        """
        logger.info("Using fallback color-based segmentation")
        
        # Convert to HSV for better color segmentation
        hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        
        # Create mask based on color range (assumes subject has distinct colors)
        # This is a very basic approach - in practice you'd tune these ranges
        lower_bound = np.array([0, 30, 30])
        upper_bound = np.array([180, 255, 255])
        
        color_mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # Morphological operations to clean up mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_CLOSE, kernel)
        color_mask = cv2.morphologyEx(color_mask, cv2.MORPH_OPEN, kernel)
        
        # Find largest contour (assume it's the subject)
        contours, _ = cv2.findContours(color_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if contours:
            largest_contour = max(contours, key=cv2.contourArea)
            mask = np.zeros(color_mask.shape, dtype=np.uint8)
            cv2.fillPoly(mask, [largest_contour], 255)
        else:
            # If no contours found, create a central ellipse as fallback
            h, w = frame.shape[:2]
            mask = np.zeros((h, w), dtype=np.uint8)
            cv2.ellipse(mask, (w//2, h//2), (w//4, h//3), 0, 0, 360, 255, -1)
            
        return mask.astype(np.float32) / 255.0
        
    def _generate_face_mask(self, frame: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
        """
        Generate face mask within the foreground region.
        
        Args:
            frame: Input frame
            foreground_mask: Foreground mask to constrain face detection
            
        Returns:
            Binary face mask
        """
        if self.face_detector is None:
            return self._generate_face_mask_fallback(foreground_mask)
            
        try:
            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            if hasattr(self.face_detector, 'setInput'):  # DNN detector
                return self._detect_faces_dnn(frame, foreground_mask)
            else:  # Haar cascade
                return self._detect_faces_haar(gray, foreground_mask)
                
        except Exception as e:
            logger.warning(f"Face detection failed: {e}, using fallback")
            return self._generate_face_mask_fallback(foreground_mask)
            
    def _detect_faces_dnn(self, frame: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
        """Detect faces using DNN model."""
        h, w = frame.shape[:2]
        
        # Create blob from image
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104, 117, 123])
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        face_mask = np.zeros((h, w), dtype=np.uint8)
        
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            
            if confidence > 0.5:  # Confidence threshold
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                x, y, x1, y1 = box.astype(int)
                
                # Expand face region slightly
                margin = 20
                x = max(0, x - margin)
                y = max(0, y - margin)
                x1 = min(w, x1 + margin)
                y1 = min(h, y1 + margin)
                
                # Only include if within foreground
                face_region = foreground_mask[y:y1, x:x1]
                if np.mean(face_region) > 0.5:  # Mostly within foreground
                    cv2.rectangle(face_mask, (x, y), (x1, y1), 255, -1)
                    
        return face_mask > 0
        
    def _detect_faces_haar(self, gray: np.ndarray, foreground_mask: np.ndarray) -> np.ndarray:
        """Detect faces using Haar cascade."""
        faces = self.face_detector.detectMultiScale(
            gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
        )
        
        h, w = gray.shape
        face_mask = np.zeros((h, w), dtype=np.uint8)
        
        for (x, y, fw, fh) in faces:
            # Expand face region slightly
            margin = 10
            x = max(0, x - margin)
            y = max(0, y - margin)
            fw = min(w - x, fw + 2 * margin)
            fh = min(h - y, fh + 2 * margin)
            
            # Only include if within foreground
            face_region = foreground_mask[y:y+fh, x:x+fw]
            if np.mean(face_region) > 0.5:  # Mostly within foreground
                cv2.rectangle(face_mask, (x, y), (x + fw, y + fh), 255, -1)
                
        return face_mask > 0
        
    def _generate_face_mask_fallback(self, foreground_mask: np.ndarray) -> np.ndarray:
        """Generate face mask fallback using foreground mask analysis."""
        # Assume face is in upper portion of foreground
        h, w = foreground_mask.shape
        
        # Find foreground centroid
        moments = cv2.moments(foreground_mask.astype(np.uint8))
        if moments['m00'] > 0:
            cx = int(moments['m10'] / moments['m00'])
            cy = int(moments['m01'] / moments['m00'])
            
            # Face region is typically in upper 1/3 of subject
            face_y = max(0, cy - h//6)
            face_size = min(w//8, h//8)
            
            face_mask = np.zeros((h, w), dtype=np.uint8)
            cv2.circle(face_mask, (cx, face_y), face_size, 255, -1)
            
            # Intersect with foreground
            face_mask = face_mask & (foreground_mask > 0.5).astype(np.uint8) * 255
            
        else:
            face_mask = np.zeros((h, w), dtype=np.uint8)
            
        return face_mask > 0
        
    def _feather_mask(self, mask: np.ndarray) -> np.ndarray:
        """
        Apply Gaussian feathering to mask edges.
        
        Args:
            mask: Input mask
            
        Returns:
            Feathered mask
        """
        radius = self.config.mask_feather_radius
        
        if radius <= 0:
            return mask
            
        # Apply Gaussian blur for feathering
        feathered = cv2.GaussianBlur(mask, (radius * 2 + 1, radius * 2 + 1), radius / 3.0)
        
        return feathered
        
    def propagate_masks(self, reference_masks: MaskSet, current_frame: np.ndarray, frame_index: int) -> MaskSet:
        """
        Propagate masks to current frame using optical flow.
        
        Args:
            reference_masks: Reference masks from initial generation
            current_frame: Current frame to propagate masks to
            frame_index: Current frame index
            
        Returns:
            Propagated mask set
        """
        if self.previous_frame is None or frame_index == 0:
            # Return reference masks for first frame
            reference_masks.frame_index = frame_index
            return reference_masks
            
        logger.debug(f"Propagating masks to frame {frame_index}")
        
        # Calculate optical flow
        flow = self._calculate_optical_flow(self.previous_frame, current_frame)
        
        # Propagate each mask
        propagated_fg = self._warp_mask_with_flow(reference_masks.foreground, flow)
        propagated_face = self._warp_mask_with_flow(reference_masks.face.astype(np.float32), flow)
        propagated_bg = 1.0 - propagated_fg
        
        # Apply feathering
        fg_feathered = self._feather_mask(propagated_fg)
        face_feathered = self._feather_mask(propagated_face)
        bg_feathered = self._feather_mask(propagated_bg)
        
        # Calculate confidence (how well the masks propagated)
        confidence = self._calculate_propagation_confidence(flow)
        
        propagated_masks = MaskSet(
            foreground=propagated_fg,
            face=propagated_face > 0.5,  # Binarize face mask
            background=propagated_bg,
            foreground_feathered=fg_feathered,
            face_feathered=face_feathered,
            background_feathered=bg_feathered,
            frame_index=frame_index,
            confidence=confidence
        )
        
        # Update previous frame
        self.previous_frame = current_frame.copy()
        
        # Re-segment if confidence is too low
        if confidence < 0.3:
            logger.warning(f"Low propagation confidence ({confidence:.3f}), re-segmenting")
            return self.generate_masks(current_frame)
            
        return propagated_masks
        
    def _calculate_optical_flow(self, prev_frame: np.ndarray, curr_frame: np.ndarray) -> np.ndarray:
        """Calculate dense optical flow between frames."""
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.cvtColor(curr_frame, cv2.COLOR_BGR2GRAY)
        
        flow = cv2.calcOpticalFlowPyrLK(
            prev_gray, curr_gray, None, None,
            winSize=(21, 21),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        
        # Use Farneback for dense flow as fallback
        if flow is None:
            flow = cv2.calcOpticalFlowPyrLK(prev_gray, curr_gray, 0.5, 3, 15, 3, 5, 1.2, 0)
            
        return flow
        
    def _warp_mask_with_flow(self, mask: np.ndarray, flow: np.ndarray) -> np.ndarray:
        """Warp mask using optical flow."""
        h, w = mask.shape[:2]
        
        # Create coordinate grids
        y, x = np.mgrid[0:h, 0:w]
        
        # Apply flow (simplified - in practice you'd use proper flow warping)
        # For now, just return the original mask
        # TODO: Implement proper optical flow warping
        
        return mask
        
    def _calculate_mask_confidence(self, mask: np.ndarray) -> float:
        """Calculate confidence score for generated mask."""
        # Simple heuristics for mask quality
        mask_area = np.sum(mask)
        total_area = mask.size
        
        area_ratio = mask_area / total_area
        
        # Good masks should cover reasonable area (not too small or too large)
        if 0.1 <= area_ratio <= 0.6:
            confidence = 0.8
        else:
            confidence = 0.5
            
        return confidence
        
    def _calculate_propagation_confidence(self, flow: np.ndarray) -> float:
        """Calculate confidence for mask propagation based on optical flow."""
        # Simplified confidence based on flow magnitude
        # In practice, you'd analyze flow consistency and mask warping quality
        return 0.7  # Placeholder
        
    def _save_debug_masks(self, mask_set: MaskSet, frame: np.ndarray):
        """Save debug visualizations of masks."""
        masks_dir = self.config.debug_dir / "masks"
        masks_dir.mkdir(parents=True, exist_ok=True)
        
        frame_idx = mask_set.frame_index
        
        # Save individual masks
        cv2.imwrite(str(masks_dir / f"frame_{frame_idx:04d}_foreground.jpg"), 
                   (mask_set.foreground * 255).astype(np.uint8))
        cv2.imwrite(str(masks_dir / f"frame_{frame_idx:04d}_face.jpg"), 
                   (mask_set.face.astype(np.uint8) * 255))
        cv2.imwrite(str(masks_dir / f"frame_{frame_idx:04d}_background.jpg"), 
                   (mask_set.background * 255).astype(np.uint8))
        
        # Save overlay visualization
        overlay = frame.copy()
        
        # Red tint for foreground
        fg_mask_3ch = np.stack([mask_set.foreground] * 3, axis=2)
        overlay = overlay * (1 - fg_mask_3ch * 0.3) + np.array([0, 0, 255]) * fg_mask_3ch * 0.3
        
        # Blue outline for face
        face_contours, _ = cv2.findContours(
            mask_set.face.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        cv2.drawContours(overlay, face_contours, -1, (255, 0, 0), 2)
        
        cv2.imwrite(str(masks_dir / f"frame_{frame_idx:04d}_overlay.jpg"), overlay.astype(np.uint8))
        
        logger.debug(f"Debug masks saved for frame {frame_idx}")
        
    def cleanup(self):
        """Clean up resources."""
        if self.segmentation_model is not None:
            if hasattr(self.segmentation_model, 'model') and hasattr(self.segmentation_model.model, 'cuda'):
                try:
                    torch.cuda.empty_cache()
                except:
                    pass