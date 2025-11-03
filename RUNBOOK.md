# Runbook — Video Meme Compositor (band-only upscaling, single-frame inpaint)

This runbook is the concise, actionable reference your local agent/copilot will follow when generating the Python modules. It contains exact transform math, placement rules, order of operations, and the minimal checks needed to produce correct results. Keep this file next to your code so the agent can read it and produce code that honors these rules.

---
Quick summary (one line)
- Upscale only DVD top/bottom bands (integer scale from manual match = 4×), map DVD center sample to canonical output center sample exactly, run AI background inpainting once on the single background key frame, never generatively replace foreground/face — only restore/upscale/recolor/track/compose.

---
Definitions / fixed names
- canonical_center: output pixel (Ox,Oy) = (floor(output_w/2), floor(output_h/2))
- dvd_center: source DVD center pixel (Sx = floor(720/2), Sy = floor(540/2))
- measured_center_sample: your manual overlay measurement -> 960×540 center sample used to derive scale
- band_upscale_factor: 4 (integer), per manual sample mapping (540 → 2160)
- face_asset: `El_xox_sillyface.jpg`
- generated_background_key: the single inpainted background image (saved once per shot)

---
Required assumptions (verify before running)
- You have a representative matched pair (one DVD frame and its corresponding widescreen frame) and the manual visual match that produced 960×540 center sample.
- Output canvas chosen by pipeline (example canonical sizes):
  - Example final canvas: width = 3840, height = as required (e.g., 2160 or 1610). The runbook calculations assume widescreen sampling derived from your manual measurement.
- Tools available: upscaler (Real-ESRGAN / Swin2SR), segmentation (SAM or RMBG), inpainting (Stable Diffusion/Automatic1111), tracking (Optical flow / lk opencv), basic image ops (Pillow/OpenCV), ffmpeg for export.
- VRAM constraints: plan for tiled upscaling of 4× bands (GTX 1050 Mobile 4GB).

---
Exact placement & transform rules (authoritative)

1) Canonical center mapping (pixel-exact)
- Compute canonical_center (Ox,Oy) = (floor(output_w/2), floor(output_h/2)).
- For each source (widescreen, dvd):
  - Compute source_center = (floor(source_w/2), floor(source_h/2)).
  - The source center pixel sample MUST map to (Ox,Oy) exactly.
  - Transform = translate + scale only (no rotation). For integer scale, compute:
    - scaled_source_w = source_w * scale, scaled_source_h = source_h * scale
    - paste_x = Ox - floor(scaled_source_w/2)
    - paste_y = Oy - floor(scaled_source_h/2)
  - If scale is non-integer (avoid if possible), compute affine with float translation so source_center → (Ox,Oy) before rasterizing; preserve subpixel offset until final rasterize.

2) Band geometry extraction (representative frame)
- Given widescreen_downsampled and dvd_frame:
  - Identify vertical extents of widescreen overlap on the DVD (top index T_ws and bottom index B_ws) by center alignment or a quick SSIM/phase-corr search over vertical offsets.
  - DVD extra bands are:
    - top_band_dvd = dvd_frame[0 : Sy - H_top_core, :]
    - core_dvd = dvd_frame[Sy - H_top_core : Sy + H_bottom_core, :]
    - bottom_band_dvd = dvd_frame[Sy + H_bottom_core : end, :]
  - For your manual case: measured core width = 960×540 sample implies band_upscale_factor = 4×.

3) Band upscaling rule
- Only upscale the top_band_dvd and bottom_band_dvd.
- Upscale factor = band_upscale_factor (4).
- After upscaling:
  - up_top_h = top_band_dvd_h * 4
  - up_bottom_h = bottom_band_dvd_h * 4
  - Place upscaled bands so the upscaled dvd_center sample maps to Oy exactly:
    - compute paste coordinates using the scaled band heights and Ox,Oy mapping described earlier.

4) Composite real-data layers (pixel priority)
- For each output pixel, the priority is:
  1. If the widescreen original provides a real sample at that output pixel → use widescreen pixel (subject/detail priority).
  2. Else if an upscaled DVD band provides a real sample at that pixel → use upscaled band pixel.
  3. Else if the single generated background (inpaint) provides a pixel → use generated pixel.
  4. Else fill with neutral background (rare — should not happen after step 6).
- Implement via mask-based compositing (use per-layer alpha generated from availability masks, feather seams).

---
Single-shot inpaint rule (one time only)
- Choose the background key frame: pick the first matched pair where wizard will be overlaid (or best representative).
- Build inpaint_mask_key = (missing_or_padding_pixels) ∧ background_mask (explicitly exclude Foreground and Face masks).
- Run inpainting exactly once with:
  - low denoising (e.g., sd strength 0.2–0.35),
  - prompt constrained to describe only background style (do NOT mention wizard/face),
  - seed optionally fixed for reproducibility.
- Save result as generated_background_key and reuse for whole shot.

---
Segmentation & mask policies
- Run segmentation once per shot on composite real-data image (widescreen + upscaled bands).
- Produce and save:
  - mask_foreground (float [0..1], feathered)
  - mask_face (bool/submask)
  - mask_background = 1 - mask_foreground
- Feather masks 8–20 px. Store both binary and feathered versions.
- Do not re-run inpainting when masks change — inpainting must have excluded foreground/face.

---
Foreground processing (per-frame)
- For each frame:
  1. Propagate mask_foreground and mask_face via tracking (optical flow) to that frame; recalc or refine if drift > threshold (e.g., > 2 px error).
  2. Recolor: apply hue/sat shift to foreground pixels only (use HSV/HSL or blend mode = Color, preserve luminance).
  3. Restore/upsample center/core of subject only by non‑generative upscalers if required; prefer local restoration (GFPGAN for face restoration, if used only as 'restoration').
  4. Apply chromatic aberration to foreground mask region only (split RGB, offset channels by small integer px).
  5. Punch face hole: set alpha = 0 for face_mask region in foreground layer; ensure feather at edges.

---
Face overlay & tracking
- Use the same tracked transform used for mask propagation to position `El_xox_sillyface.jpg` per frame.
- Resize/rotate according to face bounding box derived from mask_face (match scale exactly).
- Composite with face mask alpha (feather edges) and color-match exposure/white balance to surrounding face luminance.

---
Seam blending & validation
- After compositing bands → foreground → face → background:
  - Blend seams where widescreen meets upscaled band with feathered mask and Poisson seam blending (or matched-luminance crossfade).
  - Validate by pixel-diff of the authoritative center sample region: zero or minimal difference expected between widescreen sample and composite in that center area.
  - Validate that mask_foreground pixels in composite are not replaced by generated pixels (diff mask: generated_background_key ∧ mask_foreground must be empty).

---
Export requirements
- Save intermediates:
  - masks/: foreground, face, background (binary + feathered)
  - bands/: up_top.png, up_bottom.png
  - composites/: per-frame RGBA foreground cutouts, per-frame final frames
  - generated/: generated_background_key.png
- MP4 export: H.264 (libx264), CRF ~18–22, pix_fmt yuv420p
- GIF export: create palette and paletteuse; limit GIF fps per config.

---
Performance & VRAM guidance (practical)
- For 4× band upscaling on GTX 1050 (4GB VRAM):
  - Use tiled upscaling: split bands into overlapping tiles, upscale tiles, then re-stitch with seam-aware overlap (8–32 px overlap).
  - Use FP16 models if available; call torch.cuda.empty_cache() between model loads.
  - If tile-upscaling is used, validate tiles for seam continuity and run a mild global blur at seams if necessary.

---
Checks & assertions (run automatically)
- Assert: dvd_center mapped to canonical_center (abs int delta ≤ 0).
- Assert: upscaled_band_heights == band_upscale_factor × original_band_heights (exact integer).
- Assert: generated_background_key has no Foreground pixels modified (verify: diff(mask_foreground, generated_background_key) == 0 where mask_foreground=1).
- Assert: exported MP4 plays and contains the correct number of frames for the requested duration.

---
Minimal diagnostic outputs to save (for debugging)
- debug/align_report.json:
  - matched_pair_indices, measured_offsets, computed scales, paste coordinates
- debug/mask_diff.png: overlay showing any inpainted pixels under the foreground mask (should be empty)
- debug/tile_report.txt: tile upscaling stats (time, memory, last tile overlap correction)

---
What I did
I created this single-page runbook that your local agent/copilot will use as authoritative instructions: exact pixel mapping rules, integer band-only upscaling, single-shot inpainting constraints, mask policies, and the essential checks to verify compliance.

What's next
Your local agent/copilot should read this runbook and generate the module skeletons (frame extraction, band geometry calculator, tiled upscaler, mask generator/propagator, single-shot inpainter orchestrator, per-frame compositor, exporter). The runbook intentionally avoids implementation details so agents can pick appropriate libraries for your hardware constraints.

```