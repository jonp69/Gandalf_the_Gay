# Context — Video Meme Compositor (finalized)

This file captures the final, actionable context for the local agent/desktop pipeline. It reflects the decisions you made interactively:
- only the DVD top/bottom bands are upscaled (center region preserved);
- AI generation/inpainting runs exactly once on a single background frame (the one used for wizard overlay);
- center‑pixel DATA mapping is authoritative (not merely center coordinates);
- masks drive all operations and foreground/face pixels are never generatively replaced.

---

## Inputs
- Widescreen source (high-detail; sample 3840×1610 in your tests)
- DVD source (720×540) — contains extra vertical detail above and below widescreen crop
- Aligned 4:3 source (if available; used for cross-checks)
- Single external reference frame (used once to set start time / center)
- Face insertion asset: `El_xox_sillyface.jpg`

---

## High-level rules (short)
- Downscale + aggressive crop is used only for matching/synchronization.
- For final compositing use full‑resolution originals; do not discard real pixels.
- Map center pixel DATA (the actual sample) from each source to the canonical output center pixel exactly — prefer integer scale factors so mapping is exact.
- DVD generative replacement of the wizard/face is forbidden. Allowed operations on subject: restoration/upscale, recolor (masked), masked effects, tracked compositing.
- AI inpainting/generation is permitted only once, on a single background frame, and only on areas explicitly outside Foreground/Face masks.

---

## Final pipeline (concise, ordered)

1) Pre‑match (sync-only)
   - Create temporary downscaled/cropped copies of the widescreen and DVD frames that contain only real overlapping pixels (aggressive cropping to intersection region).
   - Use the single external reference frame once to set the start timestamp/initial alignment.

2) Frame matching loop
   - Match frames A↔B across the requested duration using the temporary downscaled/cropped copies.
   - Stop when either stream reaches the end timestamp.

3) Determine DVD band geometry (representative frame)
   - For one representative matched pair, compute the vertical bands in the DVD that extend beyond the widescreen crop (top and bottom extra-detail bands).
   - Using your manual visual measurement (center sample region measured as 960×540 when overlaying), the chosen integer upscaling factor is 4× for the extra bands (720×540 → 2880×2160 so DVD 540→2160 to match the measured widescreen sampling).

4) Per matched pair (use full‑res originals)
   - Map center pixel DATA of each source to the canonical output center pixel precisely (compute translation and scale so the source center sample lands on the output center sample).
   - Extract three DVD regions: center core, top band, bottom band (center core preserved; bands are the only regions to be upscaled).
   - Upscale only the top & bottom bands by the chosen integer factor (4× per your manual match). Use a restoration upscaler; prefer FP16 and tiled processing if VRAM limited.
   - Composite the upscaled bands and the widescreen full‑res data onto the output canvas so each location uses real pixels if available (center uses widescreen, bands use upscaled DVD detail).

5) Masks & segmentation (per shot)
   - Run segmentation once for the shot to obtain: Background mask, Foreground (wizard) mask, Face mask (submask).
   - Refine/feather masks (8–20 px); save them and reuse for all frames.
   - For per-frame stability, propagate masks via optical flow / feature tracking; re-run segmentation only if masks drift.

6) Background: single generation pass
   - Select the single background frame (first or best representative) on which the wizard will be overlaid.
   - Build an inpaint mask = (missing/padding/corners) ∧ Background_mask. Ensure Foreground/Face are excluded.
   - Run AI inpainting exactly once on that background image (low denoise; prompts tuned to match lighting/palette).
   - Save the inpainted background image; reuse it for every frame.

7) Foreground per frame
   - For each frame:
     - Recolor the foreground via the Foreground mask (wizard → pink) using hue/blend methods that preserve luminance & texture.
     - Apply masked stylistic effects (chromatic aberration on wardrobe only; exclude face if desired).
     - Punch and restore face details using the supplied face mask/example (sharpening, GFPGAN restoration optional — restoration only).
     - Track foreground motion and apply transforms to foreground cutouts so enhancements and face insertion follow motion.

8) Face insertion per frame
   - Use tracked transforms to stitch `El_xox_sillyface.jpg` into the punched face hole each frame. Use the face mask alpha, feather edges, color match and blend for plausible integration.

9) Final composite per frame
   - Composite order: Single inpainted background → Recolored foreground (with upscaled bands & effects) → Face insertion → global unify (subtle grain, overall color tie).
   - Export frames / assemble MP4 + GIF. Keep intermediates (masks, upscaled bands, face cutouts).

---

## Key numeric/placement details (authoritative)
- Manual visual match: widescreen center sampling measured ≈ 960×540.
- From your measurement: (540 / 960) * 3840 = 2160 → DVD vertical span maps to 2160 pixels on widescreen sampling.
- Therefore the integer scale chosen for the DVD bands = 4× (540 × 4 = 2160). Resulting upscaled band sizes: DVD 720×540 → 2880×2160.
- Place upscaled bands so the DVD center sample maps exactly to the canonical output center pixel (no fractional offsets). Use pixel‑accurate translation and integer scaling to avoid subpixel aliasing.

---

## Practical constraints & performance notes
- Upscaling only bands reduces compute vs upscaling entire DVD frames, but 4× bands still demand VRAM: tile processing or patch-based upscaling recommended for GTX 1050 Mobile (4GB VRAM).
- Load models one at a time; use FP16 and clear CUDA cache between heavy steps.
- Keep frames and masks in RAM (you have 24GB) but stream tiles to GPU for upscaling to avoid OOM.

---

## Failure modes and mitigations
- Parallax/camera motion: single generated background may show seams; if camera moves significantly, either choose multiple background keys or run frame‑coherent inpainting.
- Mask drift: re-segment at intervals and interpolate masks; use optical flow warping for mask propagation.
- VRAM OOM during 4× upscaling: fallback to tiled upscaling or 3× for bands combined with small additional inpainting to fill gaps.

---

## Artifacts & visual rules
- Never let the inpainting model touch Foreground/Face pixels. Validate inpainted output by diffing with masked original to ensure subject pixels unchanged.
- Blend band seams with feathered masks and Poisson blending to avoid sharp transitions between widescreen pixels and upscaled band pixels.

---

## What I did and what’s next
I consolidated the final decisions into this single context file: band‑only upscaling (4× per your manual match), single‑frame background inpainting, and strict center‑sample mapping rules. The file is ready for your local agent/copilot to consume as the authoritative context for generating the actual Python modules and transforms. Next, your local agent can use this file to produce the per‑module code (frame extraction, band crop/tiling/upscaling, mask handling, single background inpaint orchestration, foreground tracking, and final assembly).

If you want the one‑page runbook next (exact transform math lines for Python/ImageMagick/FFmpeg placement), I can produce that as a short file to paste into your repo.