# Video Meme Compositor

A sophisticated video meme compositor that processes DVD and widescreen video sources to create enhanced memes with face insertions and AI-generated backgrounds.

## Features

- **Band-only upscaling**: 4x integer upscaling of DVD extra bands while preserving center detail
- **Single-frame AI inpainting**: Efficient background generation using one representative frame
- **Precise pixel mapping**: Integer transforms for exact center-sample alignment
- **Face insertion**: Tracked face asset insertion with blending
- **Advanced masking**: SAM-based segmentation with optical flow propagation

## Architecture

The system implements a sophisticated pipeline based on manual measurements and precise geometric calculations:

1. **Frame Extraction** (`frame_extractor.py`) - Synchronizes widescreen and DVD sources
2. **Band Processing** (`band_processor.py`) - Calculates geometry and upscales DVD bands (4x)
3. **Mask Generation** (`mask_generator.py`) - SAM-based segmentation with propagation
4. **Background Generation** (`background_generator.py`) - Single-frame AI inpainting
5. **Face Processing** (`face_processor.py`) - Face asset insertion with tracking
6. **Final Composition** (`compositor.py`) - Seam blending and video export

## Setup Instructions

### 1. Virtual Environment Setup

Run the provided setup script which automatically handles Python environment detection and package installation:

```batch
setup.bat
```

The script will:
- Detect if you need Python 3.11 (for AI/ML packages) or can use system Python
- Create/activate virtual environment
- Install all dependencies from `requirements.txt`
- Launch the application

### 2. Manual Setup (Alternative)

If you prefer manual setup:

```batch
# Create virtual environment
python -m venv venv
call venv\Scripts\activate.bat

# Install dependencies
pip install -r requirements.txt

# Run the application
python main.py
```

## Required Files

### Easy File Management with resources.txt

You can place your files **anywhere** with **any names** you want! Just edit the `resources.txt` file to specify their locations:

```ini
[VIDEO_SOURCES]
# Your files can be anywhere with any names:
widescreen_source = C:\My Videos\high_quality_source.mp4
dvd_source = D:\Downloads\dvd_rip.avi
aligned_4x3_source = videos\optional_4x3.mp4

[FACE_ASSETS]  
# Face images can be in any folder:
face_asset = assets\faces\custom_face.png

[REFERENCE_FRAMES]
# Reference frame can be from ANY source - not just your video sources!
reference_frame = reference\sync_frame_from_anywhere.jpg
```

### Input Videos

- **Widescreen source**: High-detail video (any resolution, e.g., 3840×1610)
  - Your primary high-quality source
  - Can be named anything, stored anywhere
  
- **DVD source**: 720×540 video with extra vertical bands  
  - Provides additional detail in top/bottom bands
  - Can be named anything, stored anywhere
  
- **Reference frame** (optional): Single frame for synchronization
  - **Can be from ANY source** - DVD, widescreen, or completely different video!
  - Can be a screenshot from any 720×540 or similar source
  - Used to establish timing alignment between sources
  - Example sources: frame from DVD, frame from widescreen, screenshot from different video

### Face Asset

- **Face insertion image**: Any image file (jpg, png, bmp, etc.)
  - This is the face image that will be inserted into the video
  - Should be a clear face image, preferably front-facing
  - Will be automatically resized and blended
  - Can be named anything (not just `El_xox_sillyface.jpg`)

### AI Model Files

The application will prompt you to download these if not found:

#### Upscaling Model
- **Real-ESRGAN**: `models/RealESRGAN_x4plus.pth`
  - Download from: https://github.com/xinntao/Real-ESRGAN/releases
  - Used for 4x upscaling of DVD bands

#### Segmentation Model  
- **Segment Anything Model**: `models/sam_vit_h_4b8939.pth`
  - Download from: https://github.com/facebookresearch/segment-anything
  - Used for foreground/background segmentation

#### Face Detection (Optional)
- **OpenCV DNN Face Detector**:
  - `models/deploy.prototxt`
  - `models/res10_300x300_ssd_iter_140000.caffemodel`
  - Download from OpenCV documentation
  - Falls back to Haar cascade if not available

## Configuration

Edit `config.yaml` to specify your input files and processing parameters:

```yaml
# Required input files
widescreen_source: "path/to/your/widescreen_video.mp4"
dvd_source: "path/to/your/dvd_video.mp4"  
face_asset: "El_xox_sillyface.jpg"

# Processing parameters
start_time: 0.0      # Start time in seconds
duration: 10.0       # Duration to process
output_width: 3840   # Final output width
output_height: 2160  # Final output height

# Performance settings (adjust for your GPU)
device: "cuda"       # "cuda" or "cpu"
max_vram_gb: 4.0     # Your GPU VRAM limit
tile_size: 512       # Tile size for upscaling
```

## Usage

### Option 1: Using resources.txt (Recommended)
1. Place your video files anywhere on your system  
2. Place your face image anywhere on your system
3. Edit `resources.txt` with the paths to your files
4. Run: `setup.bat` or `python main.py`

### Option 2: Using config.yaml (Traditional)
1. Place your video files in the project directory
2. Place your face image in the project directory  
3. Edit `config.yaml` with your file paths
4. Run: `setup.bat` or `python main.py`

**Note**: resources.txt takes precedence over config.yaml if both exist.

The application will:
1. Extract and synchronize frames from both sources
2. Calculate DVD band geometry from manual measurements  
3. Upscale DVD bands using Real-ESRGAN (4x)
4. Generate masks using SAM
5. Create AI background using inpainting
6. Insert face asset with tracking
7. Export final MP4 and GIF

## Output

The application creates:
- `output/final_video.mp4` - Final composed video
- `output/final_video.gif` - GIF version
- `output/masks/` - Generated masks for debugging
- `output/bands/` - Upscaled band images  
- `debug/` - Debug frames and reports

## Hardware Requirements

- **GPU**: NVIDIA GPU with 4GB+ VRAM recommended (GTX 1050 Mobile minimum)
- **RAM**: 8GB+ system RAM (24GB+ recommended for full resolution)
- **Storage**: 5GB+ free space for models and output

## Troubleshooting

### Common Issues

1. **CUDA out of memory**: Reduce `tile_size` in config.yaml
2. **Models not found**: Download required model files to `models/` directory
3. **Video sync issues**: Provide a reference frame for better synchronization
4. **Poor segmentation**: Ensure good contrast between subject and background

### Debug Output

Enable debug mode: `python main.py --debug`

This creates additional debug files in `debug/` directory for troubleshooting.

## Technical Details

This implementation follows the specifications in `RUNBOOK.md` and `context.md`:

- Uses manual measurement (960×540 center sample) to derive 4x upscale factor
- Maps DVD center pixel to canonical output center pixel exactly
- Applies integer transforms to avoid subpixel aliasing
- Runs AI inpainting only once on background key frame
- Never generatively replaces foreground/face pixels

The pipeline is optimized for GTX 1050 Mobile (4GB VRAM) with tiled processing and FP16 precision.