# Example File Organization

This shows how you can organize your files with complete flexibility using resources.txt:

## Example Directory Structure:

```
C:\My Projects\Gandalf_the_Gay\          # Main project directory
├── src\                                  # Source code (don't change)
├── setup.bat                            # Setup script
├── main.py                              # Main application  
├── resources.txt                        # YOUR FILE PATHS (edit this!)
├── config.yaml                          # Optional fallback config
└── README.md

C:\Videos\Source_Material\               # Your videos can be anywhere!
├── high_res_widescreen.mp4              # Your widescreen source
├── dvd_rip_720x540.avi                  # Your DVD source  
└── some_4x3_video.mp4                   # Optional 4:3 source

D:\Downloads\Assets\                     # Assets can be anywhere too!
├── faces\
│   ├── El_xox_sillyface.jpg            # Your face asset
│   └── backup_face.png                 # Additional faces
└── reference\
    ├── sync_frame_dvd.jpg              # Reference from DVD
    ├── sync_frame_external.png         # Reference from different source
    └── timing_reference.bmp            # Any format works

E:\AI_Models\                           # Models can be elsewhere too
├── RealESRGAN_x4plus.pth              # Upscaling model  
└── sam_vit_h_4b8939.pth               # Segmentation model
```

## Your resources.txt would look like:

```ini
[VIDEO_SOURCES]
widescreen_source = C:\Videos\Source_Material\high_res_widescreen.mp4
dvd_source = C:\Videos\Source_Material\dvd_rip_720x540.avi  
aligned_4x3_source = C:\Videos\Source_Material\some_4x3_video.mp4

[FACE_ASSETS]
face_asset = D:\Downloads\Assets\faces\El_xox_sillyface.jpg

[REFERENCE_FRAMES] 
# This can be from ANY source - not just your video sources!
# Examples:
# - Frame extracted from your DVD source
# - Frame extracted from your widescreen source  
# - Screenshot from a completely different video
# - Any image that helps with timing synchronization
reference_frame = D:\Downloads\Assets\reference\sync_frame_dvd.jpg

# SPEED BOOST: Smart timestamp hints for lightning-fast sync!
# Use any format that's convenient for you:
widescreen_reference_time = 1:23.5    # 1 minute 23.5 seconds (MM:SS)
dvd_reference_time = 42               # 42 seconds (simple)
# Also works: 1:02:30 (HH:MM:SS), 1h23m45s (text), 83.5 (decimal seconds)
# This changes search from 2+ minutes to just 3 seconds!

[MODELS]
# Optional: specify custom model locations
realesrgan_model = E:\AI_Models\RealESRGAN_x4plus.pth
sam_model = E:\AI_Models\sam_vit_h_4b8939.pth

[OUTPUT]
output_directory = C:\My Projects\Gandalf_the_Gay\output
debug_directory = C:\My Projects\Gandalf_the_Gay\debug
output_video_name = my_custom_gandalf_meme.mp4
output_gif_name = my_custom_gandalf_meme.gif
```

## Key Benefits:

1. **Files can have any names** - not restricted to specific filenames
2. **Files can be in any folders** - organize however you want
3. **Reference frame flexibility** - can be from DVD, widescreen, or completely external source
4. **Multiple drives supported** - spread files across different drives
5. **Custom output names** - name your final video anything you want

## Reference Frame Sources:

The reference frame is **extremely flexible**:

- ✅ Frame extracted from your DVD source video
- ✅ Frame extracted from your widescreen source video  
- ✅ Screenshot from a completely different 720×540 video
- ✅ Image from any external source that matches timing
- ✅ Any image format: jpg, png, bmp, tiff, etc.

The system will automatically search both video sources to find the best matching timestamp!