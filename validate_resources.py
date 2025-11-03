#!/usr/bin/env python3
"""
Resource validation script for Video Meme Compositor.

This script helps validate your resources.txt file and checks that all 
specified files exist and are accessible.
"""

import configparser
from pathlib import Path
import sys

def validate_resources():
    """Validate all resources specified in resources.txt"""
    
    resources_file = Path("resources.txt")
    
    if not resources_file.exists():
        print("❌ resources.txt file not found!")
        print("📝 Please create a resources.txt file with your asset paths.")
        print("📖 See EXAMPLE_FILE_ORGANIZATION.md for examples.")
        return False
    
    print("🔍 Validating resources.txt...")
    print("=" * 50)
    
    try:
        config = configparser.ConfigParser()
        config.read(resources_file)
        
        all_valid = True
        
        # Validate video sources
        if 'VIDEO_SOURCES' in config:
            print("\n📹 VIDEO SOURCES:")
            video_section = config['VIDEO_SOURCES']
            
            # Check widescreen source
            if 'widescreen_source' in video_section:
                path = Path(video_section['widescreen_source'])
                if path.exists():
                    print(f"  ✅ Widescreen source: {path}")
                else:
                    print(f"  ❌ Widescreen source NOT FOUND: {path}")
                    all_valid = False
            else:
                print("  ❌ widescreen_source not specified!")
                all_valid = False
            
            # Check DVD source
            if 'dvd_source' in video_section:
                path = Path(video_section['dvd_source'])
                if path.exists():
                    print(f"  ✅ DVD source: {path}")
                else:
                    print(f"  ❌ DVD source NOT FOUND: {path}")
                    all_valid = False
            else:
                print("  ❌ dvd_source not specified!")
                all_valid = False
            
            # Check optional 4:3 source
            if 'aligned_4x3_source' in video_section:
                path = Path(video_section['aligned_4x3_source'])
                if path.exists():
                    print(f"  ✅ 4:3 source: {path}")
                else:
                    print(f"  ⚠️  4:3 source NOT FOUND (optional): {path}")
        else:
            print("  ❌ [VIDEO_SOURCES] section missing!")
            all_valid = False
        
        # Validate face assets
        if 'FACE_ASSETS' in config:
            print("\n😄 FACE ASSETS:")
            face_section = config['FACE_ASSETS']
            
            if 'face_asset' in face_section:
                path = Path(face_section['face_asset'])
                if path.exists():
                    print(f"  ✅ Face asset: {path}")
                else:
                    print(f"  ❌ Face asset NOT FOUND: {path}")
                    all_valid = False
            else:
                print("  ❌ face_asset not specified!")
                all_valid = False
        else:
            print("  ❌ [FACE_ASSETS] section missing!")
            all_valid = False
        
        # Validate reference frames (optional)
        if 'REFERENCE_FRAMES' in config:
            print("\n🎯 REFERENCE FRAMES:")
            ref_section = config['REFERENCE_FRAMES']
            
            if 'reference_frame' in ref_section:
                path = Path(ref_section['reference_frame'])
                if path.exists():
                    print(f"  ✅ Reference frame: {path}")
                    print("     (Can be from DVD, widescreen, or any external source)")
                else:
                    print(f"  ⚠️  Reference frame NOT FOUND (optional): {path}")
            else:
                print("  ℹ️  No reference frame specified (will use default timing)")
        else:
            print("  ℹ️  No [REFERENCE_FRAMES] section (will use default timing)")
        
        # Validate output settings
        if 'OUTPUT' in config:
            print("\n📤 OUTPUT SETTINGS:")
            output_section = config['OUTPUT']
            
            if 'output_directory' in output_section:
                path = Path(output_section['output_directory'])
                path.mkdir(parents=True, exist_ok=True)  # Create if needed
                print(f"  ✅ Output directory: {path}")
            
            if 'debug_directory' in output_section:
                path = Path(output_section['debug_directory'])
                path.mkdir(parents=True, exist_ok=True)  # Create if needed
                print(f"  ✅ Debug directory: {path}")
                
            if 'output_video_name' in output_section:
                print(f"  ✅ Output video name: {output_section['output_video_name']}")
                
            if 'output_gif_name' in output_section:
                print(f"  ✅ Output GIF name: {output_section['output_gif_name']}")
        
        # Validate model paths (optional)
        if 'MODELS' in config:
            print("\n🤖 AI MODELS:")
            models_section = config['MODELS']
            
            model_files = [
                ('realesrgan_model', 'Real-ESRGAN upscaler'),
                ('sam_model', 'Segment Anything Model'),
                ('face_detector_prototxt', 'Face detector config'),
                ('face_detector_model', 'Face detector model')
            ]
            
            for key, description in model_files:
                if key in models_section:
                    path = Path(models_section[key])
                    if path.exists():
                        print(f"  ✅ {description}: {path}")
                    else:
                        print(f"  ⚠️  {description} NOT FOUND (will auto-download): {path}")
        
        print("\n" + "=" * 50)
        
        if all_valid:
            print("🎉 All required resources validated successfully!")
            print("🚀 You're ready to run the Video Meme Compositor!")
            print("💡 Run: setup.bat or python main.py")
            return True
        else:
            print("❌ Some required resources are missing!")
            print("📋 Please check the paths in your resources.txt file.")
            print("📖 See EXAMPLE_FILE_ORGANIZATION.md for examples.")
            return False
            
    except Exception as e:
        print(f"❌ Error reading resources.txt: {e}")
        return False


def print_example_resources():
    """Print an example resources.txt file"""
    print("\n📝 Example resources.txt file:")
    print("=" * 50)
    
    example = """[VIDEO_SOURCES]
# Your video files can be anywhere with any names
widescreen_source = C:\\Videos\\my_widescreen.mp4
dvd_source = D:\\Downloads\\dvd_rip.avi
aligned_4x3_source = videos\\optional_4x3.mp4

[FACE_ASSETS]
# Face image can be anywhere with any name
face_asset = assets\\faces\\my_face.jpg

[REFERENCE_FRAMES]
# Reference frame can be from ANY source (DVD, widescreen, or external)
reference_frame = reference\\sync_frame.jpg

[OUTPUT]
output_directory = output
debug_directory = debug
output_video_name = my_gandalf_meme.mp4
output_gif_name = my_gandalf_meme.gif

[MODELS]
# Optional: specify custom model locations
realesrgan_model = models\\RealESRGAN_x4plus.pth
sam_model = models\\sam_vit_h_4b8939.pth"""
    
    print(example)
    print("=" * 50)


if __name__ == "__main__":
    print("🎬 Video Meme Compositor - Resource Validator")
    print("=" * 50)
    
    if len(sys.argv) > 1 and sys.argv[1] == "--example":
        print_example_resources()
        sys.exit(0)
    
    success = validate_resources()
    
    if not success:
        print("\n💡 Need help? Run: python validate_resources.py --example")
        sys.exit(1)
    else:
        sys.exit(0)