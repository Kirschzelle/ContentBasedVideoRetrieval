#!/usr/bin/env python3
"""
DaVinci Resolve script to export current frame and send to search server.

To use:
1. Copy this script to DaVinci Resolve Scripts folder
2. Run from DaVinci Console or Scripts menu
3. Current frame will be exported and sent to search server
"""

import os
import tempfile
import requests
import json
from pathlib import Path

def export_current_frame_to_search(server_url="http://localhost:8000"):
    """
    Export current frame from DaVinci timeline and send to search server.
    
    Args:
        server_url: URL of your Django search server
    """
    try:
        import DaVinciResolveScript as dvr
    except ImportError:
        print("ERROR: DaVinci Resolve API not available")
        return False
    
    # Connect to DaVinci Resolve
    resolve = dvr.scriptapp("Resolve")
    if not resolve:
        print("ERROR: Could not connect to DaVinci Resolve")
        return False
    
    project = resolve.GetProjectManager().GetCurrentProject()
    if not project:
        print("ERROR: No project open in DaVinci Resolve")
        return False
    
    timeline = project.GetCurrentTimeline()
    if not timeline:
        print("ERROR: No timeline open")
        return False
    
    try:
        # Get current playhead position
        current_frame = timeline.GetCurrentTimecode()
        print(f"Exporting frame at timecode: {current_frame}")
        
        # Create temporary export path
        temp_dir = tempfile.gettempdir()
        export_filename = f"davinci_export_frame_{current_frame.replace(':', '')}.jpg"
        export_path = os.path.join(temp_dir, export_filename)
        
        # Set up render settings for single frame export
        project.SetRenderSettings({
            "SelectAllFrames": False,
            "MarkIn": timeline.GetCurrentTimecode(),
            "MarkOut": timeline.GetCurrentTimecode(),
            "TargetDir": temp_dir,
            "CustomName": export_filename.replace('.jpg', ''),
            "ImageFormat": "jpg",
            "ExportVideo": False,
            "ExportAudio": False,
            "FormatWidth": 1920,
            "FormatHeight": 1080,
            "PixelAspectRatio": "Square"
        })
        
        # Start render
        print("Rendering current frame...")
        project.StartRendering()
        
        # Wait for render to complete
        while project.IsRenderingInProgress():
            import time
            time.sleep(0.1)
        
        # Check if file was created
        if not os.path.exists(export_path):
            print(f"ERROR: Export file not found at {export_path}")
            return False
        
        print(f"Frame exported to: {export_path}")
        
        # Upload to search server
        upload_success = upload_frame_to_server(export_path, server_url)
        
        # Clean up exported file
        try:
            os.remove(export_path)
            print("Cleaned up exported file")
        except Exception as e:
            print(f"Warning: Could not clean up file: {e}")
        
        return upload_success
        
    except Exception as e:
        print(f"ERROR during export: {str(e)}")
        return False

def upload_frame_to_server(file_path, server_url):
    """
    Upload exported frame to search server.
    
    Args:
        file_path: Path to exported frame
        server_url: URL of Django search server
    """
    try:
        upload_url = f"{server_url}/external-frame/upload/"
        
        with open(file_path, 'rb') as f:
            files = {'frame_image': f}
            
            print(f"Uploading frame to: {upload_url}")
            response = requests.post(upload_url, files=files, timeout=30)
            
            if response.status_code == 200:
                result = response.json()
                if result.get('success'):
                    print(f"✓ Upload successful: {result.get('message')}")
                    print(f"  Image info: {result.get('image_info')}")
                    print(f"  Features extracted: {', '.join(result.get('features_available', []))}")
                    print("")
                    print("You can now use this frame as a filter in your search!")
                    return True
                else:
                    print(f"✗ Upload failed: {result.get('error')}")
                    return False
            else:
                print(f"✗ HTTP error {response.status_code}: {response.text}")
                return False
                
    except requests.exceptions.RequestException as e:
        print(f"✗ Network error: {str(e)}")
        print("Make sure your search server is running and accessible")
        return False
    except Exception as e:
        print(f"✗ Upload error: {str(e)}")
        return False

def main():
    """Main function for running the script."""
    print("=== DaVinci Frame Export to Search ===")
    print("Exporting current frame and sending to search server...")
    print("")
    
    # You can modify this URL to match your server
    SERVER_URL = "http://localhost:8000"
    
    success = export_current_frame_to_search(SERVER_URL)
    
    if success:
        print("")
        print("🎉 Success! Frame exported and uploaded.")
        print("Switch to your web browser to use it as a search filter.")
    else:
        print("")
        print("❌ Export failed. Check the errors above.")

if __name__ == "__main__":
    main()

# For DaVinci Console usage:
# exec(open('path/to/export_current_frame.py').read())