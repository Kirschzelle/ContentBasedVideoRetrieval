# DaVinci Resolve Integration Setup

## Prerequisites

1. **DaVinci Resolve** (any recent version - 17, 18, or 19)
2. **Python 3.x** (same version as your Django app)

## Required Setup Steps

### 1. Enable DaVinci Resolve API

**In DaVinci Resolve:**
1. Go to **DaVinci Resolve > Preferences** (or **File > Preferences** on Windows)
2. Navigate to **System > General**
3. Find **External scripting using** section
4. Check **"Network"** 
5. Set **Listen IP** to `127.0.0.1` (localhost)
6. Set **Listen Port** to `8081` (default)
7. Click **Save**
8. **Restart DaVinci Resolve**

### 2. Install DaVinci Python API

The Python API comes with DaVinci Resolve installation:

**Windows:**
```cmd
# Add DaVinci's Python modules to your path
# Location is usually:
# C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules
```

**Add to your Django settings or startup script:**
```python
import sys
import os

# Add DaVinci Resolve API path
davinci_path = r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting\Modules"
if os.path.exists(davinci_path):
    sys.path.append(davinci_path)
```

**Mac:**
```bash
# Usually located at:
# /Applications/DaVinci Resolve/DaVinci Resolve.app/Contents/Resources/Developer/Scripting/Modules/
```

**Linux:**
```bash
# Usually located at:
# /opt/resolve/Developer/Scripting/Modules/
```

### 3. Test the Connection

```python
# Test script to verify setup
try:
    import DaVinciResolveScript as dvr
    resolve = dvr.scriptapp("Resolve")
    
    if resolve:
        print("✅ Successfully connected to DaVinci Resolve!")
        
        project = resolve.GetProjectManager().GetCurrentProject()
        if project:
            print(f"✅ Current project: {project.GetName()}")
        else:
            print("⚠️  No project open - create/open a project in DaVinci")
    else:
        print("❌ Could not connect to DaVinci Resolve")
        print("Make sure DaVinci is running and API is enabled")
        
except ImportError:
    print("❌ DaVinci Resolve API not found")
    print("Check that the API path is correctly added to Python path")
```

## Usage Workflow

1. **Start DaVinci Resolve**
2. **Create or open a project** (required!)
3. **Open your video search web interface**
4. **Search for clips**
5. **Click "🎬 Send to DaVinci"** on any result
6. **DaVinci will load the clip in preview** with exact in/out points
7. **Playhead jumps to the keyframe** that matched your search

## Troubleshooting

### "Could not connect to DaVinci Resolve"
- Make sure DaVinci Resolve is running
- Check that API is enabled in Preferences > System > General
- Try restarting DaVinci after enabling API

### "DaVinci Resolve API not available"
- Check Python path includes DaVinci modules directory
- Verify DaVinci installation includes Developer tools
- Try running test script above

### "No project open in DaVinci Resolve"
- Create a new project: File > New Project
- Or open existing project: File > Open Project

### Import Errors
- If video file fails to import, check file path is accessible
- Make sure DaVinci can read the video format
- Check file permissions

## What Happens When You Click "Send to DaVinci"

1. **Imports video file** to DaVinci media pool (if not already there)
2. **Sets in/out points** to the exact clip boundaries found by search
3. **Loads in source viewer** (non-destructive - won't affect timeline)
4. **Jumps to keyframe position** - the exact moment that matched your search
5. **Adds marker** with transcript text (if available)

This gives you immediate preview of the exact moment found by AI, with full context to scrub around and decide if you want to use it.

## Notes

- **Non-destructive**: Only loads in preview, doesn't touch your timeline
- **Original quality**: Uses original video files, not web proxies
- **Frame accurate**: Exact timecodes preserved
- **Professional workflow**: Integrates with your existing DaVinci project

## Security

The DaVinci API only accepts connections from localhost by default, so it's secure for local development. For production deployments, ensure proper network security if enabling remote connections.