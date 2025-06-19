# ContentBasedVideoRetrieval
A Content-Based Video Retrieval System developed on the context of a course on the University of Klagenfurt.

# System Requirements:
- Python 3.11.X
- FFmpeg

## Install FFmpeg

### Linux (Ubuntu/Debian)
```bash
sudo apt update
sudo apt install ffmpeg
```

### macOS (with Homebrew)
```bash
brew install ffmpeg
```

### Windows
1. Download FFmpeg from [gyan.dev](https://www.gyan.dev/ffmpeg/builds/), choose a "release full" ZIP.
2. Extract to `C:\ffmpeg\`.
3. Add `C:\ffmpeg\bin` to your system PATH.
4. Open Command Prompt and check:
   ```bash
   ffmpeg -version
   ffprobe -version
   ```

# Setup
Clone project:
   ```bash
   git clone --recurse-submodules https://github.com/Kirschzelle/ContentBasedVideoRetrieval.git
   ```
Download videos you want to search and place them into './data/videos/'.
   ```bash
   python manage.py makemigrations
   python manage.py migrate
   python manage.py import_videos
   python manage.py extract_clips
   ```