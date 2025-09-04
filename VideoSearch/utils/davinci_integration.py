import logging
import subprocess
import json
import os
from pathlib import Path

logger = logging.getLogger(__name__)

def get_python312_path():
    return "py -3.12"

def get_subprocess_script_path():
    return os.path.join(os.path.dirname(__file__), '..', '..', 'davinci_subprocess.py')

def check_davinci_status() -> dict:
    try:
        script_path = get_subprocess_script_path()
        result = subprocess.run(
            f"{get_python312_path()} {script_path} status",
            shell=True,
            capture_output=True,
            text=True,
            timeout=10
        )
        
        if result.returncode == 0:
            return json.loads(result.stdout)
        else:
            return {
                'available': False,
                'error': f'Script failed: {result.stderr}'
            }
    except subprocess.TimeoutExpired:
        return {
            'available': False,
            'error': 'DaVinci status check timed out'
        }
    except Exception as e:
        return {
            'available': False,
            'error': f'Unexpected error: {str(e)}'
        }

def get_current_frame_from_davinci() -> dict:
    try:
        script_path = get_subprocess_script_path()
        result = subprocess.run(
            f"{get_python312_path()} {script_path} get_current_frame",
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            if response.get('success'):
                logger.info(f"Successfully retrieved current frame from DaVinci at {response.get('timecode')}")
            return response
        else:
            return {
                'success': False,
                'error': f'Script failed: {result.stderr}'
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'DaVinci operation timed out'
        }
    except Exception as e:
        logger.error(f"DaVinci frame capture error: {str(e)}")
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }

def send_clip_to_davinci_preview(keyframe) -> dict:
    try:
        video_path = keyframe.clip.video.file_path
        fps = keyframe.clip.video.fps()
        
        script_path = get_subprocess_script_path()
        cmd = (
            f"{get_python312_path()} {script_path} send_clip "
            f"--video-path \"{video_path}\" "
            f"--start-frame {keyframe.clip.start_frame} "
            f"--end-frame {keyframe.clip.end_frame} "
            f"--keyframe-frame {keyframe.frame} "
            f"--fps {fps}"
        )
        
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=30
        )
        
        if result.returncode == 0:
            response = json.loads(result.stdout)
            if response.get('success'):
                logger.info(f"Successfully sent clip to DaVinci: {video_path}")
            return response
        else:
            return {
                'success': False,
                'error': f'Script failed: {result.stderr}'
            }
            
    except subprocess.TimeoutExpired:
        return {
            'success': False,
            'error': 'DaVinci operation timed out'
        }
    except Exception as e:
        logger.error(f"DaVinci integration error: {str(e)}")
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }