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

def process_davinci_frame_for_filters(frame_path: str) -> dict:
    try:
        from PIL import Image
        from VideoSearch.utils.visual_feature_extractor import VisualFeatureExtractor  
        from VideoSearch.utils.objects import ObjectDetector
        import uuid
        from django.conf import settings
        from pathlib import Path
        
        image = Image.open(frame_path)
        
        visual_extractor = VisualFeatureExtractor()
        object_extractor = ObjectDetector()
        
        features = visual_extractor.extract_features(image)
        object_vector = object_extractor.extract_vector(image)
        
        frame_id = str(uuid.uuid4())
        temp_dir = Path(settings.MEDIA_ROOT) / 'temp_frames'
        temp_dir.mkdir(exist_ok=True)
        
        temp_image_path = temp_dir / f"{frame_id}.jpg"
        image.save(temp_image_path, 'JPEG', quality=85)
        
        return {
            'success': True,
            'frame_id': frame_id,
            'image_path': str(temp_image_path),
            'features': {
                'clip_emb': features['clip_emb'],
                'dino_emb': features.get('dino_emb'),
                'histogram': features.get('histogram'),
                'palette': features.get('palette'),
                'colorfulness': features.get('colorfulness'),
                'object_vector': object_vector
            },
            'image_info': {
                'width': image.width,
                'height': image.height
            }
        }
        
    except Exception as e:
        logger.error(f"DaVinci frame processing error: {str(e)}")
        return {
            'success': False,
            'error': f'Feature extraction failed: {str(e)}'
        }