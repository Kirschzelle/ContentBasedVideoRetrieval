"""
Simple DaVinci Resolve integration for sending clips to preview.
"""

import logging
from typing import Optional

logger = logging.getLogger(__name__)

def frames_to_timecode(frame: int, fps: float) -> str:
    """Convert frame number to timecode (HH:MM:SS:FF)."""
    total_seconds = frame / fps
    hours = int(total_seconds // 3600)
    minutes = int((total_seconds % 3600) // 60)
    seconds = int(total_seconds % 60)
    frames = int(frame % fps)
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"

def send_clip_to_davinci_preview(keyframe) -> dict:
    """
    Send a clip to DaVinci Resolve preview with in/out points set.
    
    Args:
        keyframe: Keyframe object with clip and video data
        
    Returns:
        Dict with success status and any error messages
    """
    try:
        # Import DaVinci Resolve API
        try:
            import DaVinciResolveScript as dvr
        except ImportError:
            return {
                'success': False,
                'error': 'DaVinci Resolve API not available. Make sure DaVinci is installed and API is enabled.'
            }
        
        # Connect to DaVinci Resolve
        resolve = dvr.scriptapp("Resolve")
        if not resolve:
            return {
                'success': False,
                'error': 'Could not connect to DaVinci Resolve. Make sure DaVinci is running and API is enabled.'
            }
        
        # Get current project
        project_manager = resolve.GetProjectManager()
        project = project_manager.GetCurrentProject()
        
        if not project:
            return {
                'success': False,
                'error': 'No project open in DaVinci Resolve. Please create or open a project.'
            }
        
        # Get media pool
        media_pool = project.GetMediaPool()
        
        # Get video file path (original, not web proxy)
        video_path = keyframe.clip.video.file_path
        
        # Import media file to media pool (if not already there)
        try:
            media_items = media_pool.ImportMedia([video_path])
            if not media_items:
                return {
                    'success': False,
                    'error': f'Could not import video file: {video_path}'
                }
            media_item = media_items[0]
        except Exception as e:
            return {
                'success': False,
                'error': f'Failed to import media: {str(e)}'
            }
        
        # Calculate timecodes
        fps = keyframe.clip.video.fps()
        start_timecode = frames_to_timecode(keyframe.clip.start_frame, fps)
        end_timecode = frames_to_timecode(keyframe.clip.end_frame, fps)
        keyframe_timecode = frames_to_timecode(keyframe.frame, fps)
        
        # Set in/out points on media item
        media_item.SetClipProperty("Start TC", start_timecode)
        media_item.SetClipProperty("End TC", end_timecode)
        
        # Load into source viewer
        media_item.LoadClipIntoViewer("source")
        
        # Jump to keyframe position
        resolve.GetCurrentPage().GetViewer().SetCurrentTimecode(keyframe_timecode)
        
        # Add marker at keyframe position with transcript if available
        if keyframe.transcript_text:
            # Note: AddMarker might not work on all versions, so we'll try/catch it
            try:
                media_item.AddMarker(
                    keyframe.frame - keyframe.clip.start_frame,  # Relative to clip start
                    "Red",
                    keyframe.transcript_text[:100],  # Limit marker text length
                    1.0
                )
            except Exception:
                # Marker addition failed, but that's okay
                pass
        
        logger.info(f"Successfully sent clip to DaVinci: {video_path} at {keyframe_timecode}")
        
        return {
            'success': True,
            'message': f'Clip loaded in DaVinci preview at {keyframe_timecode}',
            'video_name': keyframe.clip.video.file_name,
            'timecode': keyframe_timecode,
            'clip_range': f"{start_timecode} - {end_timecode}"
        }
        
    except Exception as e:
        logger.error(f"DaVinci integration error: {str(e)}")
        return {
            'success': False,
            'error': f'Unexpected error: {str(e)}'
        }