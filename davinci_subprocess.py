import sys
import os
import json
import argparse

def setup_davinci_env():
    resolve_api_path = r"C:\ProgramData\Blackmagic Design\DaVinci Resolve\Support\Developer\Scripting"
    resolve_lib_path = r"C:\Program Files\Blackmagic Design\DaVinci Resolve\fusionscript.dll"
    modules_path = os.path.join(resolve_api_path, "Modules")

    os.environ['RESOLVE_SCRIPT_API'] = resolve_api_path
    os.environ['RESOLVE_SCRIPT_LIB'] = resolve_lib_path
    sys.path.append(modules_path)

def check_status():
    try:
        setup_davinci_env()
        import DaVinciResolveScript as dvr
        
        resolve = dvr.scriptapp("Resolve")
        if resolve:
            project_manager = resolve.GetProjectManager()
            project = project_manager.GetCurrentProject()
            
            return {
                'available': True,
                'project_open': project is not None,
                'project_name': project.GetName() if project else None
            }
        else:
            return {'available': False, 'error': 'Could not connect to DaVinci Resolve'}
            
    except ImportError:
        return {'available': False, 'error': 'DaVinci Resolve API not available'}
    except Exception as e:
        return {'available': False, 'error': str(e)}

def send_clip(video_path, start_frame, end_frame, keyframe_frame, fps):
    try:
        setup_davinci_env()
        import DaVinciResolveScript as dvr
        
        def frame_to_timecode(frame, fps):
            total_seconds = frame / fps
            hours = int(total_seconds // 3600)
            minutes = int((total_seconds % 3600) // 60)
            seconds = int(total_seconds % 60)
            frames = int(frame % fps)
            return f"{hours:02d}:{minutes:02d}:{seconds:02d}:{frames:02d}"
        
        resolve = dvr.scriptapp("Resolve")
        if not resolve:
            return {'success': False, 'error': 'Could not connect to DaVinci Resolve'}
        
        project_manager = resolve.GetProjectManager()
        project = project_manager.GetCurrentProject()
        
        if not project:
            return {'success': False, 'error': 'No project open in DaVinci Resolve'}
        
        media_pool = project.GetMediaPool()
        
        media_items = media_pool.ImportMedia([video_path])
        if not media_items:
            return {'success': False, 'error': f'Could not import video file: {video_path}'}
        
        media_item = media_items[0]
        
        start_tc = frame_to_timecode(start_frame, fps)
        end_tc = frame_to_timecode(end_frame, fps)
        keyframe_tc = frame_to_timecode(keyframe_frame, fps)
        
        media_item.SetClipProperty("Start TC", start_tc)
        media_item.SetClipProperty("End TC", end_tc)
        
        resolve.OpenPage("edit")
        
        return {
            'success': True,
            'message': f'Clip loaded in DaVinci at {keyframe_tc}',
            'video_name': os.path.basename(video_path),
            'timecode': keyframe_tc
        }
        
    except Exception as e:
        return {'success': False, 'error': str(e)}

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('command', choices=['status', 'send_clip'])
    parser.add_argument('--video-path')
    parser.add_argument('--start-frame', type=int)
    parser.add_argument('--end-frame', type=int)
    parser.add_argument('--keyframe-frame', type=int)
    parser.add_argument('--fps', type=float)
    
    args = parser.parse_args()
    
    if args.command == 'status':
        result = check_status()
    elif args.command == 'send_clip':
        if not all([args.video_path, args.start_frame is not None, args.end_frame is not None, 
                   args.keyframe_frame is not None, args.fps is not None]):
            result = {'success': False, 'error': 'Missing required arguments for send_clip'}
        else:
            result = send_clip(args.video_path, args.start_frame, args.end_frame, args.keyframe_frame, args.fps)
    
    print(json.dumps(result))

if __name__ == "__main__":
    main()