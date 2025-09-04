from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.models import Clip, ClipPredictionCache
from VideoSearch.utils.clip_processor import ClipProcessor
from multiprocessing import Pool
from tqdm import tqdm

def init_worker():
    import os
    os.environ.setdefault("DJANGO_SETTINGS_MODULE", "ContentBasedVideoRetrieval.settings")
    import django
    django.setup()

def process_clip_worker(clip_id, whisper_model_size, progress_info):
    from VideoSearch.models import Clip
    
    try:
        clip = Clip.objects.select_related('video').get(id=clip_id)
        processor = ClipProcessor(whisper_model_size=whisper_model_size)
        keyframes = processor.process_clip(clip)
        
        ClipPredictionCache.objects.filter(clip=clip).delete()
        
        progress_str = f"[{progress_info[0]}/{progress_info[1]}]" if progress_info else ""
        return f"{progress_str} Processed clip {clip_id}: {len(keyframes)} keyframes"
        
    except Exception as e:
        return f"ERROR processing clip {clip_id}: {str(e)}"

class Command(BaseCommand):
    def add_arguments(self, parser):
        parser.add_argument(
            '--whisper-model', 
            type=str, 
            default='base',
            choices=['tiny', 'base', 'small', 'medium', 'large'],
            help='Whisper model size for audio transcription'
        )
        parser.add_argument(
            '--workers', 
            type=int, 
            default=1, 
            help='Number of worker processes (1 disables multiprocessing)'
        )
        parser.add_argument(
            '--clip-id',
            type=int,
            help='Process specific clip ID only'
        )

    def handle(self, *args, **options):
        whisper_model = options['whisper_model']
        workers = options['workers']
        specific_clip_id = options.get('clip_id')
        
        if specific_clip_id:
            clips = Clip.objects.filter(id=specific_clip_id).select_related('video')
            if not clips.exists():
                self.stdout.write(self.style_error(f"Clip {specific_clip_id} not found"))
                return
        else:
            clip_caches = ClipPredictionCache.objects.select_related('clip', 'clip__video').all()
            clips = [cache.clip for cache in clip_caches]
        
        if not clips:
            self.stdout.write(self.style_warning("No clips found for processing"))
            return
        
        self.stdout.write(self.style_info(f"Processing {len(clips)} clips with Whisper model: {whisper_model}"))
        
        if workers == 1:
            processor = ClipProcessor(whisper_model_size=whisper_model, command=self)
            
            with tqdm(total=len(clips), desc="Processing clips", unit="clip", 
                     dynamic_ncols=True, leave=True, ascii=True) as pbar:
                
                for i, clip in enumerate(clips, 1):
                    pbar.set_postfix_str(f"Clip {clip.id}")
                    pbar.refresh()
                    
                    try:
                        keyframes = processor.process_clip(clip)
                        pbar.set_postfix_str(f"Clip {clip.id}: {len(keyframes)} keyframes")
                        tqdm.write(f"[OK] Clip {clip.id}: {len(keyframes)} keyframes created")
                        
                        ClipPredictionCache.objects.filter(clip=clip).delete()
                        
                    except Exception as e:
                        tqdm.write(f"[ERROR] Clip {clip.id}: {str(e)}")
                        pbar.set_postfix_str(f"ERROR on clip {clip.id}")
                    
                    pbar.update(1)
        else:
            args_list = [
                (clip.id, whisper_model, (i, len(clips)))
                for i, clip in enumerate(clips, 1)
            ]
            
            with Pool(processes=workers, initializer=init_worker) as pool:
                results = pool.starmap(process_clip_worker, args_list)
            
            for result in results:
                if "ERROR" in result:
                    self.stdout.write(self.style_error(result))
                else:
                    self.stdout.write(self.style_success(result))
        
        total_keyframes = sum(clip.keyframe_set.count() for clip in clips)
        self.stdout.write(self.style_success(f"Processing complete! Total keyframes created: {total_keyframes}"))