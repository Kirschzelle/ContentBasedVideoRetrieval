from VideoSearch.management.base import StyledCommand as BaseCommand
from VideoSearch.models import Video, Keyframe
from VideoSearch.utils.audio import AudioTranscriber, keyframe_time_from_video_info
from collections import defaultdict
from pathlib import Path

class Command(BaseCommand):
    help = "Extract audio transcripts for keyframes missing them."

    def add_arguments(self, parser):
        parser.add_argument(
            '--model-size',
            type=str,
            default='base',
            choices=['tiny', 'base', 'small', 'medium', 'large'],
            help='Whisper model size (default: base)'
        )
        parser.add_argument(
            '--context-window',
            type=float,
            default=5.0,
            help='Context window in seconds around keyframes (default: 5.0)'
        )
        parser.add_argument(
            '--batch-videos',
            type=int,
            default=1,
            help='Number of videos to process in parallel (default: 1)'
        )

    def handle(self, *args, **options):
        model_size = options['model_size']
        context_window = options['context_window']
        batch_videos = options['batch_videos']

        # Initialize transcriber
        transcriber = AudioTranscriber(model_size=model_size, command=self)

        # Find keyframes without transcript data
        keyframes_without_transcripts = Keyframe.objects.filter(
            transcript_text__isnull=True
        ).select_related('clip', 'clip__video')

        if not keyframes_without_transcripts.exists():
            self.stdout.write(self.style_success("All keyframes already have transcripts."))
            return

        # Group keyframes by video for efficient processing
        videos_to_process = defaultdict(list)
        for keyframe in keyframes_without_transcripts:
            videos_to_process[keyframe.clip.video].append(keyframe)

        total_videos = len(videos_to_process)
        total_keyframes = keyframes_without_transcripts.count()

        self.stdout.write(f"[AUDIO] Found {total_keyframes} keyframes from {total_videos} videos needing transcripts")
        self.stdout.write(f"   Using model: {model_size}")
        self.stdout.write(f"   Context window: {context_window}s")

        processed_videos = 0
        processed_keyframes = 0

        for video, keyframes in videos_to_process.items():
            try:
                processed_videos += 1
                self.stdout.write(f"\\n[{processed_videos}/{total_videos}] Processing: {Path(video.file_path).name}")
                self.stdout.write(f"   Keyframes to process: {len(keyframes)}")

                # Check if video file exists
                video_path = Path(video.processing_path)
                if not video_path.exists():
                    self.stdout.write(self.style_error(f"   ERROR: Video file not found: {video_path}"))
                    continue

                # Transcribe the video
                try:
                    segments = transcriber.transcribe_video(str(video_path))
                    self.stdout.write(f"   Generated {len(segments)} transcript segments")
                    
                    if not segments:
                        self.stdout.write(self.style_warning("   WARNING: No speech detected in video"))
                        # Mark keyframes as processed with empty transcripts
                        for keyframe in keyframes:
                            keyframe.transcript_text = ""
                            keyframe.transcript_confidence = 0.0
                            keyframe.transcript_context = ""
                            keyframe.save(update_fields=[
                                'transcript_text', 'transcript_confidence', 'transcript_context'
                            ])
                            processed_keyframes += 1
                        continue

                    # Process each keyframe
                    for keyframe in keyframes:
                        # Calculate keyframe time
                        keyframe_time = keyframe_time_from_video_info(
                            video, keyframe.clip, keyframe.frame
                        )

                        # Map transcript to keyframe
                        primary_text, confidence, context_text = transcriber.map_transcript_to_keyframe(
                            keyframe_time, segments, context_window
                        )

                        # Update keyframe with transcript data
                        keyframe.transcript_text = primary_text or ""
                        keyframe.transcript_confidence = confidence or 0.0
                        keyframe.transcript_context = context_text or ""
                        keyframe.save(update_fields=[
                            'transcript_text', 'transcript_confidence', 'transcript_context'
                        ])
                        processed_keyframes += 1

                        if processed_keyframes % 10 == 0:
                            self.stdout.write(f"   Progress: {processed_keyframes}/{total_keyframes} keyframes")

                    self.stdout.write(f"   DONE: Completed {len(keyframes)} keyframes")

                except Exception as e:
                    self.stdout.write(self.style_error(f"   ERROR: Transcription failed: {e}"))
                    continue

            except Exception as e:
                self.stdout.write(self.style_error(f"ERROR: Error processing video {video.id}: {e}"))
                continue

        # Summary
        self.stdout.write(f"\\n[SUMMARY]")
        self.stdout.write(f"   Videos processed: {processed_videos}/{total_videos}")
        self.stdout.write(f"   Keyframes updated: {processed_keyframes}/{total_keyframes}")

        if processed_keyframes > 0:
            self.stdout.write(self.style_success("COMPLETE: Audio transcription finished!"))
        else:
            self.stdout.write(self.style_warning("WARNING: No keyframes were updated"))

        # Show some sample results
        sample_keyframes = Keyframe.objects.filter(
            transcript_text__isnull=False,
            transcript_text__gt=""
        ).order_by('-id')[:3]

        if sample_keyframes.exists():
            self.stdout.write(f"\\n[SAMPLE TRANSCRIPTS]")
            for kf in sample_keyframes:
                video_name = Path(kf.clip.video.file_path).name
                transcript_preview = kf.transcript_text[:100] + "..." if len(kf.transcript_text) > 100 else kf.transcript_text
                confidence = kf.transcript_confidence or 0.0
                self.stdout.write(f"   {video_name} (conf: {confidence:.2f}): {transcript_preview}")