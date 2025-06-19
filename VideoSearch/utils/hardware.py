import torch
try:
    import pynvml
    pynvml.nvmlInit()
    NVML_AVAILABLE = True
except ImportError:
    NVML_AVAILABLE = False


class EmbeddingModelSelector:
    @staticmethod
    def select(command=None):
        def log(msg, level='info'):
            if command:
                command.stdout.write(getattr(command, f'style_{level}')(msg))
            else:
                print(msg)

        if not torch.cuda.is_available():
            log("[Embedding] No GPU found, using CPU-only mode.", "warning")
            return "openai/clip-vit-base-patch32", None, "cpu-only"

        props = torch.cuda.get_device_properties(0)
        total_mem = props.total_memory / (1024 ** 3)

        used_mem = 0
        if NVML_AVAILABLE:
            try:
                handle = pynvml.nvmlDeviceGetHandleByIndex(0)
                meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
                used_mem = meminfo.used / (1024 ** 3)
                log(f"[Embedding] GPU total: {total_mem:.1f} GB | used: {used_mem:.1f} GB", "info")

                if used_mem / total_mem > 0.2:
                    log("[Embedding] Warning: High GPU memory usage detected - model loading might fail.", "warning")
            except Exception:
                log("[Embedding] Could not access GPU memory usage - estimating conservatively.", "warning")
        else:
            log("[Embedding] pynvml not installed - usage stats unavailable.", "warning")

        available = total_mem

        if available >= 5:
            clip = "openai/clip-vit-large-patch14"
            dino = "vit_base_patch16_224_dino"
            mode = "full"
        elif available >= 4:
            clip = "openai/clip-vit-large-patch14"
            dino = "vit_small_patch16_224_dino"
            mode = "full"
        elif available >= 3:
            clip = "openai/clip-vit-large-patch14"
            dino = "vit_tiny_patch16_224_dino"
            mode = "full"
        elif available >= 2:
            clip = "openai/clip-vit-base-patch32"
            dino = "vit_tiny_patch16_224_dino"
            mode = "full"
        elif available >= 1:
            clip = "openai/clip-vit-base-patch32"
            dino = None
            mode = "clip-only"
        else:
            clip = "openai/clip-vit-base-patch32"
            dino = None
            mode = "cpu-only"
            log("[Embedding] Very low VRAM - falling back to CPU-only embedding.", "warning")

        log(f"[Embedding] Selected config: CLIP={clip}, DINO={dino}, Mode={mode}", "info")
        return clip, dino, mode
