try:
    from .audio_lens import AudioLens
    __all__ = ["AudioLens"]
except ModuleNotFoundError:
    __all__ = []
