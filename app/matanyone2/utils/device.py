import torch
import functools
import gc

def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif torch.backends.mps.is_built() and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")

def clean_vram():
    """Aggressively free GPU memory on CUDA and MPS."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if hasattr(torch, 'mps') and hasattr(torch.mps, 'empty_cache'):
        try:
            torch.mps.empty_cache()
        except Exception:
            pass

def safe_autocast_decorator(enabled=True):
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            device = get_default_device()
            if device.type in ["cuda", "cpu"]:
                with torch.amp.autocast(device_type=device.type, enabled=enabled):
                    return func(*args, **kwargs)
            else:
                return func(*args, **kwargs)
        return wrapper
    return decorator

import contextlib
@contextlib.contextmanager
def safe_autocast(enabled=True):
    device = get_default_device()
    if device.type in ["cuda", "cpu"]:
        with torch.amp.autocast(device_type=device.type, enabled=enabled):
            yield
    else:
        yield  # MPS or other unsupported backends skip autocast
