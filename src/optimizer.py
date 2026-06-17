GPU_AVAILABLE = False

try:
    import cupy as cp
    cp.cuda.Device(0).use()
    from .optimizer_gpu import run_cvar_optimization
    GPU_AVAILABLE = True
except Exception:
    from .optimizer_cpu import run_cvar_optimization

__all__ = ["run_cvar_optimization", "GPU_AVAILABLE"]
