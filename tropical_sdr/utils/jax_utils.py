"""JAX-specific utility functions."""

import jax
import jax.numpy as jnp
from typing import Callable, Any, List, Tuple, Optional
import time
from functools import partial, wraps
import psutil
import GPUtil


def create_rng_keys(seed: int = 0, n_keys: int = 1):
    """Create JAX random keys.
    
    Args:
        seed: Random seed
        n_keys: Number of keys to create
        
    Returns:
        Single key or list of keys
    """
    key = jax.random.PRNGKey(seed)
    
    if n_keys == 1:
        return key
    else:
        keys = jax.random.split(key, n_keys)
        return list(keys)


def batch_process(fn: Callable,
                 data: jnp.ndarray,
                 batch_size: int = 32,
                 axis: int = 0) -> jnp.ndarray:
    """Process data in batches.
    
    Args:
        fn: Function to apply
        data: Input data
        batch_size: Batch size
        axis: Batch axis
        
    Returns:
        Processed data
    """
    n_samples = data.shape[axis]
    n_batches = (n_samples + batch_size - 1) // batch_size
    
    results = []
    for i in range(n_batches):
        start = i * batch_size
        end = min((i + 1) * batch_size, n_samples)
        
        if axis == 0:
            batch = data[start:end]
        elif axis == 1:
            batch = data[:, start:end]
        else:
            raise ValueError(f"Unsupported axis: {axis}")
            
        result = fn(batch)
        results.append(result)
        
    return jnp.concatenate(results, axis=axis)


def profile_function(fn: Callable) -> Callable:
    """Decorator to profile function execution.
    
    Args:
        fn: Function to profile
        
    Returns:
        Wrapped function with profiling
    """
    @wraps(fn)
    def wrapper(*args, **kwargs):
        # CPU usage before
        cpu_before = psutil.cpu_percent()
        
        # GPU usage before (if available)
        try:
            gpus = GPUtil.getGPUs()
            gpu_before = gpus[0].memoryUsed if gpus else 0
        except:
            gpu_before = 0
            
        # Time execution
        start = time.perf_counter()
        result = fn(*args, **kwargs)
        
        # Wait for JAX to finish
        if isinstance(result, jnp.ndarray):
            result.block_until_ready()
            
        elapsed = time.perf_counter() - start
        
        # CPU usage after
        cpu_after = psutil.cpu_percent()
        
        # GPU usage after
        try:
            gpus = GPUtil.getGPUs()
            gpu_after = gpus[0].memoryUsed if gpus else 0
        except:
            gpu_after = 0
            
        print(f"\n{fn.__name__} Profile:")
        print(f"  Time: {elapsed:.3f}s")
        print(f"  CPU: {cpu_before:.1f}% -> {cpu_after:.1f}%")
        if gpu_before > 0 or gpu_after > 0:
            print(f"  GPU Memory: {gpu_before:.0f}MB -> {gpu_after:.0f}MB")
            
        return result
        
    return wrapper


def memory_info():
    """Print current memory usage."""
    # System memory
    mem = psutil.virtual_memory()
    print(f"System Memory: {mem.used/1e9:.2f}GB / {mem.total/1e9:.2f}GB "
          f"({mem.percent:.1f}%)")
    
    # JAX device memory
    for device in jax.devices():
        stats = device.memory_stats()
        if stats:
            used = stats.get('bytes_in_use', 0) / 1e9
            limit = stats.get('bytes_limit', 0) / 1e9
            print(f"{device.platform.upper()} {device.id}: "
                  f"{used:.2f}GB / {limit:.2f}GB")
                  
    # GPU memory (if available)
    try:
        gpus = GPUtil.getGPUs()
        for gpu in gpus:
            print(f"GPU {gpu.id}: {gpu.memoryUsed:.0f}MB / "
                  f"{gpu.memoryTotal:.0f}MB ({gpu.memoryUtil*100:.1f}%)")
    except:
        pass


def jit_compile_all(module):
    """JIT compile all jittable functions in a module.
    
    Args:
        module: Module containing functions to compile
    """
    compiled = []
    
    for name in dir(module):
        obj = getattr(module, name)
        
        # Check if it's a function that can be jitted
        if callable(obj) and not name.startswith('_'):
            try:
                # Try to JIT compile
                jitted = jax.jit(obj)
                setattr(module, name, jitted)
                compiled.append(name)
            except:
                # Some functions may not be jittable
                pass
                
    print(f"JIT compiled {len(compiled)} functions: {', '.join(compiled)}")


@partial(jax.jit, static_argnames=['padding'])
def pad_to_shape(array: jnp.ndarray, 
                 target_shape: Tuple[int, ...],
                 padding: str = 'constant') -> jnp.ndarray:
    """Pad array to target shape.
    
    Args:
        array: Input array
        target_shape: Target shape
        padding: Padding mode
        
    Returns:
        Padded array
    """
    pad_widths = []
    for current, target in zip(array.shape, target_shape):
        total_pad = max(0, target - current)
        pad_before = total_pad // 2
        pad_after = total_pad - pad_before
        pad_widths.append((pad_before, pad_after))
        
    return jnp.pad(array, pad_widths, mode=padding)


@jax.jit
def normalize(array: jnp.ndarray, 
             axis: Optional[int] = None) -> jnp.ndarray:
    """Normalize array to [0, 1] range.
    
    Args:
        array: Input array
        axis: Axis to normalize along
        
    Returns:
        Normalized array
    """
    min_val = jnp.min(array, axis=axis, keepdims=True)
    max_val = jnp.max(array, axis=axis, keepdims=True)
    
    return (array - min_val) / (max_val - min_val + 1e-10)


def vmap_over_batch(fn: Callable,
                   in_axes: int = 0,
                   out_axes: int = 0) -> Callable:
    """Decorator to vmap function over batch dimension.
    
    Args:
        fn: Function to vmap
        in_axes: Input axes to map over
        out_axes: Output axes
        
    Returns:
        Vmapped function
    """
    return jax.vmap(fn, in_axes=in_axes, out_axes=out_axes)


class JAXTimer:
    """Context manager for timing JAX operations."""
    
    def __init__(self, name: str = "Operation"):
        self.name = name
        self.start = None
        self.elapsed = None
        
    def __enter__(self):
        self.start = time.perf_counter()
        return self
        
    def __exit__(self, *args):
        # Make sure JAX operations complete
        jax.random.normal(jax.random.PRNGKey(0), ()).block_until_ready()
        
        self.elapsed = time.perf_counter() - self.start
        print(f"{self.name}: {self.elapsed:.3f}s")