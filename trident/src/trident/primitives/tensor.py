"""
Tensor primitives for Trident.

Provides high-level tensor operations that compile to efficient JAX code.
All operations are designed to be TPU-friendly.
"""

from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from functools import wraps

# Try to import JAX, fall back to numpy
try:
    import jax
    import jax.numpy as jnp
    from jax import jit, vmap, pmap, grad
    HAS_JAX = True
except ImportError:
    import numpy as jnp
    HAS_JAX = False
    
    # Dummy decorators
    def jit(fn): return fn
    def vmap(fn, *args, **kwargs): return fn
    def pmap(fn, *args, **kwargs): return fn
    def grad(fn, *args, **kwargs): return fn


# Type alias for arrays
Array = Any


# =============================================================================
# Array Creation
# =============================================================================

def zeros(shape: Union[int, Tuple[int, ...]], dtype: str = "float32") -> Array:
    """Create array of zeros."""
    if isinstance(shape, int):
        shape = (shape,)
    return jnp.zeros(shape, dtype=dtype)


def ones(shape: Union[int, Tuple[int, ...]], dtype: str = "float32") -> Array:
    """Create array of ones."""
    if isinstance(shape, int):
        shape = (shape,)
    return jnp.ones(shape, dtype=dtype)


def full(shape: Union[int, Tuple[int, ...]], value: float, dtype: str = "float32") -> Array:
    """Create array filled with value."""
    if isinstance(shape, int):
        shape = (shape,)
    return jnp.full(shape, value, dtype=dtype)


def arange(start: int, stop: Optional[int] = None, step: int = 1) -> Array:
    """Create array with evenly spaced values."""
    if stop is None:
        return jnp.arange(start)
    return jnp.arange(start, stop, step)


def linspace(start: float, stop: float, num: int = 50) -> Array:
    """Create array with linearly spaced values."""
    return jnp.linspace(start, stop, num)


def randn(*shape: int, key: Optional[Any] = None) -> Array:
    """Create array with random normal values."""
    if HAS_JAX:
        if key is None:
            key = jax.random.PRNGKey(0)
        return jax.random.normal(key, shape)
    else:
        import numpy as np
        return np.random.randn(*shape).astype(jnp.float32)


def rand(*shape: int, key: Optional[Any] = None) -> Array:
    """Create array with random uniform values [0, 1)."""
    if HAS_JAX:
        if key is None:
            key = jax.random.PRNGKey(0)
        return jax.random.uniform(key, shape)
    else:
        import numpy as np
        return np.random.rand(*shape).astype(jnp.float32)


def eye(n: int, dtype: str = "float32") -> Array:
    """Create identity matrix."""
    return jnp.eye(n, dtype=dtype)


def from_list(data: List[Any]) -> Array:
    """Create array from Python list."""
    return jnp.array(data)


# =============================================================================
# Mathematical Operations
# =============================================================================

def matmul(a: Array, b: Array) -> Array:
    """Matrix multiplication."""
    return jnp.matmul(a, b)


def dot(a: Array, b: Array) -> Array:
    """Dot product."""
    return jnp.dot(a, b)


def transpose(a: Array, axes: Optional[Tuple[int, ...]] = None) -> Array:
    """Transpose array."""
    return jnp.transpose(a, axes)


def reshape(a: Array, shape: Tuple[int, ...]) -> Array:
    """Reshape array."""
    return jnp.reshape(a, shape)


def squeeze(a: Array, axis: Optional[int] = None) -> Array:
    """Remove single-dimensional entries."""
    return jnp.squeeze(a, axis)


def expand_dims(a: Array, axis: int) -> Array:
    """Expand array dimensions."""
    return jnp.expand_dims(a, axis)


def concatenate(arrays: Sequence[Array], axis: int = 0) -> Array:
    """Concatenate arrays."""
    return jnp.concatenate(arrays, axis)


def stack(arrays: Sequence[Array], axis: int = 0) -> Array:
    """Stack arrays along new axis."""
    return jnp.stack(arrays, axis)


def split(a: Array, indices_or_sections: Union[int, Sequence[int]], axis: int = 0) -> List[Array]:
    """Split array into sub-arrays."""
    return list(jnp.split(a, indices_or_sections, axis))


# =============================================================================
# Reduction Operations
# =============================================================================

def sum(a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """Sum of array elements."""
    return jnp.sum(a, axis=axis, keepdims=keepdims)


def mean(a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """Mean of array elements."""
    return jnp.mean(a, axis=axis, keepdims=keepdims)


def max(a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """Maximum of array elements."""
    return jnp.max(a, axis=axis, keepdims=keepdims)


def min(a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """Minimum of array elements."""
    return jnp.min(a, axis=axis, keepdims=keepdims)


def prod(a: Array, axis: Optional[int] = None, keepdims: bool = False) -> Array:
    """Product of array elements."""
    return jnp.prod(a, axis=axis, keepdims=keepdims)


def argmax(a: Array, axis: Optional[int] = None) -> Array:
    """Indices of maximum values."""
    return jnp.argmax(a, axis=axis)


def argmin(a: Array, axis: Optional[int] = None) -> Array:
    """Indices of minimum values."""
    return jnp.argmin(a, axis=axis)


def norm(a: Array, ord: Optional[int] = None, axis: Optional[int] = None) -> Array:
    """Vector/matrix norm."""
    return jnp.linalg.norm(a, ord=ord, axis=axis)


# =============================================================================
# Element-wise Operations
# =============================================================================

def exp(a: Array) -> Array:
    """Element-wise exponential."""
    return jnp.exp(a)


def log(a: Array) -> Array:
    """Element-wise natural logarithm."""
    return jnp.log(a)


def sqrt(a: Array) -> Array:
    """Element-wise square root."""
    return jnp.sqrt(a)


def abs(a: Array) -> Array:
    """Element-wise absolute value."""
    return jnp.abs(a)


def sin(a: Array) -> Array:
    """Element-wise sine."""
    return jnp.sin(a)


def cos(a: Array) -> Array:
    """Element-wise cosine."""
    return jnp.cos(a)


def tanh(a: Array) -> Array:
    """Element-wise hyperbolic tangent."""
    return jnp.tanh(a)


def clip(a: Array, min_val: float, max_val: float) -> Array:
    """Clip values to range."""
    return jnp.clip(a, min_val, max_val)


def where(condition: Array, x: Array, y: Array) -> Array:
    """Element-wise conditional selection."""
    return jnp.where(condition, x, y)


# =============================================================================
# Neural Network Operations
# =============================================================================

def softmax(a: Array, axis: int = -1) -> Array:
    """Softmax activation."""
    if HAS_JAX:
        return jax.nn.softmax(a, axis=axis)
    else:
        exp_a = jnp.exp(a - jnp.max(a, axis=axis, keepdims=True))
        return exp_a / jnp.sum(exp_a, axis=axis, keepdims=True)


def log_softmax(a: Array, axis: int = -1) -> Array:
    """Log-softmax."""
    if HAS_JAX:
        return jax.nn.log_softmax(a, axis=axis)
    else:
        return jnp.log(softmax(a, axis))


def relu(a: Array) -> Array:
    """ReLU activation."""
    if HAS_JAX:
        return jax.nn.relu(a)
    else:
        return jnp.maximum(0, a)


def gelu(a: Array) -> Array:
    """GELU activation."""
    if HAS_JAX:
        return jax.nn.gelu(a)
    else:
        return 0.5 * a * (1 + jnp.tanh(jnp.sqrt(2 / jnp.pi) * (a + 0.044715 * a**3)))


def sigmoid(a: Array) -> Array:
    """Sigmoid activation."""
    if HAS_JAX:
        return jax.nn.sigmoid(a)
    else:
        return 1 / (1 + jnp.exp(-a))


def layer_norm(a: Array, axis: int = -1, eps: float = 1e-5) -> Array:
    """Layer normalization."""
    mean_val = jnp.mean(a, axis=axis, keepdims=True)
    var_val = jnp.var(a, axis=axis, keepdims=True)
    return (a - mean_val) / jnp.sqrt(var_val + eps)


def batch_norm(a: Array, axis: int = 0, eps: float = 1e-5) -> Array:
    """Batch normalization (inference mode)."""
    mean_val = jnp.mean(a, axis=axis, keepdims=True)
    var_val = jnp.var(a, axis=axis, keepdims=True)
    return (a - mean_val) / jnp.sqrt(var_val + eps)


def dropout(a: Array, rate: float = 0.5, key: Optional[Any] = None, training: bool = True) -> Array:
    """Dropout regularization."""
    if not training or rate == 0:
        return a
    
    if HAS_JAX:
        if key is None:
            key = jax.random.PRNGKey(0)
        mask = jax.random.bernoulli(key, 1 - rate, a.shape)
        return a * mask / (1 - rate)
    else:
        import numpy as np
        mask = np.random.binomial(1, 1 - rate, a.shape)
        return a * mask / (1 - rate)


# =============================================================================
# Attention Mechanisms
# =============================================================================

def scaled_dot_product_attention(
    query: Array,
    key: Array,
    value: Array,
    mask: Optional[Array] = None,
) -> Array:
    """
    Scaled dot-product attention.
    
    Args:
        query: Query tensor of shape [..., seq_len, d_k]
        key: Key tensor of shape [..., seq_len, d_k]
        value: Value tensor of shape [..., seq_len, d_v]
        mask: Optional attention mask
    
    Returns:
        Attention output of shape [..., seq_len, d_v]
    """
    d_k = query.shape[-1]
    
    # Compute attention scores
    scores = jnp.matmul(query, jnp.swapaxes(key, -2, -1)) / jnp.sqrt(d_k)
    
    # Apply mask
    if mask is not None:
        scores = jnp.where(mask, scores, jnp.finfo(scores.dtype).min)
    
    # Softmax and apply to values
    attn_weights = softmax(scores, axis=-1)
    return jnp.matmul(attn_weights, value)


def multi_head_attention(
    query: Array,
    key: Array,
    value: Array,
    num_heads: int,
    mask: Optional[Array] = None,
) -> Array:
    """
    Multi-head attention.
    
    Splits inputs into multiple heads, applies attention, and concatenates.
    """
    batch_size, seq_len, d_model = query.shape
    d_k = d_model // num_heads
    
    # Reshape for multi-head
    def split_heads(x):
        return reshape(x, (batch_size, seq_len, num_heads, d_k)).swapaxes(1, 2)
    
    q = split_heads(query)
    k = split_heads(key)
    v = split_heads(value)
    
    # Apply attention per head
    attn_output = scaled_dot_product_attention(q, k, v, mask)
    
    # Concatenate heads
    attn_output = attn_output.swapaxes(1, 2)
    return reshape(attn_output, (batch_size, seq_len, d_model))


# =============================================================================
# JIT Compilation Helpers
# =============================================================================

def compile(fn: Callable) -> Callable:
    """JIT compile a function for TPU/GPU execution."""
    if HAS_JAX:
        return jit(fn)
    return fn


def vectorize(fn: Callable, in_axes: int = 0, out_axes: int = 0) -> Callable:
    """Vectorize a function over a batch dimension."""
    if HAS_JAX:
        return vmap(fn, in_axes=in_axes, out_axes=out_axes)
    return fn


def parallelize(fn: Callable, axis_name: str = "batch") -> Callable:
    """Parallelize a function across devices."""
    if HAS_JAX:
        return pmap(fn, axis_name=axis_name)
    return fn


def gradient(fn: Callable, argnums: int = 0) -> Callable:
    """Get gradient function."""
    if HAS_JAX:
        return grad(fn, argnums=argnums)
    raise NotImplementedError("Gradients require JAX")
