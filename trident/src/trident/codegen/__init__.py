"""
Trident Code Generation - Compiling to JAX/Python.
"""

from trident.codegen.ir import (
    IRNode,
    IRProgram,
    IRFunction,
    IRPipeline,
    IRInstruction,
    IRBlock,
)
from trident.codegen.jax_emitter import JAXEmitter, compile_to_jax
from trident.codegen.optimizer import IROptimizer, optimize

__all__ = [
    # IR
    "IRNode",
    "IRProgram",
    "IRFunction",
    "IRPipeline",
    "IRInstruction",
    "IRBlock",
    # Emitter
    "JAXEmitter",
    "compile_to_jax",
    # Optimizer
    "IROptimizer",
    "optimize",
]
