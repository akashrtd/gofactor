"""
Intermediate Representation (IR) for Trident.

The IR is a lower-level representation than the AST, closer to the
target execution model (JAX). It facilitates:
1. Optimization passes
2. Code generation
3. Backend-agnostic transformations

IR Structure:
    IRProgram
    ├── IRPipeline[]
    │   ├── ModelBindings
    │   └── IRFunction[]
    └── IRFunction[]
        └── IRBlock[]
            └── IRInstruction[]
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional, Sequence
from enum import Enum, auto


class OpCode(Enum):
    """IR operation codes."""
    # Control flow
    JUMP = auto()
    BRANCH = auto()
    RETURN = auto()
    CALL = auto()
    
    # Memory
    LOAD = auto()
    STORE = auto()
    ALLOC = auto()
    
    # Arithmetic
    ADD = auto()
    SUB = auto()
    MUL = auto()
    DIV = auto()
    MOD = auto()
    POW = auto()
    NEG = auto()
    
    # Tensor operations
    MATMUL = auto()
    TRANSPOSE = auto()
    RESHAPE = auto()
    BROADCAST = auto()
    REDUCE = auto()
    SLICE = auto()
    
    # Comparison
    EQ = auto()
    NE = auto()
    LT = auto()
    GT = auto()
    LE = auto()
    GE = auto()
    
    # Logical
    AND = auto()
    OR = auto()
    NOT = auto()
    
    # AI primitives
    OCR_EXTRACT = auto()
    OCR_LAYOUT = auto()
    LLM_QUERY = auto()
    LLM_EMBED = auto()
    VISION_READ = auto()
    VISION_PREPROCESS = auto()
    
    # Special
    NOP = auto()
    PHI = auto()  # SSA phi node
    CONSTANT = auto()


@dataclass(frozen=True)
class IRType:
    """IR type representation."""
    name: str
    shape: tuple[int | str, ...] = ()
    dtype: str = "float32"
    
    def __str__(self) -> str:
        if self.shape:
            shape_str = "x".join(str(s) for s in self.shape)
            return f"{self.name}[{self.dtype}, {shape_str}]"
        return self.name


# Common IR types
IR_INT = IRType("int")
IR_FLOAT = IRType("float")
IR_STRING = IRType("string")
IR_BOOL = IRType("bool")
IR_NONE = IRType("none")
IR_TENSOR = IRType("tensor")
IR_IMAGE = IRType("image")
IR_DOCUMENT = IRType("document")


@dataclass
class IRNode(ABC):
    """Base class for all IR nodes."""
    
    @abstractmethod
    def __str__(self) -> str:
        pass


@dataclass
class IRValue:
    """Represents a value (variable, constant, or temporary)."""
    name: str
    ir_type: IRType
    is_const: bool = False
    const_value: Any = None
    
    def __str__(self) -> str:
        if self.is_const:
            return f"const({self.const_value})"
        return f"%{self.name}"


@dataclass
class IRInstruction(IRNode):
    """
    A single IR instruction.
    
    Format: result = opcode(operands)
    """
    opcode: OpCode
    result: Optional[IRValue] = None
    operands: tuple[IRValue, ...] = ()
    attributes: dict[str, Any] = field(default_factory=dict)
    
    def __str__(self) -> str:
        operands_str = ", ".join(str(op) for op in self.operands)
        attrs_str = ""
        if self.attributes:
            attrs_str = " {" + ", ".join(f"{k}={v}" for k, v in self.attributes.items()) + "}"
        
        if self.result:
            return f"{self.result} = {self.opcode.name}({operands_str}){attrs_str}"
        return f"{self.opcode.name}({operands_str}){attrs_str}"


@dataclass
class IRBlock(IRNode):
    """
    A basic block of instructions.
    
    Basic blocks have a single entry point (label) and single exit.
    """
    label: str
    instructions: list[IRInstruction] = field(default_factory=list)
    predecessors: list[str] = field(default_factory=list)
    successors: list[str] = field(default_factory=list)
    
    def add_instruction(self, inst: IRInstruction) -> None:
        """Add an instruction to the block."""
        self.instructions.append(inst)
    
    def is_terminated(self) -> bool:
        """Check if block ends with a terminator instruction."""
        if not self.instructions:
            return False
        last = self.instructions[-1]
        return last.opcode in (OpCode.JUMP, OpCode.BRANCH, OpCode.RETURN)
    
    def __str__(self) -> str:
        lines = [f"{self.label}:"]
        for inst in self.instructions:
            lines.append(f"  {inst}")
        return "\n".join(lines)


@dataclass
class IRFunction(IRNode):
    """
    An IR function definition.
    """
    name: str
    params: tuple[IRValue, ...] = ()
    return_type: IRType = IR_NONE
    blocks: list[IRBlock] = field(default_factory=list)
    is_jit: bool = True  # Whether to JIT compile
    
    # Annotations from source
    annotations: dict[str, Any] = field(default_factory=dict)
    
    @property
    def entry_block(self) -> Optional[IRBlock]:
        """Get the entry block."""
        return self.blocks[0] if self.blocks else None
    
    def add_block(self, label: str) -> IRBlock:
        """Create and add a new block."""
        block = IRBlock(label=label)
        self.blocks.append(block)
        return block
    
    def __str__(self) -> str:
        params_str = ", ".join(f"{p.name}: {p.ir_type}" for p in self.params)
        jit_str = "@jit " if self.is_jit else ""
        
        lines = [f"{jit_str}fn {self.name}({params_str}) -> {self.return_type}:"]
        for block in self.blocks:
            lines.append(str(block))
        return "\n".join(lines)


@dataclass
class IRPipeline(IRNode):
    """
    An IR pipeline definition.
    
    Pipelines group functions with AI model bindings.
    """
    name: str
    functions: list[IRFunction] = field(default_factory=list)
    model_bindings: dict[str, str] = field(default_factory=dict)  # category -> model name
    hardware_config: dict[str, Any] = field(default_factory=dict)
    intent: str = ""
    
    def __str__(self) -> str:
        lines = [f"pipeline {self.name}:"]
        if self.intent:
            lines.append(f"  # intent: {self.intent}")
        if self.model_bindings:
            lines.append(f"  # models: {self.model_bindings}")
        for func in self.functions:
            for line in str(func).split("\n"):
                lines.append(f"  {line}")
        return "\n".join(lines)


@dataclass
class IRProgram(IRNode):
    """
    The top-level IR program.
    """
    name: str = "program"
    pipelines: list[IRPipeline] = field(default_factory=list)
    functions: list[IRFunction] = field(default_factory=list)
    globals: dict[str, IRValue] = field(default_factory=dict)
    imports: list[str] = field(default_factory=list)
    
    def __str__(self) -> str:
        lines = [f"# IR Program: {self.name}", ""]
        
        if self.imports:
            for imp in self.imports:
                lines.append(f"import {imp}")
            lines.append("")
        
        if self.globals:
            lines.append("# Globals")
            for name, val in self.globals.items():
                lines.append(f"global {name}: {val.ir_type}")
            lines.append("")
        
        for pipeline in self.pipelines:
            lines.append(str(pipeline))
            lines.append("")
        
        for func in self.functions:
            lines.append(str(func))
            lines.append("")
        
        return "\n".join(lines)


# =============================================================================
# IR Builder
# =============================================================================

@dataclass
class IRBuilder:
    """
    Helper for building IR from AST.
    
    Provides convenience methods for creating IR nodes and managing
    control flow.
    """
    
    program: IRProgram = field(default_factory=IRProgram)
    _current_function: Optional[IRFunction] = field(default=None, init=False)
    _current_block: Optional[IRBlock] = field(default=None, init=False)
    _temp_counter: int = field(default=0, init=False)
    
    def create_function(self, name: str, params: list[tuple[str, IRType]], 
                        return_type: IRType = IR_NONE) -> IRFunction:
        """Create a new function."""
        ir_params = tuple(
            IRValue(name=pname, ir_type=ptype)
            for pname, ptype in params
        )
        func = IRFunction(
            name=name,
            params=ir_params,
            return_type=return_type,
        )
        self.program.functions.append(func)
        return func
    
    def create_pipeline(self, name: str) -> IRPipeline:
        """Create a new pipeline."""
        pipeline = IRPipeline(name=name)
        self.program.pipelines.append(pipeline)
        return pipeline
    
    def set_function(self, func: IRFunction) -> None:
        """Set current function context."""
        self._current_function = func
    
    def create_block(self, label: str) -> IRBlock:
        """Create a new block in current function."""
        if not self._current_function:
            raise RuntimeError("No current function")
        return self._current_function.add_block(label)
    
    def set_block(self, block: IRBlock) -> None:
        """Set current block for instruction emission."""
        self._current_block = block
    
    def emit(self, opcode: OpCode, operands: list[IRValue],
             result_type: Optional[IRType] = None,
             result_name: Optional[str] = None,
             **attrs: Any) -> Optional[IRValue]:
        """Emit an instruction and return the result value."""
        if not self._current_block:
            raise RuntimeError("No current block")
        
        result = None
        if result_type:
            name = result_name or self._next_temp()
            result = IRValue(name=name, ir_type=result_type)
        
        inst = IRInstruction(
            opcode=opcode,
            result=result,
            operands=tuple(operands),
            attributes=attrs,
        )
        self._current_block.add_instruction(inst)
        return result
    
    def emit_constant(self, value: Any, ir_type: IRType) -> IRValue:
        """Create a constant value."""
        return IRValue(
            name=f"const_{self._temp_counter}",
            ir_type=ir_type,
            is_const=True,
            const_value=value,
        )
    
    def emit_return(self, value: Optional[IRValue] = None) -> None:
        """Emit a return instruction."""
        operands = [value] if value else []
        self.emit(OpCode.RETURN, operands)
    
    def emit_jump(self, target: str) -> None:
        """Emit an unconditional jump."""
        self.emit(OpCode.JUMP, [], target=target)
    
    def emit_branch(self, cond: IRValue, true_target: str, false_target: str) -> None:
        """Emit a conditional branch."""
        self.emit(OpCode.BRANCH, [cond], true_target=true_target, false_target=false_target)
    
    def _next_temp(self) -> str:
        """Generate next temporary name."""
        name = f"t{self._temp_counter}"
        self._temp_counter += 1
        return name
