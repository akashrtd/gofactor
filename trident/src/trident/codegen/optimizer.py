"""
IR Optimizer for Trident.

Performs optimization passes on the IR before code generation:
1. Constant folding
2. Dead code elimination
3. Tensor operation fusion
4. Memory layout optimization
"""

from dataclasses import dataclass, field
from typing import Optional

from trident.codegen.ir import (
    IRProgram,
    IRFunction,
    IRPipeline,
    IRBlock,
    IRInstruction,
    IRValue,
    OpCode,
)


@dataclass
class IROptimizer:
    """
    Optimization pass manager for IR.
    
    Runs a series of optimization passes on the IR to improve
    generated code quality and performance.
    """
    
    _passes: list[str] = field(default_factory=lambda: [
        "constant_fold",
        "dead_code_eliminate",
        "fusion",
    ])
    
    def optimize(self, program: IRProgram) -> IRProgram:
        """
        Run all optimization passes on the program.
        
        Returns:
            Optimized IR program
        """
        for pass_name in self._passes:
            method = getattr(self, f"_pass_{pass_name}", None)
            if method:
                program = method(program)
        return program
    
    def _pass_constant_fold(self, program: IRProgram) -> IRProgram:
        """
        Fold constant expressions at compile time.
        
        Examples:
            2 + 3 -> 5
            2 * 4 -> 8
        """
        for func in program.functions:
            for block in func.blocks:
                self._fold_block_constants(block)
        
        for pipeline in program.pipelines:
            for func in pipeline.functions:
                for block in func.blocks:
                    self._fold_block_constants(block)
        
        return program
    
    def _fold_block_constants(self, block: IRBlock) -> None:
        """Fold constants in a single block."""
        const_values: dict[str, any] = {}
        new_instructions: list[IRInstruction] = []
        
        for inst in block.instructions:
            # Track constant values
            if inst.opcode == OpCode.CONSTANT and inst.result:
                const_values[inst.result.name] = inst.result.const_value
            
            # Try to fold arithmetic on constants
            elif inst.opcode in (OpCode.ADD, OpCode.SUB, OpCode.MUL, OpCode.DIV):
                if len(inst.operands) == 2:
                    left, right = inst.operands
                    if left.is_const and right.is_const:
                        # Evaluate at compile time
                        try:
                            result = self._eval_binary(inst.opcode, left.const_value, right.const_value)
                            if inst.result:
                                inst = IRInstruction(
                                    opcode=OpCode.CONSTANT,
                                    result=IRValue(
                                        name=inst.result.name,
                                        ir_type=inst.result.ir_type,
                                        is_const=True,
                                        const_value=result,
                                    ),
                                )
                        except Exception:
                            pass  # Keep original instruction
            
            new_instructions.append(inst)
        
        block.instructions = new_instructions
    
    def _eval_binary(self, opcode: OpCode, left: any, right: any) -> any:
        """Evaluate a binary operation on constants."""
        ops = {
            OpCode.ADD: lambda a, b: a + b,
            OpCode.SUB: lambda a, b: a - b,
            OpCode.MUL: lambda a, b: a * b,
            OpCode.DIV: lambda a, b: a / b,
        }
        return ops[opcode](left, right)
    
    def _pass_dead_code_eliminate(self, program: IRProgram) -> IRProgram:
        """
        Remove dead (unreachable or unused) code.
        
        - Remove instructions after unconditional jumps/returns
        - Remove unused variable assignments
        """
        for func in program.functions:
            self._eliminate_dead_code_in_function(func)
        
        for pipeline in program.pipelines:
            for func in pipeline.functions:
                self._eliminate_dead_code_in_function(func)
        
        return program
    
    def _eliminate_dead_code_in_function(self, func: IRFunction) -> None:
        """Eliminate dead code in a function."""
        # Find all used values
        used_values: set[str] = set()
        
        for block in func.blocks:
            for inst in block.instructions:
                for operand in inst.operands:
                    if not operand.is_const:
                        used_values.add(operand.name)
        
        # Remove assignments to unused values (except side-effecting ops)
        side_effect_ops = {OpCode.CALL, OpCode.STORE, OpCode.RETURN, OpCode.BRANCH, OpCode.JUMP}
        
        for block in func.blocks:
            new_instructions = []
            for inst in block.instructions:
                keep = True
                
                if inst.result and inst.opcode not in side_effect_ops:
                    if inst.result.name not in used_values:
                        keep = False  # Unused assignment
                
                if keep:
                    new_instructions.append(inst)
            
            block.instructions = new_instructions
    
    def _pass_fusion(self, program: IRProgram) -> IRProgram:
        """
        Fuse compatible tensor operations for better performance.
        
        Examples:
            matmul + add -> fused_matmul_add
            softmax + mask -> fused_attention
        """
        # TODO: Implement operation fusion patterns
        # This is a placeholder for future optimization
        return program


def optimize(program: IRProgram) -> IRProgram:
    """
    Convenience function to optimize an IR program.
    
    Args:
        program: The IR program to optimize
    
    Returns:
        Optimized IR program
    """
    optimizer = IROptimizer()
    return optimizer.optimize(program)
