"""
Tests for the Trident code generator.
"""

import pytest
from trident.parser import parse
from trident.codegen import compile_to_jax


class TestCodeGenerator:
    """Test JAX code generation."""
    
    def test_empty_program(self):
        """Empty program generates valid Python."""
        ast = parse("")
        code = compile_to_jax(ast)
        
        assert "import jax" in code
        assert "import jax.numpy as jnp" in code
    
    def test_simple_function(self):
        """Simple function generates Python function."""
        source = """fn hello():
    print("Hello, World!")"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "def hello():" in code
        assert "print" in code
    
    def test_function_with_params(self):
        """Function with parameters generates correctly."""
        source = """fn add(a: Int, b: Int) -> Int:
    return a + b"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "def add(a: int, b: int) -> int:" in code
        assert "return (a + b)" in code
    
    def test_let_statement(self):
        """Let statements generate assignments."""
        source = """fn test():
    let x = 42
    let y: Float = 3.14"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "x = 42" in code
        assert "y: float = 3.14" in code
    
    def test_if_statement(self):
        """If statements generate correctly."""
        source = """fn test():
    if x > 0:
        return 1
    else:
        return 0"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "if (x > 0):" in code
        assert "else:" in code
    
    def test_for_loop(self):
        """For loops generate correctly."""
        source = """fn test():
    for i in range(10):
        print(i)"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "for i in range(10):" in code
    
    def test_pipeline_generates_class(self):
        """Pipeline generates as Python class."""
        source = """@intent("Test")
pipeline TestPipeline:
    fn process():
        pass"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "class TestPipeline:" in code
        assert "def __init__(self):" in code
        assert "def process(self):" in code
    
    def test_tensor_literal(self):
        """Tensor literals use jnp.array."""
        source = """fn test():
    let arr = [1, 2, 3, 4]"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "jnp.array([1, 2, 3, 4])" in code
    
    def test_matmul_operator(self):
        """Matrix multiplication uses @ operator."""
        source = """fn test():
    let c = a @ b"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "(a @ b)" in code
    
    def test_binary_operators(self):
        """Binary operators generate correctly."""
        source = """fn test():
    let a = 1 + 2
    let b = 3 - 4
    let c = 5 * 6
    let d = 7 / 8
    let e = 9 ** 2"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "(1 + 2)" in code
        assert "(3 - 4)" in code
        assert "(5 * 6)" in code
        assert "(7 / 8)" in code
        assert "(9 ** 2)" in code
    
    def test_comparison_operators(self):
        """Comparison operators generate correctly."""
        source = """fn test():
    let a = x == y
    let b = x != y
    let c = x < y
    let d = x > y"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "(x == y)" in code
        assert "(x != y)" in code
        assert "(x < y)" in code
        assert "(x > y)" in code
    
    def test_jit_decorator(self):
        """Functions get @jit decorator."""
        source = """fn compute():
    return 42"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "@jit" in code


class TestPrimitiveMapping:
    """Test mapping of Trident primitives to runtime calls."""
    
    def test_vision_read(self):
        """vision.read maps correctly."""
        source = """fn test():
    let img = vision.read("path.jpg")"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "vision.read_image" in code
    
    def test_ocr_extract(self):
        """ocr.extract maps correctly."""
        source = """fn test():
    let text = ocr.extract(image)"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "vision.ocr_extract" in code
    
    def test_llm_query(self):
        """llm.query maps correctly."""
        source = """fn test():
    let result = llm.query(context, "prompt")"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "nlp.llm_query" in code
    
    def test_softmax(self):
        """softmax maps to jax.nn.softmax."""
        source = """fn test():
    let y = softmax(x)"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "jax.nn.softmax" in code
    
    def test_sqrt(self):
        """sqrt maps to jnp.sqrt."""
        source = """fn test():
    let y = sqrt(x)"""
        
        ast = parse(source)
        code = compile_to_jax(ast)
        
        assert "jnp.sqrt" in code
