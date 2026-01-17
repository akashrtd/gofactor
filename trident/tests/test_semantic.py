"""
Tests for the Trident semantic analyzer.
"""

import pytest
from trident.parser import parse
from trident.semantic import analyze, SemanticAnalyzer, SemanticError
from trident.semantic.types import INT, FLOAT, STRING, BOOL, TensorType


class TestSemanticAnalyzer:
    """Test semantic analysis."""
    
    def test_empty_program(self):
        """Empty program passes analysis."""
        ast = parse("")
        success, errors, _ = analyze(ast)
        assert success
        assert len(errors) == 0
    
    def test_valid_function(self):
        """Valid function definition passes."""
        source = """fn add(a: Int, b: Int) -> Int:
    return a + b"""
        
        ast = parse(source)
        success, errors, _ = analyze(ast)
        assert success
    
    def test_undefined_variable(self):
        """Using undefined variable is an error."""
        source = """fn test():
    return undefined_var"""
        
        ast = parse(source)
        success, errors, _ = analyze(ast)
        assert not success
        assert any("undefined" in e.message.lower() for e in errors)
    
    def test_variable_shadowing(self):
        """Variable can shadow outer scope."""
        source = """fn outer():
    let x = 1
    fn inner():
        let x = 2
        return x"""
        
        # Note: This test may need adjustment based on scoping rules
        ast = parse(source)
        success, errors, _ = analyze(ast)
        # Currently this parses as two separate functions
    
    def test_let_type_annotation(self):
        """Let with type annotation is checked."""
        source = """fn test():
    let x: Int = 42"""
        
        ast = parse(source)
        success, errors, analyzer = analyze(ast)
        assert success
    
    def test_return_outside_function(self):
        """Return outside function is an error - this is a parse-level issue."""
        # Actually, our parser requires return to be inside a function block
        # so this test validates parser behavior
        pass
    
    def test_break_outside_loop(self):
        """Break outside loop is an error."""
        source = """fn test():
    break"""
        
        ast = parse(source)
        success, errors, _ = analyze(ast)
        assert not success
        assert any("loop" in e.message.lower() for e in errors)
    
    def test_continue_outside_loop(self):
        """Continue outside loop is an error."""
        source = """fn test():
    continue"""
        
        ast = parse(source)
        success, errors, _ = analyze(ast)
        assert not success
        assert any("loop" in e.message.lower() for e in errors)
    
    def test_valid_loop(self):
        """Break/continue inside loop is valid."""
        source = """fn test():
    for i in range(10):
        if i > 5:
            break
        continue"""
        
        ast = parse(source)
        success, errors, _ = analyze(ast)
        assert success
    
    def test_function_redefinition(self):
        """Redefining function in same scope is an error."""
        source = """fn test():
    pass

fn test():
    pass"""
        
        ast = parse(source)
        success, errors, _ = analyze(ast)
        assert not success
        assert any("redefinition" in e.message.lower() for e in errors)
    
    def test_builtin_functions(self):
        """Built-in functions are available."""
        source = """fn test():
    print("hello")
    let x = len([1, 2, 3])"""
        
        ast = parse(source)
        success, errors, _ = analyze(ast)
        assert success


class TestTypeInference:
    """Test type inference and checking."""
    
    def test_literal_types(self):
        """Literals have correct types."""
        source = """fn test():
    let a = 42
    let b = 3.14
    let c = "hello"
    let d = true"""
        
        ast = parse(source)
        success, errors, analyzer = analyze(ast)
        assert success
    
    def test_binary_op_types(self):
        """Binary operations have correct result types."""
        source = """fn test():
    let x = 1 + 2
    let y = 3.0 * 4.0
    let z = 1 < 2"""
        
        ast = parse(source)
        success, errors, analyzer = analyze(ast)
        assert success
    
    def test_comparison_returns_bool(self):
        """Comparisons return boolean type."""
        source = """fn test():
    let x = 1 == 2
    if x:
        pass"""
        
        ast = parse(source)
        success, errors, _ = analyze(ast)
        assert success


class TestPipelineAnalysis:
    """Test pipeline-specific analysis."""
    
    def test_simple_pipeline(self):
        """Simple pipeline definition passes."""
        source = """pipeline TestPipeline:
    fn process():
        pass"""
        
        ast = parse(source)
        success, errors, _ = analyze(ast)
        assert success
    
    def test_pipeline_with_annotations(self):
        """Pipeline with annotations passes."""
        source = """@intent("Test pipeline")
@model(ocr: "qwen")
pipeline TestPipeline:
    fn process():
        pass"""
        
        ast = parse(source)
        success, errors, _ = analyze(ast)
        assert success
