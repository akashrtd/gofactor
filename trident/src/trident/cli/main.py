"""
Main CLI entry point for Trident.

Provides commands for:
- Running Trident programs
- Compiling to JAX
- Starting the REPL
- Checking/formatting code
"""

import sys
from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

console = Console()


@click.group()
@click.version_option(version="0.1.0", prog_name="trident")
def cli():
    """🔱 Trident - A tri-modal programming language for AI-Human-TPU communication."""
    pass


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--debug", is_flag=True, help="Enable debug output")
def run(file: str, debug: bool):
    """Run a Trident program."""
    from trident.parser import parse
    from trident.semantic import analyze, SemanticError
    from trident.codegen import compile_to_jax
    from trident.runtime import TridentRuntime
    
    filepath = Path(file)
    
    console.print(f"[bold blue]🔱 Running:[/] {filepath.name}")
    
    try:
        # Read source
        source = filepath.read_text()
        
        if debug:
            console.print("[dim]Parsing...[/]")
        
        # Parse
        ast = parse(source, str(filepath))
        
        if debug:
            console.print("[dim]Analyzing...[/]")
        
        # Semantic analysis
        success, errors, analyzer = analyze(ast)
        
        if not success:
            console.print("[bold red]Semantic errors:[/]")
            for error in errors:
                console.print(f"  {error.location}: {error.message}")
            sys.exit(1)
        
        if debug:
            console.print("[dim]Compiling to JAX...[/]")
        
        # Compile to JAX
        jax_code = compile_to_jax(ast)
        
        if debug:
            console.print(Panel(Syntax(jax_code, "python", theme="monokai"), title="Generated JAX"))
        
        # Execute
        runtime = TridentRuntime()
        
        # Create temporary module
        import types
        module = types.ModuleType("__trident_main__")
        module.__dict__["__trident_runtime__"] = runtime
        
        exec(jax_code, module.__dict__)
        
        console.print("[bold green]✓ Completed[/]")
        
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        if debug:
            import traceback
            traceback.print_exc()
        sys.exit(1)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("-o", "--output", type=click.Path(), help="Output file path")
@click.option("--print", "print_output", is_flag=True, help="Print to stdout")
def compile(file: str, output: Optional[str], print_output: bool):
    """Compile a Trident program to JAX/Python."""
    from trident.parser import parse
    from trident.semantic import analyze
    from trident.codegen import compile_to_jax
    
    filepath = Path(file)
    
    console.print(f"[bold blue]🔱 Compiling:[/] {filepath.name}")
    
    try:
        source = filepath.read_text()
        ast = parse(source, str(filepath))
        
        success, errors, _ = analyze(ast)
        if not success:
            for error in errors:
                console.print(f"[red]{error.location}: {error.message}[/]")
            sys.exit(1)
        
        jax_code = compile_to_jax(ast)
        
        if print_output:
            console.print(Syntax(jax_code, "python", theme="monokai"))
        elif output:
            Path(output).write_text(jax_code)
            console.print(f"[green]✓ Written to {output}[/]")
        else:
            # Default output path
            out_path = filepath.with_suffix(".py")
            out_path.write_text(jax_code)
            console.print(f"[green]✓ Written to {out_path}[/]")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
def check(file: str):
    """Type-check a Trident program without running it."""
    from trident.parser import parse
    from trident.semantic import analyze
    
    filepath = Path(file)
    
    console.print(f"[bold blue]🔱 Checking:[/] {filepath.name}")
    
    try:
        source = filepath.read_text()
        ast = parse(source, str(filepath))
        
        success, errors, analyzer = analyze(ast)
        
        if errors:
            console.print(f"[bold red]Found {len(errors)} error(s):[/]")
            for error in errors:
                console.print(f"  [red]{error.location}:[/] {error.message}")
            sys.exit(1)
        else:
            console.print("[bold green]✓ No errors found[/]")
            
    except Exception as e:
        console.print(f"[bold red]Error:[/] {e}")
        sys.exit(1)


@cli.command()
@click.argument("file", type=click.Path(exists=True))
@click.option("--tokens", is_flag=True, help="Show tokens")
@click.option("--ast", "show_ast", is_flag=True, help="Show AST")
def debug(file: str, tokens: bool, show_ast: bool):
    """Debug a Trident program by showing tokens/AST."""
    from trident.lexer import Lexer
    from trident.parser import parse
    
    filepath = Path(file)
    source = filepath.read_text()
    
    if tokens:
        console.print("[bold]Tokens:[/]")
        lexer = Lexer(source, str(filepath))
        for token in lexer.tokenize():
            console.print(f"  {token}")
    
    if show_ast:
        console.print("[bold]AST:[/]")
        ast = parse(source, str(filepath))
        _print_ast(ast)


def _print_ast(node, indent: int = 0):
    """Recursively print AST nodes."""
    prefix = "  " * indent
    console.print(f"{prefix}{node.__class__.__name__}")
    
    for field_name in node.__dataclass_fields__:
        if field_name == "location":
            continue
        value = getattr(node, field_name)
        if isinstance(value, tuple):
            if value:
                console.print(f"{prefix}  {field_name}:")
                for item in value:
                    if hasattr(item, "__dataclass_fields__"):
                        _print_ast(item, indent + 2)
                    else:
                        console.print(f"{prefix}    {item}")
        elif hasattr(value, "__dataclass_fields__"):
            console.print(f"{prefix}  {field_name}:")
            _print_ast(value, indent + 2)
        else:
            console.print(f"{prefix}  {field_name}: {value}")


@cli.command()
def repl():
    """Start the interactive REPL."""
    from trident.cli.repl import start_repl
    start_repl()


@cli.command()
def info():
    """Show Trident installation info."""
    from trident import __version__
    from trident.runtime import TridentRuntime
    
    runtime = TridentRuntime()
    
    console.print(Panel(
        f"""[bold blue]🔱 Trident[/] v{__version__}
A tri-modal programming language for AI-Human-TPU communication

{runtime.device_manager.info()}

[dim]Run 'trident --help' for available commands[/]""",
        title="Trident Info",
    ))


def main():
    """Main entry point."""
    cli()


if __name__ == "__main__":
    main()
