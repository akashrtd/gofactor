"""
Trident REPL - Interactive Read-Eval-Print Loop.

Provides an interactive environment for experimenting with Trident.
"""

from typing import Optional
from rich.console import Console
from rich.syntax import Syntax
from rich.panel import Panel

console = Console()


class TridentREPL:
    """Interactive Trident REPL."""
    
    def __init__(self):
        self.history: list[str] = []
        self.context: dict = {}
        self._setup()
    
    def _setup(self):
        """Initialize REPL environment."""
        from trident.runtime import TridentRuntime
        from trident.primitives import vision, nlp, tensor
        
        self.runtime = TridentRuntime()
        
        # Pre-populate context
        self.context = {
            "__trident_runtime__": self.runtime,
            "vision": vision,
            "nlp": nlp,
            "tensor": tensor,
        }
        
        # Try to import JAX
        try:
            import jax
            import jax.numpy as jnp
            self.context["jax"] = jax
            self.context["jnp"] = jnp
        except ImportError:
            pass
    
    def run(self):
        """Start the REPL loop."""
        console.print(Panel(
            """[bold blue]🔱 Trident REPL[/]
Type Trident code or Python expressions.
Special commands:
  :help     - Show help
  :compile  - Compile last input to JAX
  :clear    - Clear context
  :quit     - Exit REPL""",
            title="Welcome",
        ))
        
        buffer: list[str] = []
        prompt = "[bold green]>>> [/]"
        continuation = "[bold green]... [/]"
        
        while True:
            try:
                if buffer:
                    line = console.input(continuation)
                else:
                    line = console.input(prompt)
            except (EOFError, KeyboardInterrupt):
                console.print("\n[dim]Goodbye![/]")
                break
            
            # Handle special commands
            if line.startswith(":"):
                self._handle_command(line)
                continue
            
            # Handle multi-line input
            buffer.append(line)
            
            if self._is_complete("\n".join(buffer)):
                source = "\n".join(buffer)
                buffer = []
                self._eval(source)
            elif not line.strip():
                # Empty line ends multi-line input
                source = "\n".join(buffer[:-1])
                buffer = []
                if source.strip():
                    self._eval(source)
    
    def _is_complete(self, source: str) -> bool:
        """Check if input is a complete statement."""
        # Simple heuristic: complete if no trailing colon or open brackets
        stripped = source.rstrip()
        if stripped.endswith(":"):
            return False
        
        # Check balanced brackets
        brackets = {"(": ")", "[": "]", "{": "}"}
        stack = []
        in_string = False
        string_char = None
        
        for char in source:
            if char in ('"', "'") and not in_string:
                in_string = True
                string_char = char
            elif char == string_char and in_string:
                in_string = False
            elif not in_string:
                if char in brackets:
                    stack.append(brackets[char])
                elif char in brackets.values():
                    if not stack or stack[-1] != char:
                        return True  # Syntax error, let it fail
                    stack.pop()
        
        return len(stack) == 0
    
    def _eval(self, source: str):
        """Evaluate Trident/Python code."""
        self.history.append(source)
        
        # Try as Trident first
        try:
            result = self._eval_trident(source)
            if result is not None:
                console.print(result)
            return
        except Exception:
            pass
        
        # Fall back to Python
        try:
            result = self._eval_python(source)
            if result is not None:
                console.print(result)
        except Exception as e:
            console.print(f"[red]Error:[/] {e}")
    
    def _eval_trident(self, source: str):
        """Evaluate as Trident code."""
        from trident.parser import parse
        from trident.codegen import compile_to_jax
        
        ast = parse(source, "<repl>")
        jax_code = compile_to_jax(ast)
        
        # Execute the generated code
        exec(jax_code, self.context)
    
    def _eval_python(self, source: str):
        """Evaluate as Python code directly."""
        try:
            # Try as expression first
            result = eval(source, self.context)
            return result
        except SyntaxError:
            # Try as statement
            exec(source, self.context)
            return None
    
    def _handle_command(self, command: str):
        """Handle REPL commands."""
        cmd = command.lower().strip()
        
        if cmd in (":quit", ":q", ":exit"):
            raise EOFError()
        
        elif cmd == ":help":
            console.print("""
[bold]REPL Commands:[/]
  :help     - Show this help
  :compile  - Show JAX code for last input
  :history  - Show input history
  :clear    - Clear context
  :context  - Show context variables
  :info     - Show runtime info
  :quit     - Exit REPL
""")
        
        elif cmd == ":compile":
            if self.history:
                try:
                    from trident.parser import parse
                    from trident.codegen import compile_to_jax
                    
                    ast = parse(self.history[-1])
                    jax_code = compile_to_jax(ast)
                    console.print(Syntax(jax_code, "python", theme="monokai"))
                except Exception as e:
                    console.print(f"[red]Compilation failed:[/] {e}")
            else:
                console.print("[dim]No history[/]")
        
        elif cmd == ":history":
            for i, entry in enumerate(self.history[-10:], 1):
                console.print(f"[dim]{i}.[/] {entry[:50]}...")
        
        elif cmd == ":clear":
            self._setup()
            console.print("[dim]Context cleared[/]")
        
        elif cmd == ":context":
            for name, value in self.context.items():
                if not name.startswith("_"):
                    type_name = type(value).__name__
                    console.print(f"  [bold]{name}[/]: {type_name}")
        
        elif cmd == ":info":
            console.print(self.runtime.info())
        
        else:
            console.print(f"[red]Unknown command:[/] {cmd}")


def start_repl():
    """Start the Trident REPL."""
    repl = TridentREPL()
    repl.run()


if __name__ == "__main__":
    start_repl()
