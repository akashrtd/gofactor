"""
Librarian CLI - Package manager commands.

Provides commands for:
- Creating new projects
- Adding/removing dependencies
- Installing packages
- Publishing packages
"""

from pathlib import Path
from typing import Optional

import click
from rich.console import Console
from rich.table import Table
from rich.panel import Panel

console = Console()


@click.group()
def librarian_cli():
    """📚 Librarian - Trident Package Manager"""
    pass


@librarian_cli.command()
@click.argument("name", required=False)
@click.option("--path", "-p", type=click.Path(), help="Project path")
def init(name: Optional[str], path: Optional[str]):
    """Initialize a new Trident project."""
    from trident.librarian.manifest import Manifest
    
    project_path = Path(path) if path else Path.cwd()
    
    if not name:
        name = project_path.name
    
    console.print(f"[bold blue]📚 Creating project:[/] {name}")
    
    # Create directory structure
    (project_path / "src").mkdir(parents=True, exist_ok=True)
    (project_path / "tests").mkdir(exist_ok=True)
    
    # Create manifest
    manifest = Manifest(
        name=name,
        version="0.1.0",
        description=f"A Trident project",
        main="src/main.tri",
    )
    manifest.save(project_path / "trident.toml")
    
    # Create main file
    main_file = project_path / "src" / "main.tri"
    main_file.write_text('''# Trident project: {name}

fn main():
    print("Hello from Trident! 🔱")
'''.format(name=name))
    
    # Create gitignore
    gitignore = project_path / ".gitignore"
    gitignore.write_text("""# Trident
.trident/
*.pyc
__pycache__/
dist/
build/
""")
    
    console.print(f"[green]✓ Created project in {project_path}[/]")
    console.print("""
[dim]Next steps:
  cd {path}
  trident run src/main.tri[/]
""".format(path=project_path))


@librarian_cli.command()
@click.argument("package")
@click.option("--dev", is_flag=True, help="Add as dev dependency")
@click.option("--version", "-v", help="Specific version")
def add(package: str, dev: bool, version: Optional[str]):
    """Add a dependency."""
    from trident.librarian.manifest import Manifest
    from trident.librarian.resolver import DependencyResolver, Dependency
    from trident.librarian.installer import PackageInstaller
    
    manifest_path = Path("trident.toml")
    
    if not manifest_path.exists():
        console.print("[red]Error: No trident.toml found. Run 'librarian init' first.[/]")
        return
    
    manifest = Manifest.from_file(manifest_path)
    version_spec = version or "*"
    
    console.print(f"[bold blue]📚 Adding:[/] {package}@{version_spec}")
    
    # Resolve dependencies
    resolver = DependencyResolver()
    try:
        resolved = resolver.resolve([Dependency(name=package, version=version_spec)])
        
        for pkg in resolved:
            console.print(f"  [green]✓[/] {pkg.name}@{pkg.version}")
        
    except Exception as e:
        console.print(f"[red]Resolution failed: {e}[/]")
        return
    
    # Update manifest
    if dev:
        manifest.dev_dependencies[package] = version_spec
    else:
        manifest.dependencies[package] = version_spec
    
    manifest.save(manifest_path)
    
    # Install
    installer = PackageInstaller()
    for pkg in resolved:
        installer.install(pkg.name, str(pkg.version))
    
    console.print(f"[green]✓ Added {package}[/]")


@librarian_cli.command()
@click.argument("package")
def remove(package: str):
    """Remove a dependency."""
    from trident.librarian.manifest import Manifest
    from trident.librarian.installer import PackageInstaller
    
    manifest_path = Path("trident.toml")
    
    if not manifest_path.exists():
        console.print("[red]Error: No trident.toml found.[/]")
        return
    
    manifest = Manifest.from_file(manifest_path)
    
    removed = False
    if package in manifest.dependencies:
        del manifest.dependencies[package]
        removed = True
    if package in manifest.dev_dependencies:
        del manifest.dev_dependencies[package]
        removed = True
    
    if removed:
        manifest.save(manifest_path)
        
        installer = PackageInstaller()
        installer.uninstall(package)
        
        console.print(f"[green]✓ Removed {package}[/]")
    else:
        console.print(f"[yellow]Package {package} not found in dependencies[/]")


@librarian_cli.command()
def install():
    """Install all dependencies from trident.toml."""
    from trident.librarian.manifest import Manifest
    from trident.librarian.resolver import DependencyResolver, Dependency
    from trident.librarian.installer import PackageInstaller
    
    manifest_path = Path("trident.toml")
    
    if not manifest_path.exists():
        console.print("[red]Error: No trident.toml found.[/]")
        return
    
    manifest = Manifest.from_file(manifest_path)
    
    console.print(f"[bold blue]📚 Installing dependencies for {manifest.name}[/]")
    
    # Collect all dependencies
    deps = [Dependency(name=n, version=v) for n, v in manifest.dependencies.items()]
    deps += [Dependency(name=n, version=v) for n, v in manifest.dev_dependencies.items()]
    
    if not deps:
        console.print("[dim]No dependencies to install[/]")
        return
    
    # Resolve
    resolver = DependencyResolver()
    try:
        resolved = resolver.resolve(deps)
    except Exception as e:
        console.print(f"[red]Resolution failed: {e}[/]")
        return
    
    # Install
    installer = PackageInstaller()
    for pkg in resolved:
        console.print(f"  Installing {pkg.name}@{pkg.version}...")
        installer.install(pkg.name, str(pkg.version))
    
    # Generate lockfile
    lockfile = resolver.generate_lockfile(resolved)
    Path("trident.lock").write_text(lockfile)
    
    console.print(f"[green]✓ Installed {len(resolved)} package(s)[/]")


@librarian_cli.command()
def list():
    """List installed packages."""
    from trident.librarian.installer import PackageInstaller
    
    installer = PackageInstaller()
    
    local_pkgs = installer.list_installed(local=True)
    global_pkgs = installer.list_installed(local=False)
    
    table = Table(title="Installed Packages")
    table.add_column("Package", style="cyan")
    table.add_column("Location", style="green")
    
    for pkg in local_pkgs:
        table.add_row(pkg, "local")
    
    for pkg in global_pkgs:
        if pkg not in local_pkgs:
            table.add_row(pkg, "global")
    
    if local_pkgs or global_pkgs:
        console.print(table)
    else:
        console.print("[dim]No packages installed[/]")


@librarian_cli.command()
@click.argument("query")
def search(query: str):
    """Search for packages (placeholder)."""
    console.print(f"[bold blue]📚 Searching for:[/] {query}")
    console.print("[dim]Package registry not yet available. Coming soon![/]")


@librarian_cli.command()
def publish():
    """Publish package to registry (placeholder)."""
    console.print("[dim]Package publishing not yet available. Coming soon![/]")


def main():
    """Main entry point for librarian CLI."""
    librarian_cli()


if __name__ == "__main__":
    main()
