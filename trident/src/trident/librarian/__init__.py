"""
Librarian - Package manager for Trident.

Manages Trident packages with:
- trident.toml manifest files
- Dependency resolution
- Package installation
- Registry integration
"""

from trident.librarian.cli import main, librarian_cli
from trident.librarian.manifest import Manifest
from trident.librarian.resolver import DependencyResolver
from trident.librarian.installer import PackageInstaller

__all__ = [
    "main",
    "librarian_cli",
    "Manifest",
    "DependencyResolver",
    "PackageInstaller",
]
