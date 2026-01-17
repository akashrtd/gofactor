"""
Package Installer for Trident.

Handles downloading and installing packages.
"""

from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional
import shutil


@dataclass
class PackageInstaller:
    """
    Installs Trident packages.
    
    Packages are installed to:
    - Global: ~/.trident/packages/
    - Local: .trident/packages/
    """
    
    global_path: Path = field(default_factory=lambda: Path.home() / ".trident" / "packages")
    local_path: Path = field(default_factory=lambda: Path(".trident") / "packages")
    
    def __post_init__(self) -> None:
        """Ensure package directories exist."""
        self.global_path.mkdir(parents=True, exist_ok=True)
    
    def install(
        self,
        package: str,
        version: Optional[str] = None,
        local: bool = True,
    ) -> bool:
        """
        Install a package.
        
        Args:
            package: Package name
            version: Specific version or None for latest
            local: Install locally (True) or globally (False)
        
        Returns:
            True if successful
        """
        target = self.local_path if local else self.global_path
        target.mkdir(parents=True, exist_ok=True)
        
        pkg_dir = target / package
        
        # In production, this would download from a registry
        # For now, just create a placeholder
        pkg_dir.mkdir(exist_ok=True)
        
        (pkg_dir / "__init__.py").write_text(f"# Package: {package}\n")
        
        if version:
            (pkg_dir / "VERSION").write_text(version)
        
        return True
    
    def uninstall(
        self,
        package: str,
        local: bool = True,
    ) -> bool:
        """
        Uninstall a package.
        
        Args:
            package: Package name
            local: Uninstall from local (True) or global (False)
        
        Returns:
            True if successful
        """
        target = self.local_path if local else self.global_path
        pkg_dir = target / package
        
        if pkg_dir.exists():
            shutil.rmtree(pkg_dir)
            return True
        
        return False
    
    def list_installed(self, local: bool = True) -> List[str]:
        """List installed packages."""
        target = self.local_path if local else self.global_path
        
        if not target.exists():
            return []
        
        return [p.name for p in target.iterdir() if p.is_dir()]
    
    def is_installed(self, package: str, local: bool = True) -> bool:
        """Check if a package is installed."""
        target = self.local_path if local else self.global_path
        return (target / package).exists()
    
    def get_package_path(self, package: str) -> Optional[Path]:
        """Get the path to an installed package."""
        # Check local first
        local_pkg = self.local_path / package
        if local_pkg.exists():
            return local_pkg
        
        # Check global
        global_pkg = self.global_path / package
        if global_pkg.exists():
            return global_pkg
        
        return None
