"""
Application Settings and Configuration

Centralized configuration management using environment variables.
Settings can be overridden via .env file or system environment variables.
Environment variables are prefixed with 'SENTIMENT_' to avoid conflicts.

"""

import os
from pathlib import Path
from typing import Optional
from dotenv import load_dotenv

# Load .env file if it exists (development mode)
# In production, use actual environment variables
load_dotenv()


class Settings:
    """Application settings with environment variable support."""

    def __init__(self):
        """Initialize settings from environment variables with sensible defaults."""

        # Project root: auto-detect or use environment variable
        self._project_root = self._get_project_root()

        # Paths (relative to project root by default)
        self.data_dir = self._get_path("DATA_DIR", self._project_root / "data")
        self.model_dir = self._get_path("MODEL_DIR", self._project_root / "models")
        self.config_dir = self._get_path("CONFIG_DIR", self._project_root)
        self.logs_dir = self._get_path("LOGS_DIR", self._project_root / "logs")

        # Config file
        self.config_file = self._get_path(
            "CONFIG_FILE", self.config_dir / "config.yaml"
        )

        # Optional: Device configuration
        self.device = os.getenv("SENTIMENT_DEVICE", "cpu")

        # Optional: API configuration
        self.api_host = os.getenv("SENTIMENT_API_HOST", "0.0.0.0")
        self.api_port = int(os.getenv("SENTIMENT_API_PORT", "5000"))

        # Ensure directories exist
        self._ensure_dirs()

    def _get_project_root(self) -> Path:
        """
        Determine project root directory.

        Priority:
        1. SENTIMENT_PROJECT_ROOT environment variable
        2. Auto-detection (look for setup.py, config.yaml markers)
        3. Fallback to package parent directory
        """
        # Check environment variable first
        env_root = os.getenv("SENTIMENT_PROJECT_ROOT")
        if env_root:
            return Path(env_root).resolve()

        # Auto-detect: look for project markers
        current = Path(__file__).resolve().parent

        # Walk up directory tree looking for project markers
        for parent in [current] + list(current.parents):
            markers = [
                parent / "setup.py",
                parent / "config.yaml",  # / 'config' / 'config.yaml'
                parent / "pyproject.toml",
                parent / ".git",
            ]
            if any(marker.exists() for marker in markers):
                return parent

        # Fallback: assume src/sentiment_analysis structure
        return current.parent.parent

    def _get_path(self, env_var: str, default: Path) -> Path:
        """
        Get path from environment variable or use default.

        Args:
            env_var: Environment variable name (will be prefixed with SENTIMENT_)
            default: Default path if environment variable not set

        Returns:
            Resolved absolute path
        """
        # Check for environment variable which name is prefixed with SENTIMENT_
        env_value = os.getenv(f"SENTIMENT_{env_var}")
        if env_value:
            path = Path(env_value)
            # If relative path, resolve relative to project root
            if not path.is_absolute():
                path = self._project_root / path
            return path.resolve()

        return default.resolve()

    def _ensure_dirs(self):
        """Create necessary directories if they don't exist."""
        for dir_path in [self.data_dir, self.model_dir, self.logs_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

    def resolve_path(self, path: str | Path) -> Path:
        """
        Resolve a path relative to project root if it's relative.

        Args:
            path: Path to resolve (string or Path object)

        Returns:
            Absolute Path object
        """
        path = Path(path)

        # If already absolute, return as-is
        if path.is_absolute():
            return path

        # Otherwise resolve relative to project root
        return (self._project_root / path).resolve()

    @property
    def project_root(self) -> Path:
        """Get project root directory."""
        return self._project_root

    def __repr__(self) -> str:
        return (
            f"Settings(\n"
            f"  project_root={self.project_root},\n"
            f"  data_dir={self.data_dir},\n"
            f"  model_dir={self.model_dir},\n"
            f"  config_file={self.config_file},\n"
            f"  device={self.device}\n"
            f")"
        )


# Global settings instance
settings = Settings()


# Convenience function for backward compatibility
def get_settings() -> Settings:
    """Get the global settings instance."""
    return settings


def resolve_path(path: str | Path) -> Path:
    """
    Resolve a path relative to project root.

    Convenience wrapper around settings.resolve_path() for backward compatibility.

    Args:
        path: Path to resolve

    Returns:
        Absolute Path object
    """
    return settings.resolve_path(path)
