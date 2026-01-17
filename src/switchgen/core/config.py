"""Application configuration."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _detect_switchgen_root() -> Path:
    """Detect SwitchGen root directory relative to this file.

    This is primarily used for development mode where we run from the repo.
    """
    # This file is at: switchgen/src/switchgen/core/config.py
    # Root is 4 levels up: config.py -> core -> switchgen -> src -> switchgen_root
    return Path(__file__).resolve().parent.parent.parent.parent


def _get_data_root() -> Path:
    """Get the user data directory using XDG Base Directory Specification.

    Returns ~/.local/share/switchgen/ (or XDG_DATA_HOME/switchgen if set).
    """
    xdg_data_home = os.environ.get("XDG_DATA_HOME")
    if xdg_data_home:
        base = Path(xdg_data_home)
    else:
        base = Path.home() / ".local" / "share"
    return base / "switchgen"


def _detect_comfy_path() -> Path:
    """Detect ComfyUI path - system install, bundled, or environment override."""

    # 1. Check environment variable first (highest priority)
    env_path = os.environ.get("COMFYUI_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            logger.info("Using ComfyUI from COMFYUI_PATH: %s", path)
            return path
        logger.warning("COMFYUI_PATH set but path does not exist: %s", env_path)

    # 2. Check system installation (AUR package installs here)
    system_path = Path("/usr/share/switchgen/vendor/ComfyUI")
    if system_path.exists():
        logger.debug("Using system ComfyUI at %s", system_path)
        return system_path

    # 3. Check development/bundled location
    dev_path = _detect_switchgen_root() / "vendor" / "ComfyUI"
    if dev_path.exists():
        logger.debug("Using bundled ComfyUI at %s", dev_path)
        return dev_path

    # 4. Error: ComfyUI not found
    logger.error("ComfyUI not found in system or development paths")
    raise RuntimeError(
        "ComfyUI not found. Expected at:\n"
        f"  - System: {system_path}\n"
        f"  - Development: {dev_path}\n"
        "For development: run 'git submodule update --init'"
    )


def _get_effective_data_root() -> Path:
    """Determine the data root based on installation type.

    - Development mode: Use repo root (if bundled ComfyUI exists)
    - System install: Use XDG data directory (~/.local/share/switchgen/)
    """
    dev_root = _detect_switchgen_root()
    dev_comfy = dev_root / "vendor" / "ComfyUI"

    if dev_comfy.exists():
        # Development mode - use repo root for data
        logger.debug("Development mode: using repo root for data: %s", dev_root)
        return dev_root
    else:
        # System installation - use XDG data directory
        data_root = _get_data_root()
        logger.debug("System install: using XDG data root: %s", data_root)
        return data_root


@dataclass
class PathConfig:
    """Path configuration for ComfyUI and SwitchGen."""

    # ComfyUI installation path (for engine and custom nodes)
    comfy_path: Path = field(default_factory=_detect_comfy_path)

    # Data root: where user data lives (models, output, temp, input)
    # - Development: repo root (e.g., /mnt/storage/repos/switchgen/)
    # - System install: XDG data dir (e.g., ~/.local/share/switchgen/)
    data_root: Path = field(default_factory=_get_effective_data_root)

    # Legacy: still needed for some paths
    switchgen_root: Path = field(default_factory=_detect_switchgen_root)

    # User data directories (all under data_root)
    @property
    def output_dir(self) -> Path:
        return self.data_root / "output"

    @property
    def temp_dir(self) -> Path:
        return self.data_root / "temp"

    @property
    def input_dir(self) -> Path:
        return self.data_root / "input"

    @property
    def workflows_dir(self) -> Path:
        return self.data_root / "workflows"

    # Model directories (under data_root/models)
    @property
    def models_dir(self) -> Path:
        return self.data_root / "models"

    @property
    def checkpoints_dir(self) -> Path:
        return self.models_dir / "checkpoints"

    @property
    def loras_dir(self) -> Path:
        return self.models_dir / "loras"

    @property
    def vae_dir(self) -> Path:
        return self.models_dir / "vae"

    @property
    def clip_dir(self) -> Path:
        return self.models_dir / "clip"

    @property
    def controlnet_dir(self) -> Path:
        return self.models_dir / "controlnet"

    @property
    def embeddings_dir(self) -> Path:
        return self.models_dir / "embeddings"

    # Custom nodes directory (in ComfyUI installation)
    @property
    def custom_nodes_dir(self) -> Path:
        return self.comfy_path / "custom_nodes"

    def ensure_directories(self) -> None:
        """Create all data directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)
        self.models_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class MemoryConfig:
    """Memory management configuration."""

    # Reserve VRAM for Sunshine encoder (typically 100-300MB)
    sunshine_vram_reserve: int = 300 * 1024 * 1024  # 300MB in bytes

    # Maximum percentage of RAM to use for pinned memory (Linux default is 95%)
    max_pinned_ram_percent: float = 0.90

    # VRAM warning threshold (percentage)
    vram_warning_threshold: float = 0.85

    # VRAM critical threshold (percentage)
    vram_critical_threshold: float = 0.95


@dataclass
class GenerationDefaults:
    """Default generation parameters."""

    width: int = 1024
    height: int = 1024
    steps: int = 20
    cfg: float = 7.0
    sampler: str = "euler"
    scheduler: str = "normal"
    batch_size: int = 1


@dataclass
class Config:
    """Main application configuration."""

    paths: PathConfig = field(default_factory=PathConfig)
    memory: MemoryConfig = field(default_factory=MemoryConfig)
    generation: GenerationDefaults = field(default_factory=GenerationDefaults)

    # Application settings
    app_id: str = "com.switchsides.switchgen"
    app_name: str = "SwitchGen"

    # UI settings
    window_width: int = 1200
    window_height: int = 800

    @classmethod
    def load(cls, config_path: Optional[Path] = None) -> "Config":
        """Load configuration from file or return defaults."""
        # For now, just return defaults
        # TODO: Add JSON/TOML config file loading
        config = cls()
        config.paths.ensure_directories()
        logger.info(
            "Configuration loaded: comfy=%s, data=%s",
            config.paths.comfy_path,
            config.paths.data_root,
        )
        return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        logger.debug("Initializing global configuration")
        _config = Config.load()
    return _config
