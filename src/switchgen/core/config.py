"""Application configuration."""

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


def _detect_switchgen_root() -> Path:
    """Detect SwitchGen root directory relative to this file."""
    # This file is at: switchgen/src/switchgen/core/config.py
    # Root is 4 levels up: config.py -> core -> switchgen -> src -> switchgen_root
    return Path(__file__).resolve().parent.parent.parent.parent


def _detect_comfy_path() -> Path:
    """Detect ComfyUI path - bundled version or environment override."""
    switchgen_root = _detect_switchgen_root()

    # 1. Primary: Use bundled ComfyUI in vendor/
    bundled_path = switchgen_root / "vendor" / "ComfyUI"
    if bundled_path.exists():
        logger.debug("Using bundled ComfyUI at %s", bundled_path)
        return bundled_path

    # 2. Fallback: Check environment variable (for development)
    env_path = os.environ.get("COMFYUI_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            logger.info("Using ComfyUI from COMFYUI_PATH: %s", path)
            return path
        logger.warning("COMFYUI_PATH set but path does not exist: %s", env_path)

    # 3. Error: ComfyUI not found
    logger.error("ComfyUI not found at %s", bundled_path)
    raise RuntimeError(
        f"Bundled ComfyUI not found at {bundled_path}. "
        "Run 'git submodule update --init' to install."
    )


@dataclass
class PathConfig:
    """Path configuration for ComfyUI and SwitchGen."""

    # ComfyUI installation path (for custom nodes and execution engine)
    comfy_path: Path = field(default_factory=_detect_comfy_path)

    # SwitchGen paths (auto-detected relative to this file)
    switchgen_root: Path = field(default_factory=_detect_switchgen_root)

    @property
    def output_dir(self) -> Path:
        return self.switchgen_root / "output"

    @property
    def temp_dir(self) -> Path:
        return self.switchgen_root / "temp"

    @property
    def input_dir(self) -> Path:
        return self.switchgen_root / "input"

    @property
    def workflows_dir(self) -> Path:
        return self.switchgen_root / "workflows"

    # Model directories (stored in switchgen, not ComfyUI)
    @property
    def models_dir(self) -> Path:
        return self.switchgen_root / "models"

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

    # Custom nodes directory (in SwitchGen root, not ComfyUI)
    @property
    def custom_nodes_dir(self) -> Path:
        return self.switchgen_root / "custom_nodes"

    def ensure_directories(self) -> None:
        """Create output, temp, and input directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
        self.input_dir.mkdir(parents=True, exist_ok=True)
        self.workflows_dir.mkdir(parents=True, exist_ok=True)


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
            "Configuration loaded: root=%s, comfy=%s",
            config.paths.switchgen_root,
            config.paths.comfy_path,
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
