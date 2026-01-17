"""Application configuration."""

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def _detect_switchgen_root() -> Path:
    """Detect SwitchGen root directory relative to this file."""
    # This file is at: switchgen/src/switchgen/core/config.py
    # Root is 4 levels up: config.py -> core -> switchgen -> src -> switchgen_root
    return Path(__file__).resolve().parent.parent.parent.parent


def _detect_comfy_path() -> Path:
    """Detect ComfyUI path from environment or common locations."""
    # 1. Check environment variable
    env_path = os.environ.get("COMFYUI_PATH")
    if env_path:
        path = Path(env_path)
        if path.exists():
            return path

    # 2. Check sibling directory (common development setup)
    switchgen_root = _detect_switchgen_root()
    sibling_path = switchgen_root.parent / "ComfyUI"
    if sibling_path.exists():
        return sibling_path

    # 3. Check common installation paths
    common_paths = [
        Path.home() / "ComfyUI",
        Path("/opt/ComfyUI"),
        Path("/usr/local/ComfyUI"),
    ]
    for path in common_paths:
        if path.exists():
            return path

    # 4. Fallback to sibling (will be created or error later)
    return sibling_path


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

    # ComfyUI custom nodes (still uses ComfyUI installation)
    @property
    def custom_nodes_dir(self) -> Path:
        return self.comfy_path / "custom_nodes"

    def ensure_directories(self) -> None:
        """Create output and temp directories if they don't exist."""
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.temp_dir.mkdir(parents=True, exist_ok=True)
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
        return config


# Global config instance
_config: Optional[Config] = None


def get_config() -> Config:
    """Get the global configuration instance."""
    global _config
    if _config is None:
        _config = Config.load()
    return _config
