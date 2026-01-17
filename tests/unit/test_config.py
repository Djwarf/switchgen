"""Unit tests for switchgen.core.config module."""

import os
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestDetectSwitchgenRoot:
    """Tests for _detect_switchgen_root function."""

    def test_returns_path(self):
        """Should return a Path object."""
        from switchgen.core.config import _detect_switchgen_root
        result = _detect_switchgen_root()
        assert isinstance(result, Path)

    def test_path_is_absolute(self):
        """Should return an absolute path."""
        from switchgen.core.config import _detect_switchgen_root
        result = _detect_switchgen_root()
        assert result.is_absolute()


class TestDetectComfyPath:
    """Tests for _detect_comfy_path function."""

    def test_uses_bundled_path_when_exists(self, tmp_switchgen_root):
        """Should use bundled ComfyUI when it exists."""
        from switchgen.core.config import _detect_comfy_path, _detect_switchgen_root

        with patch('switchgen.core.config._detect_switchgen_root', return_value=tmp_switchgen_root):
            result = _detect_comfy_path()
            assert result == tmp_switchgen_root / "vendor" / "ComfyUI"

    def test_uses_env_var_when_bundled_missing(self, tmp_path):
        """Should use COMFYUI_PATH env var when bundled is missing."""
        from switchgen.core.config import _detect_comfy_path

        # Create a fake ComfyUI path
        fake_comfy = tmp_path / "ComfyUI"
        fake_comfy.mkdir()

        # Create a root without bundled ComfyUI
        fake_root = tmp_path / "switchgen"
        fake_root.mkdir()
        (fake_root / "vendor").mkdir()

        with patch('switchgen.core.config._detect_switchgen_root', return_value=fake_root):
            with patch.dict(os.environ, {"COMFYUI_PATH": str(fake_comfy)}):
                result = _detect_comfy_path()
                assert result == fake_comfy

    def test_raises_when_not_found(self, tmp_path):
        """Should raise RuntimeError when ComfyUI not found."""
        from switchgen.core.config import _detect_comfy_path

        fake_root = tmp_path / "switchgen"
        fake_root.mkdir()
        (fake_root / "vendor").mkdir()

        with patch('switchgen.core.config._detect_switchgen_root', return_value=fake_root):
            with patch.dict(os.environ, {"COMFYUI_PATH": ""}, clear=True):
                with pytest.raises(RuntimeError, match="Bundled ComfyUI not found"):
                    _detect_comfy_path()


class TestPathConfig:
    """Tests for PathConfig dataclass."""

    def test_output_dir_property(self, tmp_switchgen_root):
        """output_dir should be switchgen_root/output."""
        from switchgen.core.config import PathConfig

        config = PathConfig.__new__(PathConfig)
        config.switchgen_root = tmp_switchgen_root
        config.comfy_path = tmp_switchgen_root / "vendor" / "ComfyUI"

        assert config.output_dir == tmp_switchgen_root / "output"

    def test_temp_dir_property(self, tmp_switchgen_root):
        """temp_dir should be switchgen_root/temp."""
        from switchgen.core.config import PathConfig

        config = PathConfig.__new__(PathConfig)
        config.switchgen_root = tmp_switchgen_root
        config.comfy_path = tmp_switchgen_root / "vendor" / "ComfyUI"

        assert config.temp_dir == tmp_switchgen_root / "temp"

    def test_workflows_dir_property(self, tmp_switchgen_root):
        """workflows_dir should be switchgen_root/workflows."""
        from switchgen.core.config import PathConfig

        config = PathConfig.__new__(PathConfig)
        config.switchgen_root = tmp_switchgen_root
        config.comfy_path = tmp_switchgen_root / "vendor" / "ComfyUI"

        assert config.workflows_dir == tmp_switchgen_root / "workflows"

    def test_models_dir_property(self, tmp_switchgen_root):
        """models_dir should be switchgen_root/models."""
        from switchgen.core.config import PathConfig

        config = PathConfig.__new__(PathConfig)
        config.switchgen_root = tmp_switchgen_root
        config.comfy_path = tmp_switchgen_root / "vendor" / "ComfyUI"

        assert config.models_dir == tmp_switchgen_root / "models"

    def test_checkpoints_dir_property(self, tmp_switchgen_root):
        """checkpoints_dir should be models_dir/checkpoints."""
        from switchgen.core.config import PathConfig

        config = PathConfig.__new__(PathConfig)
        config.switchgen_root = tmp_switchgen_root
        config.comfy_path = tmp_switchgen_root / "vendor" / "ComfyUI"

        assert config.checkpoints_dir == tmp_switchgen_root / "models" / "checkpoints"

    def test_ensure_directories_creates_dirs(self, tmp_path):
        """ensure_directories should create output, temp, workflows dirs."""
        from switchgen.core.config import PathConfig

        root = tmp_path / "new_root"
        root.mkdir()

        config = PathConfig.__new__(PathConfig)
        config.switchgen_root = root
        config.comfy_path = root / "vendor" / "ComfyUI"

        config.ensure_directories()

        assert (root / "output").exists()
        assert (root / "temp").exists()
        assert (root / "workflows").exists()


class TestMemoryConfig:
    """Tests for MemoryConfig dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        from switchgen.core.config import MemoryConfig

        config = MemoryConfig()

        assert config.sunshine_vram_reserve == 300 * 1024 * 1024  # 300MB
        assert config.max_pinned_ram_percent == 0.90
        assert config.vram_warning_threshold == 0.85
        assert config.vram_critical_threshold == 0.95


class TestGenerationDefaults:
    """Tests for GenerationDefaults dataclass."""

    def test_default_values(self):
        """Should have sensible default values."""
        from switchgen.core.config import GenerationDefaults

        defaults = GenerationDefaults()

        assert defaults.width == 1024
        assert defaults.height == 1024
        assert defaults.steps == 20
        assert defaults.cfg == 7.0
        assert defaults.sampler == "euler"
        assert defaults.scheduler == "normal"
        assert defaults.batch_size == 1


class TestConfig:
    """Tests for Config dataclass."""

    def test_default_app_settings(self):
        """Should have correct default app settings."""
        from switchgen.core.config import Config, PathConfig, MemoryConfig, GenerationDefaults

        # Create a minimal config for testing
        config = Config.__new__(Config)
        config.paths = MagicMock()
        config.memory = MemoryConfig()
        config.generation = GenerationDefaults()
        config.app_id = "com.switchsides.switchgen"
        config.app_name = "SwitchGen"
        config.window_width = 1200
        config.window_height = 800

        assert config.app_id == "com.switchsides.switchgen"
        assert config.app_name == "SwitchGen"
        assert config.window_width == 1200
        assert config.window_height == 800


class TestGetConfig:
    """Tests for get_config singleton function."""

    def test_returns_config_instance(self):
        """Should return a Config instance."""
        from switchgen.core import config as config_module

        # Reset the global to test fresh
        original = config_module._config
        config_module._config = None

        try:
            with patch.object(config_module.Config, 'load') as mock_load:
                mock_config = MagicMock()
                mock_load.return_value = mock_config

                result = config_module.get_config()

                assert result == mock_config
                mock_load.assert_called_once()
        finally:
            config_module._config = original

    def test_returns_same_instance(self):
        """Should return the same instance on subsequent calls."""
        from switchgen.core import config as config_module

        original = config_module._config
        config_module._config = None

        try:
            with patch.object(config_module.Config, 'load') as mock_load:
                mock_config = MagicMock()
                mock_load.return_value = mock_config

                result1 = config_module.get_config()
                result2 = config_module.get_config()

                assert result1 is result2
                # Should only call load once (singleton)
                mock_load.assert_called_once()
        finally:
            config_module._config = original
