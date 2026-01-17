"""Shared test fixtures for SwitchGen tests."""

import pytest
from pathlib import Path
from unittest.mock import MagicMock, patch
from dataclasses import dataclass


@pytest.fixture
def tmp_models_dir(tmp_path):
    """Create a temporary models directory structure."""
    models_dir = tmp_path / "models"
    models_dir.mkdir()
    # Create subdirectories for each model type
    for subdir in ["checkpoints", "vae", "clip_vision", "text_encoders",
                   "loras", "controlnet", "upscale_models"]:
        (models_dir / subdir).mkdir()
    return models_dir


@pytest.fixture
def tmp_switchgen_root(tmp_path):
    """Create a temporary SwitchGen root directory structure."""
    root = tmp_path / "switchgen"
    root.mkdir()
    (root / "output").mkdir()
    (root / "temp").mkdir()
    (root / "workflows").mkdir()
    (root / "models").mkdir()
    (root / "vendor" / "ComfyUI").mkdir(parents=True)
    return root


@pytest.fixture
def sample_model_info():
    """Create a sample ModelInfo for testing."""
    from switchgen.core.models import ModelInfo, ModelType
    return ModelInfo(
        id="test_model",
        name="Test Model",
        type=ModelType.CHECKPOINT,
        repo_id="test/repo",
        filename="test_model.safetensors",
        size_mb=100,
        description="A test model",
        required_for=["text2img"],
    )


@pytest.fixture
def sample_workflow():
    """Create a sample workflow dict for testing."""
    return {
        "1": {
            "class_type": "CheckpointLoaderSimple",
            "inputs": {"ckpt_name": "test.safetensors"}
        },
        "2": {
            "class_type": "CLIPTextEncode",
            "inputs": {"text": "test prompt", "clip": ["1", 1]}
        },
        "3": {
            "class_type": "EmptyLatentImage",
            "inputs": {"width": 512, "height": 512, "batch_size": 1}
        },
        "4": {
            "class_type": "KSampler",
            "inputs": {
                "model": ["1", 0],
                "positive": ["2", 0],
                "negative": ["2", 0],
                "latent_image": ["3", 0],
                "seed": 12345,
                "steps": 20,
                "cfg": 7.0,
                "sampler_name": "euler",
                "scheduler": "normal",
                "denoise": 1.0
            }
        }
    }


@pytest.fixture
def mock_engine():
    """Create a mock GenerationEngine for testing."""
    engine = MagicMock()
    engine.initialize = MagicMock()
    engine.execute = MagicMock(return_value=MagicMock(
        success=True,
        error=None,
        images=None
    ))
    engine.interrupt = MagicMock()
    engine.set_progress_callback = MagicMock()
    return engine


@pytest.fixture
def mock_config(tmp_switchgen_root):
    """Create a mock Config with test paths."""
    from switchgen.core.config import Config, PathConfig, MemoryConfig, GenerationDefaults

    # Create a PathConfig with mocked path detection
    path_config = PathConfig.__new__(PathConfig)
    path_config.switchgen_root = tmp_switchgen_root
    path_config.comfy_path = tmp_switchgen_root / "vendor" / "ComfyUI"

    config = Config(
        paths=path_config,
        memory=MemoryConfig(),
        generation=GenerationDefaults(),
    )
    return config
