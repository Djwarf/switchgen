"""Unit tests for switchgen.core.models module."""

import pytest
from pathlib import Path


class TestModelType:
    """Tests for ModelType enum."""

    def test_enum_values(self):
        """Should have all expected model types."""
        from switchgen.core.models import ModelType

        assert ModelType.CHECKPOINT.value == "checkpoints"
        assert ModelType.VAE.value == "vae"
        assert ModelType.CLIP_VISION.value == "clip_vision"
        assert ModelType.TEXT_ENCODER.value == "text_encoders"
        assert ModelType.LORA.value == "loras"
        assert ModelType.CONTROLNET.value == "controlnet"
        assert ModelType.UPSCALER.value == "upscale_models"

    def test_all_types_count(self):
        """Should have exactly 7 model types."""
        from switchgen.core.models import ModelType
        assert len(ModelType) == 7


class TestModelInfo:
    """Tests for ModelInfo dataclass."""

    def test_get_local_filename_with_override(self):
        """Should return local_filename when set."""
        from switchgen.core.models import ModelInfo, ModelType

        model = ModelInfo(
            id="test",
            name="Test",
            type=ModelType.CHECKPOINT,
            repo_id="test/repo",
            filename="original.safetensors",
            local_filename="custom.safetensors",
            size_mb=100,
            description="Test model",
        )

        assert model.get_local_filename() == "custom.safetensors"

    def test_get_local_filename_without_override(self):
        """Should return filename when local_filename not set."""
        from switchgen.core.models import ModelInfo, ModelType

        model = ModelInfo(
            id="test",
            name="Test",
            type=ModelType.CHECKPOINT,
            repo_id="test/repo",
            filename="original.safetensors",
            size_mb=100,
            description="Test model",
        )

        assert model.get_local_filename() == "original.safetensors"

    def test_required_for_default(self):
        """required_for should default to empty list."""
        from switchgen.core.models import ModelInfo, ModelType

        model = ModelInfo(
            id="test",
            name="Test",
            type=ModelType.CHECKPOINT,
            repo_id="test/repo",
            filename="test.safetensors",
            size_mb=100,
            description="Test model",
        )

        assert model.required_for == []


class TestModelCatalog:
    """Tests for MODEL_CATALOG dictionary."""

    def test_catalog_is_not_empty(self):
        """Catalog should contain models."""
        from switchgen.core.models import MODEL_CATALOG

        assert len(MODEL_CATALOG) > 0

    def test_catalog_keys_match_ids(self):
        """Catalog keys should match model IDs."""
        from switchgen.core.models import MODEL_CATALOG

        for key, model in MODEL_CATALOG.items():
            assert key == model.id

    def test_all_models_have_required_fields(self):
        """All models should have required fields populated."""
        from switchgen.core.models import MODEL_CATALOG

        for model_id, model in MODEL_CATALOG.items():
            assert model.id, f"{model_id} missing id"
            assert model.name, f"{model_id} missing name"
            assert model.type, f"{model_id} missing type"
            assert model.repo_id, f"{model_id} missing repo_id"
            assert model.filename, f"{model_id} missing filename"
            assert model.size_mb > 0, f"{model_id} has invalid size_mb"
            assert model.description, f"{model_id} missing description"

    def test_has_checkpoint_models(self):
        """Catalog should have checkpoint models."""
        from switchgen.core.models import MODEL_CATALOG, ModelType

        checkpoints = [m for m in MODEL_CATALOG.values() if m.type == ModelType.CHECKPOINT]
        assert len(checkpoints) > 0


class TestGetModelsByType:
    """Tests for get_models_by_type function."""

    def test_returns_checkpoints(self):
        """Should return only checkpoint models."""
        from switchgen.core.models import get_models_by_type, ModelType

        result = get_models_by_type(ModelType.CHECKPOINT)

        assert all(m.type == ModelType.CHECKPOINT for m in result)
        assert len(result) > 0

    def test_returns_vae(self):
        """Should return only VAE models."""
        from switchgen.core.models import get_models_by_type, ModelType

        result = get_models_by_type(ModelType.VAE)

        assert all(m.type == ModelType.VAE for m in result)

    def test_returns_controlnet(self):
        """Should return only ControlNet models."""
        from switchgen.core.models import get_models_by_type, ModelType

        result = get_models_by_type(ModelType.CONTROLNET)

        assert all(m.type == ModelType.CONTROLNET for m in result)


class TestGetRequiredModels:
    """Tests for get_required_models function."""

    def test_text2img_requirements(self):
        """Should return models required for text2img."""
        from switchgen.core.models import get_required_models

        result = get_required_models("text2img")

        assert len(result) > 0
        assert all("text2img" in m.required_for for m in result)

    def test_audio_requirements(self):
        """Should return models required for audio."""
        from switchgen.core.models import get_required_models

        result = get_required_models("audio")

        assert len(result) > 0
        assert all("audio" in m.required_for for m in result)

    def test_3d_requirements(self):
        """Should return models required for 3D."""
        from switchgen.core.models import get_required_models

        result = get_required_models("3d")

        assert len(result) > 0
        assert all("3d" in m.required_for for m in result)

    def test_unknown_workflow_returns_empty(self):
        """Should return empty list for unknown workflow type."""
        from switchgen.core.models import get_required_models

        result = get_required_models("nonexistent_workflow")

        assert result == []


class TestGetModelInfo:
    """Tests for get_model_info function."""

    def test_returns_model_for_valid_id(self):
        """Should return ModelInfo for valid ID."""
        from switchgen.core.models import get_model_info, MODEL_CATALOG

        # Get first model ID from catalog
        first_id = next(iter(MODEL_CATALOG.keys()))

        result = get_model_info(first_id)

        assert result is not None
        assert result.id == first_id

    def test_returns_none_for_invalid_id(self):
        """Should return None for invalid ID."""
        from switchgen.core.models import get_model_info

        result = get_model_info("nonexistent_model_id")

        assert result is None


class TestIsModelInstalled:
    """Tests for is_model_installed function."""

    def test_returns_true_when_file_exists(self, tmp_models_dir, sample_model_info):
        """Should return True when model file exists."""
        from switchgen.core.models import is_model_installed

        # Create the model file
        model_dir = tmp_models_dir / sample_model_info.type.value
        model_file = model_dir / sample_model_info.get_local_filename()
        model_file.touch()

        result = is_model_installed(sample_model_info, tmp_models_dir)

        assert result is True

    def test_returns_false_when_file_missing(self, tmp_models_dir, sample_model_info):
        """Should return False when model file doesn't exist."""
        from switchgen.core.models import is_model_installed

        result = is_model_installed(sample_model_info, tmp_models_dir)

        assert result is False

    def test_uses_local_filename(self, tmp_models_dir):
        """Should check for local_filename, not original filename."""
        from switchgen.core.models import ModelInfo, ModelType, is_model_installed

        model = ModelInfo(
            id="test",
            name="Test",
            type=ModelType.CHECKPOINT,
            repo_id="test/repo",
            filename="original.safetensors",
            local_filename="renamed.safetensors",
            size_mb=100,
            description="Test model",
        )

        # Create file with local_filename (not original filename)
        model_dir = tmp_models_dir / "checkpoints"
        (model_dir / "renamed.safetensors").touch()

        result = is_model_installed(model, tmp_models_dir)

        assert result is True
