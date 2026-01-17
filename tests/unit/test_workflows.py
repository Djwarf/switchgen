"""Unit tests for switchgen.core.workflows module."""

import pytest
import json
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestWorkflowType:
    """Tests for WorkflowType enum."""

    def test_enum_values(self):
        """Should have all expected workflow types."""
        from switchgen.core.workflows import WorkflowType

        assert WorkflowType.TEXT2IMG.value == "text2img"
        assert WorkflowType.IMG2IMG.value == "img2img"
        assert WorkflowType.INPAINT.value == "inpaint"
        assert WorkflowType.AUDIO.value == "audio"
        assert WorkflowType.THREE_D.value == "3d"

    def test_all_types_count(self):
        """Should have exactly 5 workflow types."""
        from switchgen.core.workflows import WorkflowType
        assert len(WorkflowType) == 5


class TestWorkflowSpecs:
    """Tests for WORKFLOW_SPECS registry."""

    def test_has_all_workflow_types(self):
        """Should have specs for all workflow types."""
        from switchgen.core.workflows import WORKFLOW_SPECS, WorkflowType

        for wf_type in WorkflowType:
            assert wf_type in WORKFLOW_SPECS

    def test_text2img_spec(self):
        """TEXT2IMG spec should have correct properties."""
        from switchgen.core.workflows import WORKFLOW_SPECS, WorkflowType

        spec = WORKFLOW_SPECS[WorkflowType.TEXT2IMG]

        assert spec.name == "Text to Image"
        assert spec.needs_input_image is False
        assert spec.needs_mask is False
        assert spec.needs_prompt is True
        assert spec.output_type == "image"

    def test_img2img_spec(self):
        """IMG2IMG spec should require input image."""
        from switchgen.core.workflows import WORKFLOW_SPECS, WorkflowType

        spec = WORKFLOW_SPECS[WorkflowType.IMG2IMG]

        assert spec.needs_input_image is True
        assert spec.needs_mask is False
        assert spec.needs_size is False

    def test_inpaint_spec(self):
        """INPAINT spec should require image and mask."""
        from switchgen.core.workflows import WORKFLOW_SPECS, WorkflowType

        spec = WORKFLOW_SPECS[WorkflowType.INPAINT]

        assert spec.needs_input_image is True
        assert spec.needs_mask is True

    def test_audio_spec(self):
        """AUDIO spec should have correct output type."""
        from switchgen.core.workflows import WORKFLOW_SPECS, WorkflowType

        spec = WORKFLOW_SPECS[WorkflowType.AUDIO]

        assert spec.output_type == "audio"
        assert spec.needs_size is False

    def test_3d_spec(self):
        """3D spec should require input image but not prompt."""
        from switchgen.core.workflows import WORKFLOW_SPECS, WorkflowType

        spec = WORKFLOW_SPECS[WorkflowType.THREE_D]

        assert spec.needs_input_image is True
        assert spec.needs_prompt is False


class TestGenerateSeed:
    """Tests for generate_seed function."""

    def test_returns_integer(self):
        """Should return an integer."""
        from switchgen.core.workflows import generate_seed

        result = generate_seed()

        assert isinstance(result, int)

    def test_within_valid_range(self):
        """Should return a value within valid range."""
        from switchgen.core.workflows import generate_seed, MAX_SEED

        for _ in range(100):
            result = generate_seed()
            assert 0 <= result <= MAX_SEED


class TestEnsureSeed:
    """Tests for ensure_seed function."""

    def test_negative_generates_random(self):
        """Negative seed should generate random."""
        from switchgen.core.workflows import ensure_seed, MAX_SEED

        result = ensure_seed(-1)

        assert isinstance(result, int)
        assert 0 <= result <= MAX_SEED

    def test_zero_returns_zero(self):
        """Zero seed should return zero."""
        from switchgen.core.workflows import ensure_seed

        result = ensure_seed(0)

        assert result == 0

    def test_positive_returns_same(self):
        """Positive seed should return same value."""
        from switchgen.core.workflows import ensure_seed

        result = ensure_seed(12345)

        assert result == 12345

    def test_overflow_wraps(self):
        """Seed exceeding MAX_SEED should wrap."""
        from switchgen.core.workflows import ensure_seed, MAX_SEED

        result = ensure_seed(MAX_SEED + 1)

        assert result == 0

    def test_large_seed_wraps(self):
        """Very large seed should wrap to valid range."""
        from switchgen.core.workflows import ensure_seed, MAX_SEED

        large_seed = MAX_SEED * 2 + 100
        result = ensure_seed(large_seed)

        assert 0 <= result <= MAX_SEED


class TestGetWorkflowSpec:
    """Tests for get_workflow_spec function."""

    def test_returns_correct_spec(self):
        """Should return correct spec for workflow type."""
        from switchgen.core.workflows import get_workflow_spec, WorkflowType

        spec = get_workflow_spec(WorkflowType.TEXT2IMG)

        assert spec.type == WorkflowType.TEXT2IMG


class TestGetModelsForWorkflow:
    """Tests for get_models_for_workflow function."""

    def test_filters_text2img_models(self):
        """Should filter models for text2img workflow."""
        from switchgen.core.workflows import get_models_for_workflow, WorkflowType

        all_models = ["v1-5-pruned.safetensors", "stable_audio.safetensors", "inpaint.ckpt"]
        result = get_models_for_workflow(all_models, WorkflowType.TEXT2IMG)

        assert "v1-5-pruned.safetensors" in result
        assert "stable_audio.safetensors" not in result

    def test_filters_audio_models(self):
        """Should filter models for audio workflow."""
        from switchgen.core.workflows import get_models_for_workflow, WorkflowType

        all_models = ["v1-5.safetensors", "stable-audio-1.0.safetensors"]
        result = get_models_for_workflow(all_models, WorkflowType.AUDIO)

        assert "stable-audio-1.0.safetensors" in result

    def test_returns_all_if_no_matches(self):
        """Should return all models if no patterns match."""
        from switchgen.core.workflows import get_models_for_workflow, WorkflowType

        all_models = ["custom_model.safetensors"]
        result = get_models_for_workflow(all_models, WorkflowType.TEXT2IMG)

        assert result == all_models


class TestGetCompatibleWorkflows:
    """Tests for get_compatible_workflows function."""

    def test_sd15_compatible_with_text2img(self):
        """SD 1.5 model should be compatible with text2img."""
        from switchgen.core.workflows import get_compatible_workflows, WorkflowType

        result = get_compatible_workflows("v1-5-pruned.safetensors")

        assert WorkflowType.TEXT2IMG in result

    def test_inpaint_model_compatible_with_inpaint(self):
        """Inpaint model should be compatible with inpaint workflow."""
        from switchgen.core.workflows import get_compatible_workflows, WorkflowType

        result = get_compatible_workflows("sd-v1-5-inpainting.ckpt")

        assert WorkflowType.INPAINT in result

    def test_unknown_model_defaults_to_text2img(self):
        """Unknown model should default to text2img."""
        from switchgen.core.workflows import get_compatible_workflows, WorkflowType

        result = get_compatible_workflows("completely_unknown_model.safetensors")

        assert WorkflowType.TEXT2IMG in result


class TestWorkflowBuilder:
    """Tests for WorkflowBuilder class."""

    def test_add_node_returns_id(self):
        """add_node should return node ID."""
        from switchgen.core.workflows import WorkflowBuilder

        builder = WorkflowBuilder()
        node_id = builder.add_node("TestNode", {"input": "value"})

        assert isinstance(node_id, str)

    def test_add_node_with_custom_id(self):
        """add_node should use custom ID when provided."""
        from switchgen.core.workflows import WorkflowBuilder

        builder = WorkflowBuilder()
        node_id = builder.add_node("TestNode", {"input": "value"}, node_id="custom_id")

        assert node_id == "custom_id"

    def test_link_returns_reference(self):
        """link should return [node_id, output_index] reference."""
        from switchgen.core.workflows import WorkflowBuilder

        builder = WorkflowBuilder()
        link = builder.link("node_1", 0)

        assert link == ["node_1", 0]

    def test_build_returns_workflow(self):
        """build should return workflow dict."""
        from switchgen.core.workflows import WorkflowBuilder

        builder = WorkflowBuilder()
        builder.add_node("TestNode", {"input": "value"})
        workflow = builder.build()

        assert isinstance(workflow, dict)
        assert len(workflow) == 1

    def test_clear_resets_builder(self):
        """clear should reset all nodes."""
        from switchgen.core.workflows import WorkflowBuilder

        builder = WorkflowBuilder()
        builder.add_node("TestNode", {"input": "value"})
        builder.clear()
        workflow = builder.build()

        assert len(workflow) == 0

    def test_node_counter_increments(self):
        """Node IDs should increment."""
        from switchgen.core.workflows import WorkflowBuilder

        builder = WorkflowBuilder()
        id1 = builder.add_node("Node1", {})
        id2 = builder.add_node("Node2", {})

        assert id1 == "1"
        assert id2 == "2"


class TestWorkflowManager:
    """Tests for WorkflowManager class."""

    def test_list_templates_empty_dir(self, tmp_path):
        """list_templates should return empty list for empty dir."""
        from switchgen.core.workflows import WorkflowManager

        manager = WorkflowManager(tmp_path)
        result = manager.list_templates()

        assert result == []

    def test_save_and_load_template(self, tmp_path):
        """Should save and load templates correctly."""
        from switchgen.core.workflows import WorkflowManager

        manager = WorkflowManager(tmp_path)
        workflow = {"node_1": {"class_type": "Test", "inputs": {}}}

        # Save
        path = manager.save_template("test_workflow", workflow)
        assert path.exists()

        # Load
        loaded = manager.load_template("test_workflow")
        assert loaded == workflow

    def test_load_template_not_found(self, tmp_path):
        """load_template should raise WorkflowError for missing template."""
        from switchgen.core.workflows import WorkflowManager, WorkflowError

        manager = WorkflowManager(tmp_path)

        with pytest.raises(WorkflowError, match="not found"):
            manager.load_template("nonexistent")

    def test_cache_clear(self, tmp_path):
        """clear_cache should clear the internal cache."""
        from switchgen.core.workflows import WorkflowManager

        manager = WorkflowManager(tmp_path)
        workflow = {"node_1": {"class_type": "Test", "inputs": {}}}

        manager.save_template("test", workflow)
        manager.load_template("test")  # This caches it
        manager.clear_cache()

        # Cache should be empty but file still exists
        assert len(manager._cache) == 0
        assert manager.load_template("test") == workflow

    def test_list_templates(self, tmp_path):
        """list_templates should return saved template names."""
        from switchgen.core.workflows import WorkflowManager

        manager = WorkflowManager(tmp_path)
        manager.save_template("workflow_a", {"a": 1})
        manager.save_template("workflow_b", {"b": 2})

        templates = manager.list_templates()

        assert "workflow_a" in templates
        assert "workflow_b" in templates


class TestBuildText2ImgWorkflow:
    """Tests for build_text2img_workflow function."""

    def test_returns_dict(self):
        """Should return a workflow dict."""
        from switchgen.core.workflows import build_text2img_workflow

        result = build_text2img_workflow(
            checkpoint="test.safetensors",
            prompt="test prompt",
        )

        assert isinstance(result, dict)

    def test_has_checkpoint_loader(self):
        """Should contain CheckpointLoaderSimple node."""
        from switchgen.core.workflows import build_text2img_workflow

        result = build_text2img_workflow(
            checkpoint="test.safetensors",
            prompt="test prompt",
        )

        class_types = [n["class_type"] for n in result.values()]
        assert "CheckpointLoaderSimple" in class_types

    def test_has_ksampler(self):
        """Should contain KSampler node."""
        from switchgen.core.workflows import build_text2img_workflow

        result = build_text2img_workflow(
            checkpoint="test.safetensors",
            prompt="test prompt",
        )

        class_types = [n["class_type"] for n in result.values()]
        assert "KSampler" in class_types

    def test_seed_is_explicit(self):
        """Seed should be converted to explicit integer."""
        from switchgen.core.workflows import build_text2img_workflow, MAX_SEED

        result = build_text2img_workflow(
            checkpoint="test.safetensors",
            prompt="test prompt",
            seed=-1,  # Should be converted
        )

        # Find KSampler node
        ksampler = next(n for n in result.values() if n["class_type"] == "KSampler")
        seed = ksampler["inputs"]["seed"]

        assert isinstance(seed, int)
        assert 0 <= seed <= MAX_SEED

    def test_save_to_disk_uses_save_image(self):
        """save_to_disk=True should use SaveImage node."""
        from switchgen.core.workflows import build_text2img_workflow

        result = build_text2img_workflow(
            checkpoint="test.safetensors",
            prompt="test prompt",
            save_to_disk=True,
        )

        class_types = [n["class_type"] for n in result.values()]
        assert "SaveImage" in class_types

    def test_no_save_uses_return_to_app(self):
        """save_to_disk=False should use ReturnToApp node."""
        from switchgen.core.workflows import build_text2img_workflow

        result = build_text2img_workflow(
            checkpoint="test.safetensors",
            prompt="test prompt",
            save_to_disk=False,
        )

        class_types = [n["class_type"] for n in result.values()]
        assert "ReturnToApp" in class_types


class TestBuildText2ImgMemoryWorkflow:
    """Tests for build_text2img_memory_workflow function."""

    def test_returns_tuple(self):
        """Should return (workflow, seed) tuple."""
        from switchgen.core.workflows import build_text2img_memory_workflow

        result = build_text2img_memory_workflow(
            checkpoint="test.safetensors",
            prompt="test prompt",
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_actual_seed(self):
        """Should return the actual seed used."""
        from switchgen.core.workflows import build_text2img_memory_workflow, MAX_SEED

        workflow, seed = build_text2img_memory_workflow(
            checkpoint="test.safetensors",
            prompt="test prompt",
            seed=-1,
        )

        assert isinstance(seed, int)
        assert 0 <= seed <= MAX_SEED

    def test_uses_return_to_app(self):
        """Should always use ReturnToApp node."""
        from switchgen.core.workflows import build_text2img_memory_workflow

        workflow, _ = build_text2img_memory_workflow(
            checkpoint="test.safetensors",
            prompt="test prompt",
        )

        class_types = [n["class_type"] for n in workflow.values()]
        assert "ReturnToApp" in class_types
        assert "SaveImage" not in class_types


class TestBuildImg2ImgWorkflow:
    """Tests for build_img2img_workflow function."""

    def test_has_load_image_node(self):
        """Should contain LoadImage node for input."""
        from switchgen.core.workflows import build_img2img_workflow

        result = build_img2img_workflow(
            checkpoint="test.safetensors",
            image_path="/path/to/image.png",
            prompt="test prompt",
        )

        class_types = [n["class_type"] for n in result.values()]
        assert "LoadImage" in class_types

    def test_has_vae_encode(self):
        """Should contain VAEEncode node."""
        from switchgen.core.workflows import build_img2img_workflow

        result = build_img2img_workflow(
            checkpoint="test.safetensors",
            image_path="/path/to/image.png",
            prompt="test prompt",
        )

        class_types = [n["class_type"] for n in result.values()]
        assert "VAEEncode" in class_types


class TestBuildInpaintWorkflow:
    """Tests for build_inpaint_workflow function."""

    def test_returns_tuple(self):
        """Should return (workflow, seed) tuple."""
        from switchgen.core.workflows import build_inpaint_workflow

        result = build_inpaint_workflow(
            checkpoint="inpaint.ckpt",
            image_path="/path/to/image.png",
            mask_path="/path/to/mask.png",
            prompt="test prompt",
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_has_mask_handling(self):
        """Should contain mask handling nodes."""
        from switchgen.core.workflows import build_inpaint_workflow

        workflow, _ = build_inpaint_workflow(
            checkpoint="inpaint.ckpt",
            image_path="/path/to/image.png",
            mask_path="/path/to/mask.png",
            prompt="test prompt",
        )

        class_types = [n["class_type"] for n in workflow.values()]
        assert "ImageToMask" in class_types
        assert "VAEEncodeForInpaint" in class_types


class TestBuildAudioWorkflow:
    """Tests for build_audio_workflow function."""

    def test_returns_tuple(self):
        """Should return (workflow, seed) tuple."""
        from switchgen.core.workflows import build_audio_workflow

        result = build_audio_workflow(
            checkpoint="stable-audio.safetensors",
            prompt="ambient music",
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_has_audio_nodes(self):
        """Should contain audio-specific nodes."""
        from switchgen.core.workflows import build_audio_workflow

        workflow, _ = build_audio_workflow(
            checkpoint="stable-audio.safetensors",
            prompt="ambient music",
        )

        class_types = [n["class_type"] for n in workflow.values()]
        assert "EmptyLatentAudio" in class_types
        assert "VAEDecodeAudio" in class_types
        assert "SaveAudio" in class_types


class TestBuild3DZero123Workflow:
    """Tests for build_3d_zero123_workflow function."""

    def test_returns_tuple(self):
        """Should return (workflow, seed) tuple."""
        from switchgen.core.workflows import build_3d_zero123_workflow

        result = build_3d_zero123_workflow(
            checkpoint="stable_zero123.ckpt",
            image_path="/path/to/image.png",
        )

        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_has_3d_conditioning(self):
        """Should contain 3D-specific conditioning nodes."""
        from switchgen.core.workflows import build_3d_zero123_workflow

        workflow, _ = build_3d_zero123_workflow(
            checkpoint="stable_zero123.ckpt",
            image_path="/path/to/image.png",
        )

        class_types = [n["class_type"] for n in workflow.values()]
        assert "CLIPVisionLoader" in class_types
        assert "StableZero123_Conditioning" in class_types
