"""Integration tests for workflow building.

These tests verify that workflow builders produce valid, complete workflows
with all required nodes properly connected.
"""

import pytest


class TestText2ImgWorkflowIntegration:
    """Integration tests for text2img workflow building."""

    def test_complete_workflow_structure(self):
        """Workflow should have all required nodes connected."""
        from switchgen.core.workflows import build_text2img_workflow

        workflow = build_text2img_workflow(
            checkpoint="test.safetensors",
            prompt="a beautiful sunset",
            negative_prompt="blurry, low quality",
            width=512,
            height=512,
            steps=20,
            cfg=7.0,
            seed=12345,
        )

        # Verify all required node types exist
        class_types = {n["class_type"] for n in workflow.values()}
        required = {
            "CheckpointLoaderSimple",
            "CLIPTextEncode",
            "EmptyLatentImage",
            "KSampler",
            "VAEDecode",
        }
        assert required.issubset(class_types)

    def test_node_connections_valid(self):
        """All node connections should reference existing nodes."""
        from switchgen.core.workflows import build_text2img_workflow

        workflow = build_text2img_workflow(
            checkpoint="test.safetensors",
            prompt="test",
        )

        node_ids = set(workflow.keys())

        for node_id, node in workflow.items():
            for input_name, input_value in node.get("inputs", {}).items():
                # Check if input is a node reference [node_id, output_index]
                if isinstance(input_value, list) and len(input_value) == 2:
                    ref_node_id, output_idx = input_value
                    assert ref_node_id in node_ids, \
                        f"Node {node_id} references non-existent node {ref_node_id}"

    def test_checkpoint_name_propagates(self):
        """Checkpoint name should be in CheckpointLoaderSimple inputs."""
        from switchgen.core.workflows import build_text2img_workflow

        checkpoint_name = "my_custom_model.safetensors"
        workflow = build_text2img_workflow(
            checkpoint=checkpoint_name,
            prompt="test",
        )

        loader = next(n for n in workflow.values()
                     if n["class_type"] == "CheckpointLoaderSimple")
        assert loader["inputs"]["ckpt_name"] == checkpoint_name

    def test_prompt_propagates(self):
        """Prompt should be in CLIPTextEncode inputs."""
        from switchgen.core.workflows import build_text2img_workflow

        prompt = "a detailed landscape painting"
        workflow = build_text2img_workflow(
            checkpoint="test.safetensors",
            prompt=prompt,
        )

        encoders = [n for n in workflow.values()
                   if n["class_type"] == "CLIPTextEncode"]
        prompts = [e["inputs"]["text"] for e in encoders]
        assert prompt in prompts

    def test_dimensions_propagate(self):
        """Width and height should be in EmptyLatentImage inputs."""
        from switchgen.core.workflows import build_text2img_workflow

        workflow = build_text2img_workflow(
            checkpoint="test.safetensors",
            prompt="test",
            width=768,
            height=1024,
        )

        latent = next(n for n in workflow.values()
                     if n["class_type"] == "EmptyLatentImage")
        assert latent["inputs"]["width"] == 768
        assert latent["inputs"]["height"] == 1024

    def test_sampler_parameters_propagate(self):
        """Sampler parameters should be in KSampler inputs."""
        from switchgen.core.workflows import build_text2img_workflow

        workflow = build_text2img_workflow(
            checkpoint="test.safetensors",
            prompt="test",
            steps=30,
            cfg=8.5,
            seed=99999,
            sampler="dpmpp_2m",
            scheduler="karras",
        )

        sampler = next(n for n in workflow.values()
                      if n["class_type"] == "KSampler")
        assert sampler["inputs"]["steps"] == 30
        assert sampler["inputs"]["cfg"] == 8.5
        assert sampler["inputs"]["seed"] == 99999
        assert sampler["inputs"]["sampler_name"] == "dpmpp_2m"
        assert sampler["inputs"]["scheduler"] == "karras"


class TestImg2ImgWorkflowIntegration:
    """Integration tests for img2img workflow building."""

    def test_has_image_input_path(self):
        """Workflow should load the specified input image."""
        from switchgen.core.workflows import build_img2img_workflow

        image_path = "/path/to/input.png"
        workflow = build_img2img_workflow(
            checkpoint="test.safetensors",
            image_path=image_path,
            prompt="test",
        )

        loader = next(n for n in workflow.values()
                     if n["class_type"] == "LoadImage")
        assert loader["inputs"]["image"] == image_path

    def test_has_vae_encode_instead_of_empty_latent(self):
        """Should use VAEEncode instead of EmptyLatentImage."""
        from switchgen.core.workflows import build_img2img_workflow

        workflow = build_img2img_workflow(
            checkpoint="test.safetensors",
            image_path="/path/to/image.png",
            prompt="test",
        )

        class_types = {n["class_type"] for n in workflow.values()}
        assert "VAEEncode" in class_types
        assert "EmptyLatentImage" not in class_types

    def test_denoise_parameter(self):
        """Denoise value should be in KSampler inputs."""
        from switchgen.core.workflows import build_img2img_workflow

        workflow = build_img2img_workflow(
            checkpoint="test.safetensors",
            image_path="/path/to/image.png",
            prompt="test",
            denoise=0.65,
        )

        sampler = next(n for n in workflow.values()
                      if n["class_type"] == "KSampler")
        assert sampler["inputs"]["denoise"] == 0.65


class TestInpaintWorkflowIntegration:
    """Integration tests for inpaint workflow building."""

    def test_loads_both_image_and_mask(self):
        """Should have LoadImage nodes for image and mask."""
        from switchgen.core.workflows import build_inpaint_workflow

        workflow, _ = build_inpaint_workflow(
            checkpoint="inpaint.ckpt",
            image_path="/path/to/image.png",
            mask_path="/path/to/mask.png",
            prompt="test",
        )

        load_nodes = [n for n in workflow.values()
                     if n["class_type"] == "LoadImage"]
        paths = {n["inputs"]["image"] for n in load_nodes}

        assert "/path/to/image.png" in paths
        assert "/path/to/mask.png" in paths

    def test_has_mask_processing_nodes(self):
        """Should have ImageToMask and VAEEncodeForInpaint nodes."""
        from switchgen.core.workflows import build_inpaint_workflow

        workflow, _ = build_inpaint_workflow(
            checkpoint="inpaint.ckpt",
            image_path="/path/to/image.png",
            mask_path="/path/to/mask.png",
            prompt="test",
        )

        class_types = {n["class_type"] for n in workflow.values()}
        assert "ImageToMask" in class_types
        assert "VAEEncodeForInpaint" in class_types

    def test_grow_mask_parameter(self):
        """grow_mask value should be in VAEEncodeForInpaint inputs."""
        from switchgen.core.workflows import build_inpaint_workflow

        workflow, _ = build_inpaint_workflow(
            checkpoint="inpaint.ckpt",
            image_path="/path/to/image.png",
            mask_path="/path/to/mask.png",
            prompt="test",
            grow_mask=10,
        )

        encode = next(n for n in workflow.values()
                     if n["class_type"] == "VAEEncodeForInpaint")
        assert encode["inputs"]["grow_mask_by"] == 10


class TestAudioWorkflowIntegration:
    """Integration tests for audio workflow building."""

    def test_has_audio_specific_nodes(self):
        """Should have audio-specific node types."""
        from switchgen.core.workflows import build_audio_workflow

        workflow, _ = build_audio_workflow(
            checkpoint="stable-audio.safetensors",
            prompt="ambient music",
        )

        class_types = {n["class_type"] for n in workflow.values()}
        assert "EmptyLatentAudio" in class_types
        assert "VAEDecodeAudio" in class_types
        assert "SaveAudio" in class_types

    def test_seconds_parameter(self):
        """seconds value should be in EmptyLatentAudio inputs."""
        from switchgen.core.workflows import build_audio_workflow

        workflow, _ = build_audio_workflow(
            checkpoint="stable-audio.safetensors",
            prompt="ambient music",
            seconds=45.0,
        )

        latent = next(n for n in workflow.values()
                     if n["class_type"] == "EmptyLatentAudio")
        assert latent["inputs"]["seconds"] == 45.0

    def test_has_t5_encoder(self):
        """Should have CLIPLoader for T5 encoder."""
        from switchgen.core.workflows import build_audio_workflow

        workflow, _ = build_audio_workflow(
            checkpoint="stable-audio.safetensors",
            prompt="ambient music",
        )

        class_types = {n["class_type"] for n in workflow.values()}
        assert "CLIPLoader" in class_types


class TestZero123WorkflowIntegration:
    """Integration tests for 3D zero123 workflow building."""

    def test_has_3d_conditioning_nodes(self):
        """Should have zero123-specific conditioning nodes."""
        from switchgen.core.workflows import build_3d_zero123_workflow

        workflow, _ = build_3d_zero123_workflow(
            checkpoint="stable_zero123.ckpt",
            image_path="/path/to/image.png",
        )

        class_types = {n["class_type"] for n in workflow.values()}
        assert "CLIPVisionLoader" in class_types
        assert "StableZero123_Conditioning" in class_types

    def test_elevation_azimuth_parameters(self):
        """elevation and azimuth should be in conditioning inputs."""
        from switchgen.core.workflows import build_3d_zero123_workflow

        workflow, _ = build_3d_zero123_workflow(
            checkpoint="stable_zero123.ckpt",
            image_path="/path/to/image.png",
            elevation=15.0,
            azimuth=45.0,
        )

        cond = next(n for n in workflow.values()
                   if n["class_type"] == "StableZero123_Conditioning")
        assert cond["inputs"]["elevation"] == 15.0
        assert cond["inputs"]["azimuth"] == 45.0


class TestSeedConsistency:
    """Tests for seed handling across all workflow types."""

    def test_text2img_seed_consistency(self):
        """text2img memory workflow should return consistent seed."""
        from switchgen.core.workflows import build_text2img_memory_workflow

        workflow, seed = build_text2img_memory_workflow(
            checkpoint="test.safetensors",
            prompt="test",
            seed=12345,
        )

        sampler = next(n for n in workflow.values()
                      if n["class_type"] == "KSampler")
        assert sampler["inputs"]["seed"] == seed
        assert seed == 12345

    def test_img2img_seed_consistency(self):
        """img2img memory workflow should return consistent seed."""
        from switchgen.core.workflows import build_img2img_memory_workflow

        workflow, seed = build_img2img_memory_workflow(
            checkpoint="test.safetensors",
            image_path="/path/to/image.png",
            prompt="test",
            seed=54321,
        )

        sampler = next(n for n in workflow.values()
                      if n["class_type"] == "KSampler")
        assert sampler["inputs"]["seed"] == seed
        assert seed == 54321

    def test_random_seed_is_valid(self):
        """Random seed (-1) should be converted to valid range."""
        from switchgen.core.workflows import build_text2img_memory_workflow, MAX_SEED

        workflow, seed = build_text2img_memory_workflow(
            checkpoint="test.safetensors",
            prompt="test",
            seed=-1,
        )

        assert 0 <= seed <= MAX_SEED

        sampler = next(n for n in workflow.values()
                      if n["class_type"] == "KSampler")
        assert sampler["inputs"]["seed"] == seed
