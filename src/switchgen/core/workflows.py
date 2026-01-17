"""Workflow management for ComfyUI API format workflows.

ComfyUI workflows must be in API format, not the graph format from the UI.
Use "Save (API Format)" in ComfyUI Dev Mode to export.

CRITICAL (Gotcha #5): Seeds must ALWAYS be explicit integers!
The ComfyUI backend expects a specific integer seed - it does NOT handle
seed=-1 or seed=null like the web UI does. The web UI's JavaScript generates
the random number before sending to the backend.
"""

import json
import random
from copy import deepcopy
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

from .config import get_config


# =============================================================================
# Workflow Type System
# =============================================================================

class WorkflowType(Enum):
    """Available workflow types."""
    TEXT2IMG = "text2img"
    IMG2IMG = "img2img"
    INPAINT = "inpaint"
    AUDIO = "audio"
    THREE_D = "3d"


@dataclass
class WorkflowSpec:
    """Specification for a workflow type."""
    type: WorkflowType
    name: str
    description: str
    compatible_models: list[str] = field(default_factory=list)  # Model name patterns
    needs_input_image: bool = False
    needs_mask: bool = False
    needs_prompt: bool = True
    needs_size: bool = True  # Whether user can set output size
    output_type: str = "image"  # "image", "audio", "mesh"
    # Default parameters
    default_steps: int = 20
    default_cfg: float = 7.0
    default_denoise: float = 0.75


# Registry of workflow specifications
WORKFLOW_SPECS: dict[WorkflowType, WorkflowSpec] = {
    WorkflowType.TEXT2IMG: WorkflowSpec(
        type=WorkflowType.TEXT2IMG,
        name="Text to Image",
        description="Generate images from text prompts",
        compatible_models=["v1-5", "AOM", "Abyss", "orangemix"],
        output_type="image",
        default_steps=20,
        default_cfg=7.0,
    ),
    WorkflowType.IMG2IMG: WorkflowSpec(
        type=WorkflowType.IMG2IMG,
        name="Image to Image",
        description="Transform existing images with text guidance",
        compatible_models=["v1-5", "AOM", "Abyss", "orangemix"],
        needs_input_image=True,
        needs_size=False,  # Uses input image size
        output_type="image",
        default_steps=20,
        default_cfg=7.0,
        default_denoise=0.75,
    ),
    WorkflowType.INPAINT: WorkflowSpec(
        type=WorkflowType.INPAINT,
        name="Inpainting",
        description="Fill in masked regions of an image",
        compatible_models=["inpaint"],
        needs_input_image=True,
        needs_mask=True,
        needs_size=False,  # Uses input image size
        output_type="image",
        default_steps=20,
        default_cfg=7.0,
        default_denoise=1.0,  # Full denoise for inpainting
    ),
    WorkflowType.AUDIO: WorkflowSpec(
        type=WorkflowType.AUDIO,
        name="Audio",
        description="Generate audio from text prompts",
        compatible_models=["stable-audio", "ace_step"],
        needs_size=False,  # No image size for audio
        output_type="audio",
        default_steps=100,  # Audio needs more steps
        default_cfg=7.0,
    ),
    WorkflowType.THREE_D: WorkflowSpec(
        type=WorkflowType.THREE_D,
        name="3D Novel View",
        description="Generate 3D views from a single image",
        compatible_models=["stable_zero123", "hunyuan3d"],
        needs_input_image=True,
        needs_prompt=False,  # 3D uses image conditioning, not text
        needs_size=False,  # Fixed output size
        output_type="image",
        default_steps=20,
        default_cfg=4.0,  # Lower CFG for zero123
    ),
}


def get_models_for_workflow(
    all_checkpoints: list[str], wf_type: WorkflowType
) -> list[str]:
    """Filter checkpoints to only those compatible with a workflow type."""
    spec = WORKFLOW_SPECS[wf_type]
    if not spec.compatible_models:
        return all_checkpoints

    compatible = []
    for ckpt in all_checkpoints:
        ckpt_lower = ckpt.lower()
        for pattern in spec.compatible_models:
            if pattern.lower() in ckpt_lower:
                compatible.append(ckpt)
                break
    return compatible if compatible else all_checkpoints  # Fallback to all


def get_workflow_spec(wf_type: WorkflowType) -> WorkflowSpec:
    """Get the specification for a workflow type."""
    return WORKFLOW_SPECS[wf_type]


def get_compatible_workflows(model_name: str) -> list[WorkflowType]:
    """Get workflow types compatible with a given model."""
    compatible = []
    model_lower = model_name.lower()
    for wf_type, spec in WORKFLOW_SPECS.items():
        for pattern in spec.compatible_models:
            if pattern.lower() in model_lower:
                compatible.append(wf_type)
                break
    return compatible if compatible else [WorkflowType.TEXT2IMG]  # Default fallback


# Maximum seed value (2^32 - 1)
MAX_SEED = 2**32 - 1


class WorkflowError(Exception):
    """Error in workflow handling."""
    pass


def generate_seed() -> int:
    """Generate a random seed.

    CRITICAL (Gotcha #5): ComfyUI backend expects explicit integer seeds.
    The web UI generates random numbers in JavaScript before sending.
    We must do the same.
    """
    return random.randint(0, MAX_SEED)


def ensure_seed(seed: int) -> int:
    """Ensure seed is a valid explicit integer.

    Args:
        seed: Seed value (-1 or negative means generate random)

    Returns:
        Valid seed integer
    """
    if seed < 0:
        return generate_seed()
    return seed % (MAX_SEED + 1)  # Wrap to valid range


class WorkflowManager:
    """Manages workflow templates in API format."""

    def __init__(self, templates_dir: Optional[Path] = None):
        if templates_dir is None:
            templates_dir = get_config().paths.workflows_dir
        self.templates_dir = templates_dir
        self._cache: dict[str, dict] = {}

    def load_template(self, name: str) -> dict:
        """Load a workflow template by name."""
        if name in self._cache:
            return deepcopy(self._cache[name])

        path = self.templates_dir / f"{name}.json"
        if not path.exists():
            raise WorkflowError(f"Workflow template not found: {path}")

        with open(path) as f:
            workflow = json.load(f)

        self._cache[name] = workflow
        return deepcopy(workflow)

    def save_template(self, name: str, workflow: dict) -> Path:
        """Save a workflow as a template."""
        path = self.templates_dir / f"{name}.json"
        with open(path, "w") as f:
            json.dump(workflow, f, indent=2)

        self._cache[name] = deepcopy(workflow)
        return path

    def list_templates(self) -> list[str]:
        """List available workflow templates."""
        if not self.templates_dir.exists():
            return []
        return [f.stem for f in self.templates_dir.glob("*.json")]

    def clear_cache(self) -> None:
        """Clear the template cache."""
        self._cache.clear()


class WorkflowBuilder:
    """Build ComfyUI workflows programmatically."""

    def __init__(self):
        self._nodes: dict[str, dict] = {}
        self._node_counter = 0

    def _next_id(self) -> str:
        """Get next node ID."""
        self._node_counter += 1
        return str(self._node_counter)

    def add_node(self, class_type: str, inputs: dict, node_id: Optional[str] = None) -> str:
        """Add a node to the workflow."""
        if node_id is None:
            node_id = self._next_id()

        self._nodes[node_id] = {
            "class_type": class_type,
            "inputs": inputs
        }
        return node_id

    def link(self, from_node: str, output_index: int) -> list:
        """Create a link reference to another node's output."""
        return [from_node, output_index]

    def build(self) -> dict:
        """Build and return the workflow dict."""
        return deepcopy(self._nodes)

    def clear(self) -> None:
        """Clear all nodes."""
        self._nodes.clear()
        self._node_counter = 0


def build_text2img_workflow(
    checkpoint: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
    batch_size: int = 1,
    save_to_disk: bool = True,
    capture_id: str = "default",
) -> dict:
    """Build a text-to-image workflow.

    Args:
        checkpoint: Checkpoint filename (e.g., "sd_xl_base_1.0.safetensors")
        prompt: Positive prompt
        negative_prompt: Negative prompt
        width: Image width
        height: Image height
        steps: Number of sampling steps
        cfg: CFG scale
        seed: Random seed (-1 for random) - WILL BE CONVERTED TO EXPLICIT INT
        sampler: Sampler name
        scheduler: Scheduler name
        batch_size: Number of images to generate
        save_to_disk: If True, use SaveImage node; if False, use ReturnToApp
        capture_id: ID for ReturnToApp capture (only used if save_to_disk=False)

    Returns:
        Workflow dict in API format
    """
    # CRITICAL (Gotcha #5): Always ensure seed is explicit integer
    seed = ensure_seed(seed)

    builder = WorkflowBuilder()

    # 1. Load checkpoint
    ckpt_id = builder.add_node("CheckpointLoaderSimple", {
        "ckpt_name": checkpoint
    })

    # 2. CLIP encode positive prompt
    pos_id = builder.add_node("CLIPTextEncode", {
        "text": prompt,
        "clip": builder.link(ckpt_id, 1)
    })

    # 3. CLIP encode negative prompt
    neg_id = builder.add_node("CLIPTextEncode", {
        "text": negative_prompt,
        "clip": builder.link(ckpt_id, 1)
    })

    # 4. Empty latent image
    latent_id = builder.add_node("EmptyLatentImage", {
        "width": width,
        "height": height,
        "batch_size": batch_size
    })

    # 5. KSampler
    sampler_id = builder.add_node("KSampler", {
        "model": builder.link(ckpt_id, 0),
        "positive": builder.link(pos_id, 0),
        "negative": builder.link(neg_id, 0),
        "latent_image": builder.link(latent_id, 0),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": 1.0
    })

    # 6. VAE Decode
    decode_id = builder.add_node("VAEDecode", {
        "samples": builder.link(sampler_id, 0),
        "vae": builder.link(ckpt_id, 2)
    })

    # 7. Output node - either SaveImage or ReturnToApp
    if save_to_disk:
        builder.add_node("SaveImage", {
            "images": builder.link(decode_id, 0),
            "filename_prefix": "switchgen"
        })
    else:
        # Use our custom ReturnToApp node for in-memory capture (Gotcha #3)
        builder.add_node("ReturnToApp", {
            "images": builder.link(decode_id, 0),
            "capture_id": capture_id
        })

    return builder.build()


def build_text2img_memory_workflow(
    checkpoint: str,
    prompt: str,
    negative_prompt: str = "",
    width: int = 1024,
    height: int = 1024,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
    batch_size: int = 1,
    capture_id: str = "default",
) -> tuple[dict, int]:
    """Build a text-to-image workflow that returns images in memory.

    This uses the ReturnToApp custom node instead of SaveImage,
    allowing the image to be retrieved via get_captured_image().

    Args:
        Same as build_text2img_workflow, but no save_to_disk option

    Returns:
        Tuple of (workflow dict in API format, actual seed used)
    """
    # Resolve the seed first so we can return it
    actual_seed = ensure_seed(seed)

    workflow = build_text2img_workflow(
        checkpoint=checkpoint,
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        steps=steps,
        cfg=cfg,
        seed=actual_seed,  # Pass already-resolved seed
        sampler=sampler,
        scheduler=scheduler,
        batch_size=batch_size,
        save_to_disk=False,
        capture_id=capture_id,
    )
    return workflow, actual_seed


def build_img2img_workflow(
    checkpoint: str,
    image_path: str,
    prompt: str,
    negative_prompt: str = "",
    denoise: float = 0.75,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
    save_to_disk: bool = True,
    capture_id: str = "default",
) -> dict:
    """Build an image-to-image workflow.

    Args:
        checkpoint: Checkpoint filename
        image_path: Path to input image
        prompt: Positive prompt
        negative_prompt: Negative prompt
        denoise: Denoising strength (0.0-1.0)
        steps: Number of sampling steps
        cfg: CFG scale
        seed: Random seed (-1 for random) - WILL BE CONVERTED TO EXPLICIT INT
        sampler: Sampler name
        scheduler: Scheduler name
        save_to_disk: If True, use SaveImage; if False, use ReturnToApp
        capture_id: ID for ReturnToApp capture

    Returns:
        Workflow dict in API format
    """
    # CRITICAL (Gotcha #5): Always ensure seed is explicit integer
    seed = ensure_seed(seed)

    builder = WorkflowBuilder()

    # 1. Load checkpoint
    ckpt_id = builder.add_node("CheckpointLoaderSimple", {
        "ckpt_name": checkpoint
    })

    # 2. Load image
    img_id = builder.add_node("LoadImage", {
        "image": image_path
    })

    # 3. VAE Encode
    encode_id = builder.add_node("VAEEncode", {
        "pixels": builder.link(img_id, 0),
        "vae": builder.link(ckpt_id, 2)
    })

    # 4. CLIP encode positive prompt
    pos_id = builder.add_node("CLIPTextEncode", {
        "text": prompt,
        "clip": builder.link(ckpt_id, 1)
    })

    # 5. CLIP encode negative prompt
    neg_id = builder.add_node("CLIPTextEncode", {
        "text": negative_prompt,
        "clip": builder.link(ckpt_id, 1)
    })

    # 6. KSampler with denoise < 1.0
    sampler_id = builder.add_node("KSampler", {
        "model": builder.link(ckpt_id, 0),
        "positive": builder.link(pos_id, 0),
        "negative": builder.link(neg_id, 0),
        "latent_image": builder.link(encode_id, 0),
        "seed": seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": denoise
    })

    # 7. VAE Decode
    decode_id = builder.add_node("VAEDecode", {
        "samples": builder.link(sampler_id, 0),
        "vae": builder.link(ckpt_id, 2)
    })

    # 8. Output node
    if save_to_disk:
        builder.add_node("SaveImage", {
            "images": builder.link(decode_id, 0),
            "filename_prefix": "switchgen_img2img"
        })
    else:
        builder.add_node("ReturnToApp", {
            "images": builder.link(decode_id, 0),
            "capture_id": capture_id
        })

    return builder.build()


def build_img2img_memory_workflow(
    checkpoint: str,
    image_path: str,
    prompt: str,
    negative_prompt: str = "",
    denoise: float = 0.75,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
    capture_id: str = "default",
) -> tuple[dict, int]:
    """Build an image-to-image workflow that returns images in memory.

    Returns:
        Tuple of (workflow dict in API format, actual seed used)
    """
    actual_seed = ensure_seed(seed)

    workflow = build_img2img_workflow(
        checkpoint=checkpoint,
        image_path=image_path,
        prompt=prompt,
        negative_prompt=negative_prompt,
        denoise=denoise,
        steps=steps,
        cfg=cfg,
        seed=actual_seed,
        sampler=sampler,
        scheduler=scheduler,
        save_to_disk=False,
        capture_id=capture_id,
    )
    return workflow, actual_seed


def build_inpaint_workflow(
    checkpoint: str,
    image_path: str,
    mask_path: str,
    prompt: str,
    negative_prompt: str = "",
    denoise: float = 1.0,
    grow_mask: int = 6,
    steps: int = 20,
    cfg: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
    capture_id: str = "default",
) -> tuple[dict, int]:
    """Build an inpainting workflow.

    Args:
        checkpoint: Inpainting checkpoint filename (e.g., 512-inpainting-ema)
        image_path: Path to input image
        mask_path: Path to mask image (white = inpaint area)
        prompt: What to generate in masked area
        negative_prompt: What to avoid
        denoise: Denoising strength (1.0 = full regeneration)
        grow_mask: Pixels to expand mask by
        steps: Sampling steps
        cfg: CFG scale
        seed: Random seed (-1 for random)

    Returns:
        Tuple of (workflow dict, actual seed used)
    """
    actual_seed = ensure_seed(seed)
    builder = WorkflowBuilder()

    # 1. Load checkpoint
    ckpt_id = builder.add_node("CheckpointLoaderSimple", {
        "ckpt_name": checkpoint
    })

    # 2. Load image
    img_id = builder.add_node("LoadImage", {
        "image": image_path
    })

    # 3. Load mask
    mask_id = builder.add_node("LoadImage", {
        "image": mask_path
    })

    # 4. Convert mask image to mask (use alpha or red channel)
    mask_convert_id = builder.add_node("ImageToMask", {
        "image": builder.link(mask_id, 0),
        "channel": "red"
    })

    # 5. VAE Encode for Inpaint (handles mask embedding)
    encode_id = builder.add_node("VAEEncodeForInpaint", {
        "pixels": builder.link(img_id, 0),
        "vae": builder.link(ckpt_id, 2),
        "mask": builder.link(mask_convert_id, 0),
        "grow_mask_by": grow_mask
    })

    # 6. CLIP encode positive prompt
    pos_id = builder.add_node("CLIPTextEncode", {
        "text": prompt,
        "clip": builder.link(ckpt_id, 1)
    })

    # 7. CLIP encode negative prompt
    neg_id = builder.add_node("CLIPTextEncode", {
        "text": negative_prompt,
        "clip": builder.link(ckpt_id, 1)
    })

    # 8. KSampler
    sampler_id = builder.add_node("KSampler", {
        "model": builder.link(ckpt_id, 0),
        "positive": builder.link(pos_id, 0),
        "negative": builder.link(neg_id, 0),
        "latent_image": builder.link(encode_id, 0),
        "seed": actual_seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": denoise
    })

    # 9. VAE Decode
    decode_id = builder.add_node("VAEDecode", {
        "samples": builder.link(sampler_id, 0),
        "vae": builder.link(ckpt_id, 2)
    })

    # 10. Output
    builder.add_node("ReturnToApp", {
        "images": builder.link(decode_id, 0),
        "capture_id": capture_id
    })

    return builder.build(), actual_seed


def build_audio_workflow(
    checkpoint: str,
    prompt: str,
    negative_prompt: str = "",
    seconds: float = 30.0,
    steps: int = 100,
    cfg: float = 7.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
) -> tuple[dict, int]:
    """Build an audio generation workflow (stable-audio-open).

    Args:
        checkpoint: Audio model checkpoint
        prompt: Text description of desired audio
        negative_prompt: What to avoid
        seconds: Duration in seconds (1-60)
        steps: Sampling steps (100 typical for audio)
        cfg: CFG scale
        seed: Random seed (-1 for random)

    Returns:
        Tuple of (workflow dict, actual seed used)
    """
    actual_seed = ensure_seed(seed)
    builder = WorkflowBuilder()

    # Generate unique output filename
    output_filename = f"switchgen_audio_{actual_seed}"

    # 1. Load checkpoint (audio model + VAE)
    ckpt_id = builder.add_node("CheckpointLoaderSimple", {
        "ckpt_name": checkpoint
    })

    # 2. Load T5 text encoder separately (stable-audio uses T5-base)
    clip_id = builder.add_node("CLIPLoader", {
        "clip_name": "t5-base.safetensors",
        "type": "stable_audio"
    })

    # 3. CLIP encode positive prompt (using T5)
    pos_id = builder.add_node("CLIPTextEncode", {
        "text": prompt,
        "clip": builder.link(clip_id, 0)
    })

    # 4. CLIP encode negative prompt (using T5)
    neg_id = builder.add_node("CLIPTextEncode", {
        "text": negative_prompt,
        "clip": builder.link(clip_id, 0)
    })

    # 5. Empty audio latent
    latent_id = builder.add_node("EmptyLatentAudio", {
        "seconds": seconds,
        "batch_size": 1
    })

    # 6. Audio conditioning (timing)
    cond_id = builder.add_node("ConditioningStableAudio", {
        "positive": builder.link(pos_id, 0),
        "negative": builder.link(neg_id, 0),
        "seconds_start": 0.0,
        "seconds_total": seconds
    })

    # 7. KSampler
    sampler_id = builder.add_node("KSampler", {
        "model": builder.link(ckpt_id, 0),
        "positive": builder.link(cond_id, 0),
        "negative": builder.link(cond_id, 1),
        "latent_image": builder.link(latent_id, 0),
        "seed": actual_seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": 1.0
    })

    # 8. VAE Decode Audio
    decode_id = builder.add_node("VAEDecodeAudio", {
        "samples": builder.link(sampler_id, 0),
        "vae": builder.link(ckpt_id, 2)
    })

    # 9. Save Audio (outputs to file)
    builder.add_node("SaveAudio", {
        "audio": builder.link(decode_id, 0),
        "filename_prefix": output_filename
    })

    return builder.build(), actual_seed


def build_3d_zero123_workflow(
    checkpoint: str,
    image_path: str,
    elevation: float = 0.0,
    azimuth: float = 0.0,
    width: int = 256,
    height: int = 256,
    steps: int = 20,
    cfg: float = 4.0,
    seed: int = -1,
    sampler: str = "euler",
    scheduler: str = "normal",
    capture_id: str = "default",
) -> tuple[dict, int]:
    """Build a 3D generation workflow using stable_zero123.

    Generates a novel view of an object from a different camera angle.

    Args:
        checkpoint: stable_zero123 checkpoint
        image_path: Path to input image (should be object on white/neutral background)
        elevation: Camera elevation angle in degrees
        azimuth: Camera azimuth angle in degrees
        width: Output width
        height: Output height
        steps: Sampling steps
        cfg: CFG scale (typically lower, 4.0)
        seed: Random seed

    Returns:
        Tuple of (workflow dict, actual seed used)
    """
    actual_seed = ensure_seed(seed)
    builder = WorkflowBuilder()

    # 1. Load checkpoint (includes CLIP vision)
    ckpt_id = builder.add_node("CheckpointLoaderSimple", {
        "ckpt_name": checkpoint
    })

    # 2. Load CLIP Vision model (ViT-L/14 required for stable_zero123)
    clip_vision_id = builder.add_node("CLIPVisionLoader", {
        "clip_name": "clip_vit_l.safetensors"  # CLIP ViT-L/14 vision model
    })

    # 3. Load input image
    img_id = builder.add_node("LoadImage", {
        "image": image_path
    })

    # 4. StableZero123 Conditioning
    cond_id = builder.add_node("StableZero123_Conditioning", {
        "clip_vision": builder.link(clip_vision_id, 0),
        "init_image": builder.link(img_id, 0),
        "vae": builder.link(ckpt_id, 2),
        "width": width,
        "height": height,
        "batch_size": 1,
        "elevation": elevation,
        "azimuth": azimuth
    })

    # 5. KSampler
    sampler_id = builder.add_node("KSampler", {
        "model": builder.link(ckpt_id, 0),
        "positive": builder.link(cond_id, 0),
        "negative": builder.link(cond_id, 1),
        "latent_image": builder.link(cond_id, 2),
        "seed": actual_seed,
        "steps": steps,
        "cfg": cfg,
        "sampler_name": sampler,
        "scheduler": scheduler,
        "denoise": 1.0
    })

    # 6. VAE Decode
    decode_id = builder.add_node("VAEDecode", {
        "samples": builder.link(sampler_id, 0),
        "vae": builder.link(ckpt_id, 2)
    })

    # 7. Output
    builder.add_node("ReturnToApp", {
        "images": builder.link(decode_id, 0),
        "capture_id": capture_id
    })

    return builder.build(), actual_seed


# Global workflow manager
_manager: Optional[WorkflowManager] = None


def get_workflow_manager() -> WorkflowManager:
    """Get the global workflow manager."""
    global _manager
    if _manager is None:
        _manager = WorkflowManager()
    return _manager
