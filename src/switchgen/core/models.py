"""Model catalog and registry for downloadable models."""

from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Optional


class ModelType(Enum):
    """Types of models that can be downloaded."""
    CHECKPOINT = "checkpoints"
    VAE = "vae"
    CLIP_VISION = "clip_vision"
    TEXT_ENCODER = "text_encoders"
    LORA = "loras"
    CONTROLNET = "controlnet"
    UPSCALER = "upscale_models"


@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    id: str
    name: str
    type: ModelType
    repo_id: str            # HuggingFace repo (e.g., "openai/clip-vit-large-patch14")
    filename: str           # File in repo to download
    size_mb: int            # Approximate size in MB
    description: str
    local_filename: Optional[str] = None  # Override local filename (default: same as filename)
    required_for: list[str] = field(default_factory=list)  # Workflow types that need it

    def get_local_filename(self) -> str:
        """Get the filename to use locally."""
        return self.local_filename or self.filename


# Curated catalog of recommended models
MODEL_CATALOG: dict[str, ModelInfo] = {
    # =========================================================================
    # CLIP Vision Models
    # =========================================================================
    "clip_vit_l": ModelInfo(
        id="clip_vit_l",
        name="CLIP ViT-L/14",
        type=ModelType.CLIP_VISION,
        repo_id="openai/clip-vit-large-patch14",
        filename="model.safetensors",
        local_filename="clip_vit_l.safetensors",
        size_mb=890,
        description="Vision encoder for 3D workflows (stable_zero123)",
        required_for=["3d"],
    ),

    # =========================================================================
    # Text Encoders
    # =========================================================================
    "t5_base": ModelInfo(
        id="t5_base",
        name="T5 Base",
        type=ModelType.TEXT_ENCODER,
        repo_id="google/t5-v1_1-base",
        filename="model.safetensors",
        local_filename="t5-base.safetensors",
        size_mb=850,
        description="Text encoder for audio generation (stable-audio)",
        required_for=["audio"],
    ),

    # =========================================================================
    # Checkpoints - SD 1.5
    # =========================================================================
    "sd15_base": ModelInfo(
        id="sd15_base",
        name="Stable Diffusion 1.5",
        type=ModelType.CHECKPOINT,
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        filename="v1-5-pruned-emaonly.safetensors",
        size_mb=4270,
        description="Base SD 1.5 model for text2img, img2img",
        required_for=["text2img", "img2img"],
    ),
    "sd15_inpaint": ModelInfo(
        id="sd15_inpaint",
        name="SD 1.5 Inpainting",
        type=ModelType.CHECKPOINT,
        repo_id="stable-diffusion-v1-5/stable-diffusion-inpainting",
        filename="sd-v1-5-inpainting.ckpt",
        size_mb=4270,
        description="Inpainting model for filling masked regions",
        required_for=["inpaint"],
    ),

    # =========================================================================
    # Checkpoints - SDXL
    # =========================================================================
    "sdxl_base": ModelInfo(
        id="sdxl_base",
        name="SDXL Base 1.0",
        type=ModelType.CHECKPOINT,
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="sd_xl_base_1.0.safetensors",
        size_mb=6940,
        description="High quality text2img with better prompt following",
        required_for=["text2img"],
    ),
    "sdxl_refiner": ModelInfo(
        id="sdxl_refiner",
        name="SDXL Refiner 1.0",
        type=ModelType.CHECKPOINT,
        repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        filename="sd_xl_refiner_1.0.safetensors",
        size_mb=6080,
        description="Refiner for SDXL to improve final quality",
    ),

    # =========================================================================
    # Checkpoints - 3D
    # =========================================================================
    "stable_zero123": ModelInfo(
        id="stable_zero123",
        name="Stable Zero123",
        type=ModelType.CHECKPOINT,
        repo_id="stabilityai/stable-zero123",
        filename="stable_zero123.ckpt",
        size_mb=4900,
        description="Generate novel 3D views from a single image",
        required_for=["3d"],
    ),

    # =========================================================================
    # Checkpoints - Audio
    # =========================================================================
    "stable_audio": ModelInfo(
        id="stable_audio",
        name="Stable Audio Open",
        type=ModelType.CHECKPOINT,
        repo_id="stabilityai/stable-audio-open-1.0",
        filename="model.safetensors",
        local_filename="stable-audio-open-1.0.safetensors",
        size_mb=4850,
        description="Generate audio from text descriptions",
        required_for=["audio"],
    ),

    # =========================================================================
    # ControlNet Models (SD 1.5)
    # =========================================================================
    "controlnet_canny": ModelInfo(
        id="controlnet_canny",
        name="ControlNet Canny",
        type=ModelType.CONTROLNET,
        repo_id="lllyasviel/control_v11p_sd15_canny",
        filename="diffusion_pytorch_model.safetensors",
        local_filename="control_v11p_sd15_canny.safetensors",
        size_mb=1450,
        description="Edge-guided generation using Canny edge detection",
    ),
    "controlnet_depth": ModelInfo(
        id="controlnet_depth",
        name="ControlNet Depth",
        type=ModelType.CONTROLNET,
        repo_id="lllyasviel/control_v11f1p_sd15_depth",
        filename="diffusion_pytorch_model.safetensors",
        local_filename="control_v11f1p_sd15_depth.safetensors",
        size_mb=1450,
        description="Depth-guided generation for 3D-aware compositions",
    ),
    "controlnet_openpose": ModelInfo(
        id="controlnet_openpose",
        name="ControlNet OpenPose",
        type=ModelType.CONTROLNET,
        repo_id="lllyasviel/control_v11p_sd15_openpose",
        filename="diffusion_pytorch_model.safetensors",
        local_filename="control_v11p_sd15_openpose.safetensors",
        size_mb=1450,
        description="Pose-guided generation using skeleton detection",
    ),
    "controlnet_scribble": ModelInfo(
        id="controlnet_scribble",
        name="ControlNet Scribble",
        type=ModelType.CONTROLNET,
        repo_id="lllyasviel/control_v11p_sd15_scribble",
        filename="diffusion_pytorch_model.safetensors",
        local_filename="control_v11p_sd15_scribble.safetensors",
        size_mb=1450,
        description="Generate from rough sketches and scribbles",
    ),

    # =========================================================================
    # Upscalers
    # =========================================================================
    "realesrgan_x4": ModelInfo(
        id="realesrgan_x4",
        name="RealESRGAN x4",
        type=ModelType.UPSCALER,
        repo_id="ai-forever/Real-ESRGAN",
        filename="RealESRGAN_x4.pth",
        size_mb=64,
        description="4x image upscaling with enhanced details",
    ),

    # =========================================================================
    # VAE Models
    # =========================================================================
    "sdxl_vae": ModelInfo(
        id="sdxl_vae",
        name="SDXL VAE",
        type=ModelType.VAE,
        repo_id="stabilityai/sdxl-vae",
        filename="sdxl_vae.safetensors",
        size_mb=335,
        description="VAE for SDXL models with better color reproduction",
    ),
}


def get_models_by_type(model_type: ModelType) -> list[ModelInfo]:
    """Get all models of a specific type."""
    return [m for m in MODEL_CATALOG.values() if m.type == model_type]


def get_required_models(workflow_type: str) -> list[ModelInfo]:
    """Get models required for a specific workflow type."""
    return [m for m in MODEL_CATALOG.values() if workflow_type in m.required_for]


def get_model_info(model_id: str) -> Optional[ModelInfo]:
    """Get model info by ID."""
    return MODEL_CATALOG.get(model_id)


def is_model_installed(model_info: ModelInfo, models_dir: Path) -> bool:
    """Check if a model is installed."""
    target_dir = models_dir / model_info.type.value
    target_file = target_dir / model_info.get_local_filename()
    return target_file.exists()
