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


class QualityTier(Enum):
    """Quality tier for models."""
    STARTER = "starter"      # Good for beginners, lower requirements
    STANDARD = "standard"    # Balanced quality and performance
    HIGH = "high"           # Best quality, higher requirements


@dataclass
class ModelInfo:
    """Information about a downloadable model."""
    id: str
    name: str
    type: ModelType
    repo_id: str            # HuggingFace repo (e.g., "openai/clip-vit-large-patch14")
    filename: str           # File in repo to download
    size_mb: int            # Approximate size in MB
    description: str        # Short description
    local_filename: Optional[str] = None  # Override local filename (default: same as filename)
    required_for: list[str] = field(default_factory=list)  # Workflow types that need it
    # Beginner-friendly fields
    vram_gb: float = 4.0    # Minimum VRAM required in GB
    quality_tier: QualityTier = QualityTier.STANDARD
    recommended: bool = False  # Show as recommended for beginners
    tips: str = ""          # Usage tips for beginners

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
        description="Required for 3D view generation",
        required_for=["3d"],
        vram_gb=2.0,
        quality_tier=QualityTier.STANDARD,
        tips="This is automatically used by the 3D workflow. Download it along with Stable Zero123.",
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
        description="Required for audio generation",
        required_for=["audio"],
        vram_gb=2.0,
        quality_tier=QualityTier.STANDARD,
        tips="This is automatically used by the Audio workflow. Download it along with Stable Audio.",
    ),

    # =========================================================================
    # Checkpoints - SD 1.5 (Beginner Friendly)
    # =========================================================================
    "sd15_base": ModelInfo(
        id="sd15_base",
        name="Stable Diffusion 1.5",
        type=ModelType.CHECKPOINT,
        repo_id="stable-diffusion-v1-5/stable-diffusion-v1-5",
        filename="v1-5-pruned-emaonly.safetensors",
        size_mb=4270,
        description="Best for beginners - fast, low VRAM, great results",
        required_for=["text2img", "img2img"],
        vram_gb=4.0,
        quality_tier=QualityTier.STARTER,
        recommended=True,
        tips="Start here! Works on most GPUs (4GB+ VRAM). Great for learning prompts and settings. Use 512x512 for best results.",
    ),
    "sd15_inpaint": ModelInfo(
        id="sd15_inpaint",
        name="SD 1.5 Inpainting",
        type=ModelType.CHECKPOINT,
        repo_id="stable-diffusion-v1-5/stable-diffusion-inpainting",
        filename="sd-v1-5-inpainting.ckpt",
        size_mb=4270,
        description="Edit parts of images - remove or replace objects",
        required_for=["inpaint"],
        vram_gb=4.0,
        quality_tier=QualityTier.STARTER,
        tips="Use with the Inpainting workflow. Paint white over areas you want to change, then describe what should appear there.",
    ),

    # =========================================================================
    # Checkpoints - SDXL (Higher Quality)
    # =========================================================================
    "sdxl_base": ModelInfo(
        id="sdxl_base",
        name="SDXL Base 1.0",
        type=ModelType.CHECKPOINT,
        repo_id="stabilityai/stable-diffusion-xl-base-1.0",
        filename="sd_xl_base_1.0.safetensors",
        size_mb=6940,
        description="Higher quality images, better text and faces",
        required_for=["text2img"],
        vram_gb=8.0,
        quality_tier=QualityTier.HIGH,
        tips="Produces stunning 1024x1024 images. Needs 8GB+ VRAM. Better at understanding complex prompts and rendering text in images.",
    ),
    "sdxl_refiner": ModelInfo(
        id="sdxl_refiner",
        name="SDXL Refiner 1.0",
        type=ModelType.CHECKPOINT,
        repo_id="stabilityai/stable-diffusion-xl-refiner-1.0",
        filename="sd_xl_refiner_1.0.safetensors",
        size_mb=6080,
        description="Optional add-on to enhance SDXL output details",
        vram_gb=8.0,
        quality_tier=QualityTier.HIGH,
        tips="Advanced: Use after SDXL base to add extra detail. Not required for most uses - the base model alone produces great results.",
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
        description="Create 3D views of objects from a single photo",
        required_for=["3d"],
        vram_gb=6.0,
        quality_tier=QualityTier.STANDARD,
        tips="Upload a photo of an object (ideally on a plain background) and rotate the camera around it. Also requires CLIP ViT-L model.",
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
        description="Generate music and sound effects from text",
        required_for=["audio"],
        vram_gb=6.0,
        quality_tier=QualityTier.STANDARD,
        tips="Describe sounds like 'upbeat electronic music' or 'rain on a window'. Also requires the T5 Base text encoder.",
    ),

    # =========================================================================
    # ControlNet Models (SD 1.5) - Advanced
    # =========================================================================
    "controlnet_canny": ModelInfo(
        id="controlnet_canny",
        name="ControlNet Canny",
        type=ModelType.CONTROLNET,
        repo_id="lllyasviel/control_v11p_sd15_canny",
        filename="diffusion_pytorch_model.safetensors",
        local_filename="control_v11p_sd15_canny.safetensors",
        size_mb=1450,
        description="Guide generation with edge outlines",
        vram_gb=6.0,
        quality_tier=QualityTier.STANDARD,
        tips="Advanced: Extracts edges from your image and generates new content following those lines. Great for architectural drawings.",
    ),
    "controlnet_depth": ModelInfo(
        id="controlnet_depth",
        name="ControlNet Depth",
        type=ModelType.CONTROLNET,
        repo_id="lllyasviel/control_v11f1p_sd15_depth",
        filename="diffusion_pytorch_model.safetensors",
        local_filename="control_v11f1p_sd15_depth.safetensors",
        size_mb=1450,
        description="Guide generation with depth maps",
        vram_gb=6.0,
        quality_tier=QualityTier.STANDARD,
        tips="Advanced: Preserves the 3D layout of a scene while changing its content. Objects stay the same distance from camera.",
    ),
    "controlnet_openpose": ModelInfo(
        id="controlnet_openpose",
        name="ControlNet OpenPose",
        type=ModelType.CONTROLNET,
        repo_id="lllyasviel/control_v11p_sd15_openpose",
        filename="diffusion_pytorch_model.safetensors",
        local_filename="control_v11p_sd15_openpose.safetensors",
        size_mb=1450,
        description="Guide generation with body poses",
        vram_gb=6.0,
        quality_tier=QualityTier.STANDARD,
        tips="Advanced: Detects human poses and generates new people in the same position. Perfect for consistent character poses.",
    ),
    "controlnet_scribble": ModelInfo(
        id="controlnet_scribble",
        name="ControlNet Scribble",
        type=ModelType.CONTROLNET,
        repo_id="lllyasviel/control_v11p_sd15_scribble",
        filename="diffusion_pytorch_model.safetensors",
        local_filename="control_v11p_sd15_scribble.safetensors",
        size_mb=1450,
        description="Turn rough sketches into detailed images",
        vram_gb=6.0,
        quality_tier=QualityTier.STANDARD,
        tips="Advanced: Draw a simple sketch and describe what you want - it fills in all the details while following your drawing.",
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
        description="Make images 4x larger with enhanced details",
        vram_gb=2.0,
        quality_tier=QualityTier.STANDARD,
        tips="Great for enlarging your generated images for printing or sharing. Small download, big impact!",
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
        description="Improves SDXL color accuracy (optional)",
        vram_gb=1.0,
        quality_tier=QualityTier.HIGH,
        tips="Optional upgrade for SDXL. Can reduce color banding in gradients. Most users won't need this.",
    ),
}


def get_models_by_type(model_type: ModelType) -> list[ModelInfo]:
    """Get all models of a specific type."""
    return [m for m in MODEL_CATALOG.values() if m.type == model_type]


def get_recommended_models() -> list[ModelInfo]:
    """Get models marked as recommended for beginners."""
    return [m for m in MODEL_CATALOG.values() if m.recommended]


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
