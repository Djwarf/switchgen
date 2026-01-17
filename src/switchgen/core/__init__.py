"""Core generation engine and ComfyUI integration."""

from .comfy_init import (
    initialize_comfy,
    get_comfy_context,
    set_progress_callback,
    clear_progress_callback,
    get_captured_image,
    clear_captured_images,
    get_available_checkpoints,
    get_available_loras,
)
from .engine import (
    GenerationEngine,
    GenerationResult,
    ProgressInfo,
    get_engine,
    tensor_to_pil,
)
from .config import Config, get_config
from .queue import GenerationQueue, GenerationJob
from .workflows import (
    # Workflow type system
    WorkflowType,
    WorkflowSpec,
    WORKFLOW_SPECS,
    get_workflow_spec,
    get_compatible_workflows,
    get_models_for_workflow,
    # Workflow management
    WorkflowManager,
    WorkflowBuilder,
    # Workflow builders
    build_text2img_workflow,
    build_text2img_memory_workflow,
    build_img2img_workflow,
    build_img2img_memory_workflow,
    build_inpaint_workflow,
    build_audio_workflow,
    build_3d_zero123_workflow,
    # Seed utilities
    generate_seed,
    ensure_seed,
)

__all__ = [
    # Initialization
    "initialize_comfy",
    "get_comfy_context",
    # Progress & Image capture
    "set_progress_callback",
    "clear_progress_callback",
    "get_captured_image",
    "clear_captured_images",
    # Model listing
    "get_available_checkpoints",
    "get_available_loras",
    # Engine
    "GenerationEngine",
    "GenerationResult",
    "ProgressInfo",
    "get_engine",
    "tensor_to_pil",
    # Config
    "Config",
    "get_config",
    # Queue
    "GenerationQueue",
    "GenerationJob",
    # Workflow type system
    "WorkflowType",
    "WorkflowSpec",
    "WORKFLOW_SPECS",
    "get_workflow_spec",
    "get_compatible_workflows",
    "get_models_for_workflow",
    # Workflow management
    "WorkflowManager",
    "WorkflowBuilder",
    # Workflow builders
    "build_text2img_workflow",
    "build_text2img_memory_workflow",
    "build_img2img_workflow",
    "build_img2img_memory_workflow",
    "build_inpaint_workflow",
    "build_audio_workflow",
    "build_3d_zero123_workflow",
    # Seed utilities
    "generate_seed",
    "ensure_seed",
]
