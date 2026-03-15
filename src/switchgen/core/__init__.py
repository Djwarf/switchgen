"""Core generation engine and ComfyUI integration."""

from .comfy_init import (
    clear_captured_images,
    clear_progress_callback,
    get_available_checkpoints,
    get_available_loras,
    get_captured_image,
    get_comfy_context,
    initialize_comfy,
    set_progress_callback,
)
from .config import Config, get_config
from .engine import (
    GenerationEngine,
    GenerationResult,
    ProgressInfo,
    get_engine,
    tensor_to_pil,
)
from .queue import GenerationJob, GenerationQueue
from .workflows import (
    WORKFLOW_SPECS,
    WorkflowBuilder,
    # Workflow management
    WorkflowManager,
    WorkflowSpec,
    # Workflow type system
    WorkflowType,
    build_3d_zero123_workflow,
    build_audio_workflow,
    build_img2img_memory_workflow,
    build_img2img_workflow,
    build_inpaint_workflow,
    build_text2img_memory_workflow,
    # Workflow builders
    build_text2img_workflow,
    ensure_seed,
    # Seed utilities
    generate_seed,
    get_compatible_workflows,
    get_models_for_workflow,
    get_workflow_spec,
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
