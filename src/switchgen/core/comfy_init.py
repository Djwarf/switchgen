"""ComfyUI initialization for headless use.

This module handles the critical initialization steps required to use ComfyUI as a library.

CRITICAL GOTCHAS ADDRESSED:
1. Argument Parsing - sys.argv must be hijacked BEFORE importing comfy.cli_args
2. Async Node Loading - nodes.init_extra_nodes() must run in asyncio
3. In-Memory Images - ReturnToApp custom node captures images without disk I/O
4. Progress Callbacks - comfy.utils.set_progress_bar_global_hook for UI updates
5. Seeds - Must always be explicit integers (handled in workflows.py)
"""

import os

# CRITICAL: CUDA/PyTorch optimizations - MUST be set before torch is imported
# 1. Lazy loading prevents CUDA from grabbing all VRAM at startup
os.environ["CUDA_MODULE_LOADING"] = "LAZY"
# 2. Expandable segments prevents "Out of Memory" crashes due to VRAM fragmentation
#    (essential when switching between different image sizes/models)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import sys
import asyncio
from pathlib import Path
from typing import Any, Callable, Optional
import threading

from .config import Config, get_config


class ComfyInitError(Exception):
    """Error during ComfyUI initialization."""
    pass


class ComfyContext:
    """Holds initialized ComfyUI components."""

    def __init__(self):
        self.device: Any = None
        self.node_classes: dict = {}
        self.memory_manager: Any = None
        self.folder_paths: Any = None
        self.initialized: bool = False
        self.comfy_utils: Any = None  # For progress callback


# Global storage for captured images (Gotcha #3)
_captured_images: dict[str, Any] = {}
_captured_images_lock = threading.Lock()

_comfy_context: Optional[ComfyContext] = None
_original_argv: list[str] = []


def _hijack_argv() -> list[str]:
    """Hijack sys.argv to prevent ComfyUI from parsing our arguments.

    CRITICAL (Gotcha #1): ComfyUI's cli_args module parses sys.argv on import.
    If our app has its own CLI flags, ComfyUI will crash with "unrecognized argument".

    Returns:
        The original argv (our app's arguments) for later use
    """
    global _original_argv
    _original_argv = sys.argv[1:]  # Save our arguments
    sys.argv = [sys.argv[0]]  # Trick ComfyUI into thinking there are no args
    return _original_argv


def _restore_argv() -> None:
    """Restore original argv after ComfyUI imports are done."""
    global _original_argv
    sys.argv = [sys.argv[0]] + _original_argv


def _add_comfy_to_path(config: Config) -> None:
    """Add ComfyUI to Python path."""
    comfy_path = str(config.paths.comfy_path)
    if comfy_path not in sys.path:
        sys.path.insert(0, comfy_path)


def _configure_paths(config: Config) -> Any:
    """Configure ComfyUI's folder_paths for model discovery."""
    import folder_paths

    # Models are stored in the data root (XDG or repo root depending on install type)
    models_base = config.paths.data_root

    # Model type paths - ComfyUI expects these to be configured
    model_paths = {
        "checkpoints": "models/checkpoints",
        "loras": "models/loras",
        "vae": "models/vae",
        "clip": "models/clip",
        "text_encoders": "models/text_encoders",  # For CLIPLoader (T5, etc.)
        "controlnet": "models/controlnet",
        "embeddings": "models/embeddings",
        "clip_vision": "models/clip_vision",
        "style_models": "models/style_models",
        "diffusers": "models/diffusers",
        "gligen": "models/gligen",
        "hypernetworks": "models/hypernetworks",
        "upscale_models": "models/upscale_models",
    }

    for folder_type, rel_path in model_paths.items():
        full_path = str(models_base / rel_path)
        if folder_type in folder_paths.folder_names_and_paths:
            folder_paths.folder_names_and_paths[folder_type][0][0] = full_path
        else:
            folder_paths.folder_names_and_paths[folder_type] = ([full_path], set())

    # Set output, input, and temp directories
    folder_paths.set_output_directory(str(config.paths.output_dir))
    folder_paths.set_input_directory(str(config.paths.input_dir))
    folder_paths.set_temp_directory(str(config.paths.temp_dir))

    # Custom nodes path
    custom_nodes_path = str(config.paths.custom_nodes_dir)
    folder_paths.folder_names_and_paths["custom_nodes"] = ([custom_nodes_path], set())

    return folder_paths


def _register_custom_output_node() -> None:
    """Register the ReturnToApp node for capturing images in memory.

    CRITICAL (Gotcha #3): SaveImage writes to disk. For GUI apps, we need
    images in memory. This custom node intercepts the image tensor.
    """
    import nodes

    class ReturnToApp:
        """Custom node that captures images to memory instead of saving to disk."""

        @classmethod
        def INPUT_TYPES(cls):
            return {
                "required": {
                    "images": ("IMAGE",),
                },
                "optional": {
                    "capture_id": ("STRING", {"default": "default"}),
                }
            }

        RETURN_TYPES = ()
        OUTPUT_NODE = True
        FUNCTION = "capture"
        CATEGORY = "switchgen"

        def capture(self, images, capture_id="default"):
            """Capture images to global storage.

            Args:
                images: PyTorch tensor (Batch, Height, Width, Channels)
                capture_id: Identifier for retrieving this capture
            """
            with _captured_images_lock:
                _captured_images[capture_id] = images.clone()
            return {}

    # Register the node
    nodes.NODE_CLASS_MAPPINGS["ReturnToApp"] = ReturnToApp
    nodes.NODE_DISPLAY_NAME_MAPPINGS["ReturnToApp"] = "Return To App"


def _load_node_registry(config: Config, load_custom_nodes: bool = True) -> dict:
    """Load core nodes AND custom nodes.

    CRITICAL (Gotcha #2): Standard `import nodes` only loads ~100 core nodes.
    Custom nodes and comfy_extras require calling nodes.init_extra_nodes()
    which is ASYNC and must run in an asyncio event loop.
    """
    # CRITICAL: Create and set an event loop for this thread BEFORE importing nodes
    # Some custom nodes check for an event loop during import
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)

    import nodes

    node_count_before = len(nodes.NODE_CLASS_MAPPINGS)

    if load_custom_nodes:
        try:
            # init_extra_nodes() loads BOTH custom_nodes AND comfy_extras
            if hasattr(nodes, 'init_extra_nodes'):
                print("SwitchGen: Loading extra nodes (async)...")

                async def load_nodes():
                    await nodes.init_extra_nodes()

                # Run in the event loop we created/got
                loop.run_until_complete(load_nodes())

                node_count_after = len(nodes.NODE_CLASS_MAPPINGS)
                extra_loaded = node_count_after - node_count_before
                print(f"SwitchGen: Loaded {extra_loaded} extra node classes")
            else:
                print("SwitchGen: Warning - init_extra_nodes not found")

        except Exception as e:
            print(f"SwitchGen: Warning - Could not load custom nodes: {e}")

    # Register our custom output node (Gotcha #3)
    _register_custom_output_node()

    return nodes.NODE_CLASS_MAPPINGS


def _initialize_device() -> tuple[Any, Any]:
    """Initialize torch device and memory manager."""
    import comfy.model_management as mm

    device = mm.get_torch_device()

    print(f"SwitchGen: Using device: {device}")
    print(f"SwitchGen: VRAM state: {mm.vram_state.name}")

    try:
        total_vram = mm.get_total_memory(device)
        free_vram = mm.get_free_memory(device)
        print(f"SwitchGen: Total VRAM: {total_vram / (1024**3):.1f} GB")
        print(f"SwitchGen: Free VRAM: {free_vram / (1024**3):.1f} GB")
    except Exception as e:
        print(f"SwitchGen: Could not query VRAM: {e}")

    return device, mm


def _setup_progress_callback() -> Any:
    """Set up progress callback infrastructure.

    CRITICAL (Gotcha #4): Without this, the app appears frozen during generation.
    The KSampler calculates step-by-step but doesn't report progress unless
    we register a callback via comfy.utils.set_progress_bar_global_hook.
    """
    import comfy.utils
    return comfy.utils


def initialize_comfy(
    config: Optional[Config] = None,
    load_custom_nodes: bool = True
) -> ComfyContext:
    """Full ComfyUI initialization for headless use.

    This handles all critical gotchas:
    1. Hijacks sys.argv before ComfyUI parses it
    2. Loads custom nodes via async init_extra_nodes()
    3. Registers ReturnToApp node for in-memory image capture
    4. Sets up progress callback infrastructure
    5. Seeds are handled in workflows.py (always explicit integers)

    Args:
        config: Configuration object. If None, uses global config.
        load_custom_nodes: Whether to load custom nodes (slower but more features)

    Returns:
        ComfyContext with all initialized components

    Raises:
        ComfyInitError: If initialization fails
    """
    global _comfy_context

    if _comfy_context is not None and _comfy_context.initialized:
        return _comfy_context

    if config is None:
        config = get_config()

    ctx = ComfyContext()

    try:
        # CRITICAL: Ensure this thread has an event loop
        # Some custom nodes check for an event loop during import
        try:
            asyncio.get_event_loop()
        except RuntimeError:
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)

        print("SwitchGen: Initializing ComfyUI...")

        # Step 0: Add ComfyUI to path
        _add_comfy_to_path(config)

        # CRITICAL Step 1: Hijack argv BEFORE any comfy imports (Gotcha #1)
        print("SwitchGen: Hijacking sys.argv for ComfyUI...")
        _hijack_argv()

        # Now safe to import comfy modules
        import comfy.cli_args  # This parses argv, but we've hidden our args

        # Step 2: Configure paths
        print("SwitchGen: Configuring paths...")
        ctx.folder_paths = _configure_paths(config)

        # Step 3: Load node registry including custom nodes (Gotcha #2)
        print("SwitchGen: Loading node registry...")
        ctx.node_classes = _load_node_registry(config, load_custom_nodes)
        print(f"SwitchGen: Loaded {len(ctx.node_classes)} node classes")

        # Step 4: Initialize device and memory manager
        print("SwitchGen: Initializing device...")
        ctx.device, ctx.memory_manager = _initialize_device()

        # Step 5: Set up progress callback infrastructure (Gotcha #4)
        print("SwitchGen: Setting up progress callbacks...")
        ctx.comfy_utils = _setup_progress_callback()

        # Restore original argv
        _restore_argv()

        ctx.initialized = True
        _comfy_context = ctx

        print("SwitchGen: ComfyUI initialization complete")
        return ctx

    except ImportError as e:
        _restore_argv()  # Restore even on error
        raise ComfyInitError(
            f"Failed to import ComfyUI. Is it installed at {config.paths.comfy_path}? "
            f"Error: {e}"
        )
    except Exception as e:
        _restore_argv()
        raise ComfyInitError(f"ComfyUI initialization failed: {e}")


def get_comfy_context() -> ComfyContext:
    """Get the initialized ComfyUI context."""
    if _comfy_context is None or not _comfy_context.initialized:
        raise ComfyInitError("ComfyUI not initialized. Call initialize_comfy() first.")
    return _comfy_context


def set_progress_callback(callback: Callable[[int, int, Any], None]) -> None:
    """Set the global progress callback for generation.

    CRITICAL (Gotcha #4): This hooks into KSampler's step-by-step progress.

    Args:
        callback: Function(step, total, preview_image) called during sampling
    """
    ctx = get_comfy_context()

    # Wrap the callback to match the new API signature
    # New API: hook(current, total, preview, node_id=None)
    def wrapped_callback(current, total, preview, node_id=None):
        callback(current, total, preview)

    ctx.comfy_utils.set_progress_bar_global_hook(wrapped_callback)


def clear_progress_callback() -> None:
    """Clear the progress callback."""
    ctx = get_comfy_context()
    ctx.comfy_utils.set_progress_bar_global_hook(None)


def get_captured_image(capture_id: str = "default") -> Optional[Any]:
    """Get an image captured by ReturnToApp node.

    Args:
        capture_id: The capture_id used in the workflow

    Returns:
        PyTorch tensor (Batch, Height, Width, Channels) or None
    """
    with _captured_images_lock:
        return _captured_images.get(capture_id)


def clear_captured_images() -> None:
    """Clear all captured images from memory."""
    with _captured_images_lock:
        _captured_images.clear()


def get_available_checkpoints() -> list[str]:
    """Get list of available checkpoint files."""
    get_comfy_context()  # Ensure initialized
    import folder_paths
    return folder_paths.get_filename_list("checkpoints")


def get_available_loras() -> list[str]:
    """Get list of available LoRA files."""
    get_comfy_context()
    import folder_paths
    return folder_paths.get_filename_list("loras")


def get_available_vaes() -> list[str]:
    """Get list of available VAE files."""
    get_comfy_context()
    import folder_paths
    return folder_paths.get_filename_list("vae")
