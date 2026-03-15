"""Generation engine wrapping ComfyUI's PromptExecutor.

This module handles:
- MockServer for headless execution
- Progress callbacks via comfy.utils (Gotcha #4)
- VRAM cleanup after generations
- Interrupt handling
- In-memory image retrieval (Gotcha #3)
"""

import gc
import logging
import signal
import threading
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)

from .comfy_init import (
    clear_captured_images,
    clear_progress_callback,
    get_captured_image,
    get_comfy_context,
    initialize_comfy,
)
from .comfy_init import (
    set_progress_callback as set_comfy_progress_callback,
)


@dataclass
class GenerationResult:
    """Result of a generation execution."""

    prompt_id: str
    success: bool
    outputs: dict
    error: str | None = None
    images: Any | None = None  # PyTorch tensor from ReturnToApp


class ProgressInfo:
    """Progress information for UI updates."""

    def __init__(self):
        self.current_step: int = 0
        self.total_steps: int = 0
        self.current_node: str = ""
        self.preview_image: Any | None = None  # Preview tensor

    def update(self, step: int, total: int, node: str = "", preview: Any | None = None):
        self.current_step = step
        self.total_steps = total
        self.current_node = node
        self.preview_image = preview


class MockServer:
    """Mock server for headless ComfyUI execution.

    ComfyUI's PromptExecutor expects a server object to send WebSocket
    progress updates. This mock intercepts those calls for UI updates.
    """

    def __init__(self, progress_callback: Callable[[ProgressInfo], None] | None = None):
        self.client_id = "switchgen_client"
        self.progress_callback = progress_callback
        self.progress = ProgressInfo()

        # These attributes are accessed by PromptExecutor
        self.last_node_id: str | None = None
        self.last_prompt_id: str | None = None

    def send_sync(self, event: str, data: dict, sid: str | None = None) -> None:
        """Intercept WebSocket events from ComfyUI."""
        if event == "progress":
            value = data.get("value", 0)
            max_value = data.get("max", 0)
            self.progress.update(value, max_value)

            if self.progress_callback:
                self.progress_callback(self.progress)

        elif event == "executing":
            node_id = data.get("node")
            self.last_node_id = node_id
            if node_id:
                self.progress.current_node = node_id

        elif event == "executed" or event == "execution_error":
            pass

    def queue_updated(self) -> None:
        """Called when queue state changes."""
        pass


class GenerationEngine:
    """Main generation engine wrapping ComfyUI's PromptExecutor.

    Handles:
    - Workflow execution via PromptExecutor
    - Progress callbacks via comfy.utils.set_progress_bar_callback (Gotcha #4)
    - VRAM cleanup after each generation
    - Graceful interrupt handling
    - In-memory image capture via ReturnToApp node (Gotcha #3)
    """

    def __init__(self):
        self._initialized = False
        self._executor = None
        self._server: MockServer | None = None
        self._memory_manager = None
        self._interrupted = False
        self._progress_callback: Callable[[ProgressInfo], None] | None = None

    def initialize(self) -> None:
        """Initialize the engine and ComfyUI."""
        if self._initialized:
            return

        logger.info("Initializing generation engine")

        # Initialize ComfyUI (handles all gotchas in comfy_init.py)
        ctx = initialize_comfy()
        self._memory_manager = ctx.memory_manager

        # Import PromptExecutor after ComfyUI is initialized
        from execution import PromptExecutor

        # Create mock server for WebSocket event interception
        self._server = MockServer()

        # Create executor with cache settings
        # ram: GB of RAM to use for caching (default 16.0)
        # lru: number of cached items (0 = disabled)
        cache_args = {"ram": 16.0, "lru": 0}
        self._executor = PromptExecutor(server=self._server, cache_args=cache_args)

        # Setup signal handlers for graceful interrupts
        self._setup_signal_handlers()

        self._initialized = True
        logger.info("Generation engine initialized successfully")

    def _setup_signal_handlers(self) -> None:
        """Setup graceful interrupt handling."""
        import threading

        # Signal handlers can only be set from the main thread
        if threading.current_thread() is not threading.main_thread():
            return

        original_handler = signal.getsignal(signal.SIGINT)

        def handler(signum, frame):
            print("\nSwitchGen: Interrupt received, stopping generation...")
            self._interrupted = True
            if self._memory_manager:
                self._memory_manager.interrupt_current_processing()

            if callable(original_handler) and original_handler not in (
                signal.SIG_IGN,
                signal.SIG_DFL,
            ):
                original_handler(signum, frame)

        signal.signal(signal.SIGINT, handler)

    def set_progress_callback(self, callback: Callable[[ProgressInfo], None] | None) -> None:
        """Set callback for progress updates.

        CRITICAL (Gotcha #4): This uses comfy.utils.set_progress_bar_callback
        to hook into KSampler's step-by-step progress reporting.
        """
        self._progress_callback = callback

        if self._server:
            self._server.progress_callback = callback

        # Also set the comfy.utils callback for KSampler progress
        if callback:

            def comfy_callback(step: int, total: int, preview: Any) -> None:
                info = ProgressInfo()
                info.update(step, total, preview=preview)
                callback(info)

            set_comfy_progress_callback(comfy_callback)
        else:
            clear_progress_callback()

    def cleanup_vram(self) -> None:
        """Clean up VRAM after generation.

        Must be called after generation to prevent VRAM fragmentation and leaks.
        """
        if self._memory_manager:
            self._memory_manager.soft_empty_cache()
        gc.collect()

    def get_vram_usage(self) -> tuple[int, int]:
        """Get current VRAM usage (used, total) in bytes."""
        if not self._memory_manager:
            return (0, 0)

        ctx = get_comfy_context()
        try:
            total = self._memory_manager.get_total_memory(ctx.device)
            free = self._memory_manager.get_free_memory(ctx.device)
            used = total - free
            return (used, total)
        except Exception:
            return (0, 0)

    def get_vram_usage_percent(self) -> float:
        """Get VRAM usage as percentage."""
        used, total = self.get_vram_usage()
        if total == 0:
            return 0.0
        return (used / total) * 100

    def check_vram_available(self, workflow: dict) -> tuple[bool, str]:
        """Check if sufficient VRAM is available for a workflow.

        Returns:
            (ok, message) — ok is True if enough VRAM, message explains if not.
        """
        if not self._memory_manager:
            return True, ""

        ctx = get_comfy_context()
        try:
            free = self._memory_manager.get_free_memory(ctx.device)
        except Exception:
            return True, ""  # Can't check, assume OK

        # Estimate minimum VRAM: 2GB for SD 1.5, 4GB for SDXL
        min_required = 2 * 1024**3
        for node_data in workflow.values():
            ckpt = str(node_data.get("inputs", {}).get("ckpt_name", "")).lower()
            if "xl" in ckpt or "sdxl" in ckpt or "playground-v2" in ckpt:
                min_required = 4 * 1024**3
                break

        if free < min_required:
            return (
                False,
                f"Low VRAM: {free / 1024**3:.1f}GB free, need ~{min_required / 1024**3:.0f}GB",
            )
        return True, ""

    def execute(
        self,
        workflow: dict,
        extra_data: dict | None = None,
        timeout: float = 600.0,
    ) -> GenerationResult:
        """Execute a workflow in API format.

        Args:
            workflow: ComfyUI workflow in API format (from "Save API Format")
            extra_data: Optional extra data to pass to executor
            timeout: Maximum execution time in seconds (default 10 minutes)

        Returns:
            GenerationResult with outputs, captured images, or error
        """
        if not self._initialized:
            self.initialize()

        self._interrupted = False
        prompt_id = str(uuid.uuid4())
        capture_id = prompt_id  # Use unique prompt_id to avoid collisions
        start_time = time.time()

        # Start timeout watchdog
        watchdog_cancelled = threading.Event()

        def _watchdog():
            if not watchdog_cancelled.wait(timeout):
                logger.warning("Execution timeout reached (%.0fs), interrupting", timeout)
                self.interrupt()

        watchdog_thread = threading.Thread(target=_watchdog, daemon=True)
        watchdog_thread.start()

        logger.info("Starting generation (prompt_id=%s, nodes=%d)", prompt_id, len(workflow))

        if extra_data is None:
            extra_data = {}

        # Clear any previous captured images
        clear_captured_images()

        # Inject unique capture_id into ReturnToApp nodes to prevent collisions
        for node_data in workflow.values():
            if node_data.get("class_type") == "ReturnToApp":
                node_data.setdefault("inputs", {})["capture_id"] = capture_id

        try:
            # Find output nodes to execute (nodes with OUTPUT_NODE=True)
            import nodes

            execute_outputs = []
            for node_id, node_data in workflow.items():
                class_type = node_data.get("class_type")
                if class_type in nodes.NODE_CLASS_MAPPINGS:
                    cls = nodes.NODE_CLASS_MAPPINGS[class_type]
                    if getattr(cls, "OUTPUT_NODE", False):
                        execute_outputs.append(node_id)

            logger.debug("Executing workflow with %d output nodes", len(execute_outputs))

            # Execute the workflow
            self._executor.execute(
                workflow,
                prompt_id=prompt_id,
                extra_data=extra_data,
                execute_outputs=execute_outputs,
            )

            # Check if interrupted after execution returns
            if self._interrupted:
                logger.warning("Generation interrupted by user")
                clear_captured_images()
                return GenerationResult(
                    prompt_id=prompt_id,
                    success=False,
                    outputs={},
                    error="Generation interrupted by user",
                )

            # Get captured images from ReturnToApp node (Gotcha #3)
            captured = get_captured_image(capture_id)

            # Double-check: interrupt could have been set during image capture
            if self._interrupted:
                logger.warning("Generation interrupted during capture")
                clear_captured_images()
                return GenerationResult(
                    prompt_id=prompt_id,
                    success=False,
                    outputs={},
                    error="Generation interrupted by user",
                )

            elapsed = time.time() - start_time
            image_count = captured.shape[0] if captured is not None else 0
            logger.info(
                "Generation completed successfully (prompt_id=%s, images=%d, time=%.2fs)",
                prompt_id,
                image_count,
                elapsed,
            )

            return GenerationResult(prompt_id=prompt_id, success=True, outputs={}, images=captured)

        except KeyboardInterrupt:
            logger.warning("Generation interrupted by keyboard")
            if self._memory_manager:
                self._memory_manager.interrupt_current_processing()
            return GenerationResult(
                prompt_id=prompt_id, success=False, outputs={}, error="Generation interrupted"
            )

        except Exception as e:
            logger.error("Generation failed: %s", e, exc_info=True)
            return GenerationResult(prompt_id=prompt_id, success=False, outputs={}, error=str(e))

        finally:
            # Cancel timeout watchdog
            watchdog_cancelled.set()
            # Always cleanup VRAM after generation
            self.cleanup_vram()

    def execute_safe(self, workflow: dict) -> GenerationResult:
        """Execute with full interrupt handling."""
        return self.execute(workflow)

    def unload_all_models(self) -> None:
        """Unload all models from VRAM."""
        if self._memory_manager:
            self._memory_manager.unload_all_models()
            self.cleanup_vram()

    def interrupt(self) -> None:
        """Interrupt current generation."""
        self._interrupted = True
        if self._memory_manager:
            self._memory_manager.interrupt_current_processing()


def tensor_to_pil(tensor: Any) -> list:
    """Convert ComfyUI image tensor to PIL Images.

    Args:
        tensor: PyTorch tensor (Batch, Height, Width, Channels) in [0, 1] range

    Returns:
        List of PIL Image objects
    """
    import numpy as np
    from PIL import Image

    if tensor is None:
        return []

    # Ensure tensor is on CPU and convert to numpy
    if hasattr(tensor, "cpu"):
        tensor = tensor.cpu()
    if hasattr(tensor, "numpy"):
        tensor = tensor.numpy()

    # ComfyUI format: (B, H, W, C) with values in [0, 1]
    images = []
    for i in range(tensor.shape[0]):
        img_array = (tensor[i] * 255).astype(np.uint8)
        images.append(Image.fromarray(img_array))

    return images


# Global engine instance
_engine: GenerationEngine | None = None


def get_engine() -> GenerationEngine:
    """Get the global engine instance."""
    global _engine
    if _engine is None:
        _engine = GenerationEngine()
    return _engine
