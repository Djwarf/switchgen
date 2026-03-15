"""Main application window.

This module wires together the ControlPanel, PreviewPanel, and BottomBar widgets,
handles ComfyUI initialization, generation orchestration, VRAM monitoring,
keyboard shortcuts, and session persistence.
"""

import io
import threading

from ..core.logging import get_logger

logger = get_logger(__name__)

try:
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Adw, GdkPixbuf, GLib, Gtk
except (ImportError, ValueError):
    pass

from ..core import (
    WORKFLOW_SPECS,
    ProgressInfo,
    WorkflowType,
    build_3d_zero123_workflow,
    build_audio_workflow,
    build_img2img_memory_workflow,
    build_inpaint_workflow,
    build_text2img_memory_workflow,
    get_available_checkpoints,
    tensor_to_pil,
)
from ..core.config import get_config
from ..core.queue import GenerationJob, get_queue
from .model_dialog import ModelDownloadDialog
from .widgets import BottomBar, ControlPanel, PreviewPanel


class MainWindow(Adw.ApplicationWindow):
    """Main SwitchGen window.

    Composes ControlPanel (left), PreviewPanel (right), and BottomBar (bottom).
    Owns ComfyUI initialization, generation orchestration, VRAM timer, keyboard
    shortcuts, and session persistence.
    """

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_title("SwitchGen")

        # Restore window dimensions from session
        config = get_config()
        self.set_default_size(config.session.window_width, config.session.window_height)

        # State
        self._generating = False
        self._queue = None
        self._current_seed: int | None = None

        # Build composite UI
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        # Header bar
        header = Adw.HeaderBar()
        header.set_title_widget(Gtk.Label(label="SWITCHGEN", css_classes=["title-1"]))
        download_btn = Gtk.Button(icon_name="folder-download-symbolic")
        download_btn.set_tooltip_text("Download Models (Ctrl+D)")
        download_btn.connect("clicked", self._on_download_models_clicked)
        header.pack_end(download_btn)
        main_box.append(header)

        # Paned: ControlPanel | PreviewPanel
        self._paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        self._paned.set_shrink_start_child(False)
        self._paned.set_shrink_end_child(False)
        self._paned.set_position(config.session.paned_position)

        self.control_panel = ControlPanel()
        self.preview_panel = PreviewPanel()
        self._paned.set_start_child(self.control_panel)
        self._paned.set_end_child(self.preview_panel)
        main_box.append(self._paned)

        # Bottom bar
        self.bottom_bar = BottomBar()
        main_box.append(self.bottom_bar)

        # Wire callbacks
        self.control_panel.on_generate_requested = self._on_generate
        self.control_panel.on_workflow_changed = self._on_workflow_changed
        self.bottom_bar.on_generate_clicked = self._on_generate
        self.bottom_bar.on_banner_button_clicked = self._on_download_models_clicked

        # Keyboard shortcuts
        self._setup_shortcuts()

        # Restore session state into controls
        self.control_panel.restore_session(
            workflow=config.session.last_workflow,
            prompt=config.session.last_prompt,
            negative=config.session.last_negative,
            style=config.session.last_style,
        )

        # Initialize ComfyUI in the background
        self.bottom_bar.set_generate_sensitive(False)
        self.bottom_bar.set_generate_label("Initializing...")
        threading.Thread(target=self._init_comfy, daemon=True).start()
        GLib.timeout_add(1000, self._update_vram)

    # =========================================================================
    # Keyboard shortcuts
    # =========================================================================

    def _setup_shortcuts(self) -> None:
        """Register keyboard shortcuts via a Gtk.ShortcutController."""
        controller = Gtk.ShortcutController.new()
        controller.set_scope(Gtk.ShortcutScope.MANAGED)
        self.add_controller(controller)

        # Ctrl+Enter -> Generate
        controller.add_shortcut(
            Gtk.Shortcut.new(
                Gtk.ShortcutTrigger.parse_string("<Control>Return"),
                Gtk.CallbackAction.new(self._shortcut_generate),
            )
        )

        # Escape -> Cancel current generation
        controller.add_shortcut(
            Gtk.Shortcut.new(
                Gtk.ShortcutTrigger.parse_string("Escape"),
                Gtk.CallbackAction.new(self._shortcut_cancel),
            )
        )

        # Ctrl+D -> Download dialog
        controller.add_shortcut(
            Gtk.Shortcut.new(
                Gtk.ShortcutTrigger.parse_string("<Control>d"),
                Gtk.CallbackAction.new(self._shortcut_download),
            )
        )

    def _shortcut_generate(self, _widget, _args) -> bool:
        self._on_generate()
        return True

    def _shortcut_cancel(self, _widget, _args) -> bool:
        if self._generating and self._queue:
            logger.info("Cancel requested via Escape")
            self._queue.cancel_current()
            self._reset_ui()
        return True

    def _shortcut_download(self, _widget, _args) -> bool:
        self._on_download_models_clicked()
        return True

    # =========================================================================
    # Workflow change
    # =========================================================================

    def _on_workflow_changed(self) -> None:
        """Called by ControlPanel after the workflow selector changes."""
        # Nothing extra needed here; the ControlPanel manages its own UI.
        pass

    # =========================================================================
    # Generation orchestration
    # =========================================================================

    def _on_generate(self, *_args) -> None:
        """Build workflow from control params and submit to queue."""
        if self._generating or not self._queue:
            return

        params = self.control_panel.get_generation_params()

        # Validate checkpoint
        if not params["checkpoint"]:
            self.bottom_bar.show_info("Please download a model first to generate images.")
            logger.warning("Generate clicked but no model selected")
            return

        checkpoint = params["checkpoint"]
        workflow_type: WorkflowType = params["workflow"]
        prompt = params["prompt"]
        negative = params["negative"]
        steps = params["steps"]
        cfg = params["cfg"]
        seed = params["seed"]

        # Apply style suffix
        if params["style_suffix"]:
            prompt = prompt.rstrip() + params["style_suffix"]
            logger.debug("Style suffix applied")

        # Validate requirements
        spec = WORKFLOW_SPECS[workflow_type]
        if spec.needs_input_image and not params["input_image_path"]:
            return
        if spec.needs_mask and not params["mask_image_path"]:
            return

        logger.info("Generate clicked (workflow=%s, model=%s)", workflow_type.name, checkpoint)

        # Mark generating
        self._generating = True
        self.bottom_bar.set_generating(True)
        self.preview_panel.set_progress(0, "Starting...")

        # Build workflow
        workflow = None
        actual_seed = seed

        if workflow_type == WorkflowType.TEXT2IMG:
            workflow, actual_seed = build_text2img_memory_workflow(
                checkpoint=checkpoint,
                prompt=prompt,
                negative_prompt=negative,
                width=params["width"],
                height=params["height"],
                steps=steps,
                cfg=cfg,
                seed=seed,
                capture_id="default",
            )
        elif workflow_type == WorkflowType.IMG2IMG:
            workflow, actual_seed = build_img2img_memory_workflow(
                checkpoint=checkpoint,
                image_path=params["input_image_path"],
                prompt=prompt,
                negative_prompt=negative,
                denoise=params["denoise"],
                steps=steps,
                cfg=cfg,
                seed=seed,
                capture_id="default",
            )
        elif workflow_type == WorkflowType.INPAINT:
            workflow, actual_seed = build_inpaint_workflow(
                checkpoint=checkpoint,
                image_path=params["input_image_path"],
                mask_path=params["mask_image_path"],
                prompt=prompt,
                negative_prompt=negative,
                denoise=params["denoise"],
                steps=steps,
                cfg=cfg,
                seed=seed,
                capture_id="default",
            )
        elif workflow_type == WorkflowType.AUDIO:
            workflow, actual_seed = build_audio_workflow(
                checkpoint=checkpoint,
                prompt=prompt,
                negative_prompt=negative,
                seconds=params["duration"],
                steps=steps,
                cfg=cfg,
                seed=seed,
            )
        elif workflow_type == WorkflowType.THREE_D:
            workflow, actual_seed = build_3d_zero123_workflow(
                checkpoint=checkpoint,
                image_path=params["input_image_path"],
                elevation=params["elevation"],
                azimuth=params["azimuth"],
                steps=steps,
                cfg=cfg,
                seed=seed,
                capture_id="default",
            )

        if not workflow:
            self._reset_ui()
            return

        self._current_seed = actual_seed

        def on_progress(info: ProgressInfo):
            GLib.idle_add(self._update_progress, info)

        def on_complete(job: GenerationJob):
            GLib.idle_add(self._on_complete, job)

        logger.info(
            "Submitting generation (workflow=%s, model=%s, steps=%d, cfg=%.1f, seed=%s)",
            workflow_type.name,
            checkpoint,
            steps,
            cfg,
            actual_seed,
        )
        self._queue.submit(workflow=workflow, on_progress=on_progress, on_complete=on_complete)

    def _update_progress(self, info: ProgressInfo) -> bool:
        if info.total_steps > 0:
            progress = info.current_step / info.total_steps
            pct = int(progress * 100)
            self.preview_panel.set_progress(progress, f"Generating... {pct}%")
            self.bottom_bar.set_generate_label(f"Generating... {pct}%")
        return False

    def _on_complete(self, job: GenerationJob) -> bool:
        logger.info("_on_complete called: result=%s", job.result is not None)
        if job.result:
            logger.info(
                "  success=%s, images=%s, error=%s",
                job.result.success,
                job.result.images is not None,
                job.result.error,
            )
        if job.result and job.result.success and job.result.images is not None:
            try:
                images = tensor_to_pil(job.result.images)
                if images:
                    logger.info("Generation completed successfully (images=%d)", len(images))
                    self._show_image(images[0])
                else:
                    logger.warning("Generation completed but tensor_to_pil returned empty")
            except Exception as e:
                logger.error("Error converting images: %s", e, exc_info=True)
        elif job.result and not job.result.success:
            logger.error("Generation failed: %s", job.result.error)
        else:
            logger.warning("No result or images in job")
        self._reset_ui()
        return False

    def _show_image(self, pil_image) -> None:
        """Convert a PIL image to a GdkPixbuf and show it in the preview."""
        buf = io.BytesIO()
        pil_image.save(buf, format="PNG")
        buf.seek(0)
        loader = GdkPixbuf.PixbufLoader.new_with_type("png")
        loader.write(buf.read())
        loader.close()
        pixbuf = loader.get_pixbuf()

        self.preview_panel.show_image(pixbuf)

        if self._current_seed is not None:
            self.control_panel.set_seed_text(str(self._current_seed))

    def _reset_ui(self) -> None:
        """Reset UI after generation completes or is cancelled."""
        self._generating = False
        self.preview_panel.reset()
        if self.control_panel.has_valid_checkpoint:
            self.bottom_bar.set_generating(False)
        else:
            self.bottom_bar.set_generate_sensitive(False)
            self.bottom_bar.set_generate_label("No Models")

    # =========================================================================
    # Model download
    # =========================================================================

    def _on_download_models_clicked(self, *_args) -> None:
        """Open the model download dialog."""
        config = get_config()
        dialog = ModelDownloadDialog(config.paths.models_dir)
        dialog.connect("closed", self._on_download_dialog_closed)
        dialog.present(self)

    def _on_download_dialog_closed(self, _dialog) -> None:
        """Refresh the model list after downloading."""
        self._refresh_models()

    def _refresh_models(self) -> None:
        """Re-scan checkpoints and update the UI accordingly."""
        checkpoints = get_available_checkpoints()
        self.control_panel.set_checkpoints(checkpoints)

        # Trigger a dropdown refresh for the current workflow
        self.control_panel._on_workflow_changed(self.control_panel._workflow_dropdown, None)

        if checkpoints:
            self.bottom_bar.set_generating(False)
            self.bottom_bar.show_info("")
        else:
            self.bottom_bar.set_generate_sensitive(False)
            self.bottom_bar.set_generate_label("No Models")
            self.bottom_bar.show_info(
                "No models installed. Click the download button to get started."
            )

    # =========================================================================
    # ComfyUI Initialization
    # =========================================================================

    def _init_comfy(self) -> None:
        """Initialize ComfyUI in a background thread."""
        logger.info("Initializing ComfyUI...")
        try:
            self._queue = get_queue()
            checkpoints = get_available_checkpoints()
            logger.info("ComfyUI initialized (checkpoints=%d)", len(checkpoints))
            GLib.idle_add(self._on_ready, checkpoints)
        except Exception as e:
            logger.error("ComfyUI initialization failed: %s", e, exc_info=True)
            GLib.idle_add(self._on_error, str(e))

    def _on_ready(self, checkpoints: list[str]) -> bool:
        """ComfyUI ready -- runs on main thread."""
        logger.info("Main window ready")
        self.control_panel.set_checkpoints(checkpoints)
        # Trigger workflow change to filter models
        self.control_panel._on_workflow_changed(self.control_panel._workflow_dropdown, None)
        self._update_vram()

        if not checkpoints:
            self.bottom_bar.set_generate_sensitive(False)
            self.bottom_bar.set_generate_label("No Models")
            self.bottom_bar.show_info(
                "No models installed. Click the download button to get started."
            )
            GLib.idle_add(self._show_welcome_download_dialog)
        else:
            self.bottom_bar.set_generating(False)
            self.bottom_bar.show_info("")

        return False

    def _show_welcome_download_dialog(self) -> bool:
        """Auto-open download dialog for first-time users with no models."""
        config = get_config()
        dialog = ModelDownloadDialog(config.paths.models_dir)
        dialog.connect("closed", self._on_download_dialog_closed)
        dialog.present(self)
        return False

    def _on_error(self, error: str) -> bool:
        """ComfyUI initialization error."""
        self.bottom_bar.set_generate_label("Error")
        logger.error("ComfyUI error: %s", error)
        return False

    # =========================================================================
    # VRAM monitoring
    # =========================================================================

    def _update_vram(self) -> bool:
        """Periodic VRAM usage update (called by GLib.timeout_add)."""
        if self._queue and self._queue.engine._initialized:
            used, total = self._queue.engine.get_vram_usage()
            self.bottom_bar.set_vram(used, total)
        return True

    # =========================================================================
    # Session persistence
    # =========================================================================

    def _save_session(self) -> None:
        """Persist current session state to config."""
        config = get_config()
        ctrl_state = self.control_panel.get_session_state()
        config.session.last_workflow = ctrl_state["last_workflow"]
        config.session.last_prompt = ctrl_state["last_prompt"]
        config.session.last_negative = ctrl_state["last_negative"]
        config.session.last_style = ctrl_state["last_style"]

        # Window geometry
        width, height = self.get_default_size()
        if width > 0 and height > 0:
            config.session.window_width = width
            config.session.window_height = height
        config.session.paned_position = self._paned.get_position()

        try:
            config.save()
            logger.debug("Session saved")
        except Exception as e:
            logger.warning("Failed to save session: %s", e)

    def do_close_request(self) -> bool:
        """Called when the window is about to close -- save session."""
        self._save_session()
        return super().do_close_request()

    # =========================================================================
    # Legacy external API (kept for backward compatibility)
    # =========================================================================

    def set_vram_usage(self, used_gb: float, total_gb: float) -> None:
        """Update VRAM display (external API)."""
        self.bottom_bar.set_vram(int(used_gb * 1e9), int(total_gb * 1e9))
