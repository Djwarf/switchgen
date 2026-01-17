"""Main application window."""

import os
import threading
from typing import Optional

try:
    import gi
    gi.require_version('Gtk', '4.0')
    gi.require_version('Adw', '1')
    from gi.repository import Gtk, Adw, GLib, Gdk, Pango, GdkPixbuf, Gio
except (ImportError, ValueError):
    pass

from ..core import (
    get_available_checkpoints,
    build_text2img_memory_workflow,
    build_img2img_memory_workflow,
    build_inpaint_workflow,
    build_audio_workflow,
    build_3d_zero123_workflow,
    tensor_to_pil,
    ProgressInfo,
    WorkflowType,
    WORKFLOW_SPECS,
    get_models_for_workflow,
)
from ..core.queue import get_queue, GenerationJob
from ..core.config import get_config
from .model_dialog import ModelDownloadDialog


class MainWindow(Adw.ApplicationWindow):
    """Main SwitchGen window."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.set_title("SwitchGen")
        self.set_default_size(1200, 800)

        # State
        self._generating = False
        self._all_checkpoints: list[str] = []
        self._filtered_checkpoints: list[str] = []
        self._queue = None
        self._current_seed: Optional[int] = None
        self._current_workflow = WorkflowType.TEXT2IMG
        self._input_image_path: Optional[str] = None
        self._mask_image_path: Optional[str] = None

        # Build UI
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL)
        self.set_content(main_box)

        header = Adw.HeaderBar()
        header.set_title_widget(Gtk.Label(label="SWITCHGEN", css_classes=["title-1"]))

        # Download models button
        download_btn = Gtk.Button(icon_name="folder-download-symbolic")
        download_btn.set_tooltip_text("Download Models")
        download_btn.connect("clicked", self._on_download_models_clicked)
        header.pack_end(download_btn)

        main_box.append(header)

        paned = Gtk.Paned(orientation=Gtk.Orientation.HORIZONTAL)
        paned.set_shrink_start_child(False)
        paned.set_shrink_end_child(False)
        paned.set_position(320)
        main_box.append(paned)

        paned.set_start_child(self._build_controls())
        paned.set_end_child(self._build_preview())

        main_box.append(self._build_bottom_bar())

        # Initialize
        self.generate_btn.set_sensitive(False)
        self.generate_btn.set_label("Initializing...")
        threading.Thread(target=self._init_comfy, daemon=True).start()
        GLib.timeout_add(1000, self._update_vram)

    # =========================================================================
    # UI Building
    # =========================================================================

    def _build_controls(self) -> Gtk.Widget:
        """Build the left control panel."""
        scroll = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            hexpand=False, vexpand=True
        )
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        scroll.set_child(box)

        # WORKFLOW selector (first - drives everything else)
        box.append(self._label("WORKFLOW"))
        workflow_names = [WORKFLOW_SPECS[wt].name for wt in WorkflowType]
        self.workflow_dropdown = Gtk.DropDown.new_from_strings(workflow_names)
        self.workflow_dropdown.connect("notify::selected", self._on_workflow_changed)
        box.append(self.workflow_dropdown)

        # MODEL selector (filtered by workflow)
        box.append(self._label("MODEL"))
        self.model_dropdown = Gtk.DropDown.new_from_strings(["Loading..."])
        box.append(self.model_dropdown)

        # INPUT IMAGE (for img2img, inpaint, 3d)
        self.input_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.input_box.append(self._label("INPUT IMAGE"))
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.input_label = Gtk.Label(label="None", xalign=0, hexpand=True, ellipsize=Pango.EllipsizeMode.MIDDLE)
        row.append(self.input_label)
        btn = Gtk.Button(label="Browse")
        btn.connect("clicked", self._pick_input_image)
        row.append(btn)
        self.input_box.append(row)
        self.input_box.set_visible(False)
        box.append(self.input_box)

        # MASK IMAGE (for inpaint only)
        self.mask_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.mask_box.append(self._label("MASK"))
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self.mask_label = Gtk.Label(label="None", xalign=0, hexpand=True, ellipsize=Pango.EllipsizeMode.MIDDLE)
        row.append(self.mask_label)
        btn = Gtk.Button(label="Browse")
        btn.connect("clicked", self._pick_mask_image)
        row.append(btn)
        self.mask_box.append(row)
        self.mask_box.set_visible(False)
        box.append(self.mask_box)

        # PROMPT (hidden for 3D)
        self.prompt_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.prompt_box.append(self._label("PROMPT"))
        self.prompt_view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.prompt_view.set_size_request(-1, 80)
        frame = Gtk.Frame()
        frame.set_child(self.prompt_view)
        self.prompt_box.append(frame)
        box.append(self.prompt_box)

        # NEGATIVE (hidden for 3D)
        self.neg_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self.neg_box.append(self._label("NEGATIVE"))
        self.neg_view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.neg_view.set_size_request(-1, 50)
        frame = Gtk.Frame()
        frame.set_child(self.neg_view)
        self.neg_box.append(frame)
        box.append(self.neg_box)

        # PARAMETERS grid
        box.append(self._label("PARAMETERS"))
        grid = Gtk.Grid(column_spacing=8, row_spacing=6)
        row_idx = 0

        # Size (text2img only)
        self.size_label = Gtk.Label(label="Size", xalign=0)
        grid.attach(self.size_label, 0, row_idx, 1, 1)
        size_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self.width_spin = Gtk.SpinButton.new_with_range(256, 2048, 64)
        self.width_spin.set_value(512)
        size_box.append(self.width_spin)
        size_box.append(Gtk.Label(label="×"))
        self.height_spin = Gtk.SpinButton.new_with_range(256, 2048, 64)
        self.height_spin.set_value(512)
        size_box.append(self.height_spin)
        self.size_box = size_box
        grid.attach(size_box, 1, row_idx, 1, 1)
        row_idx += 1

        # Steps
        grid.attach(Gtk.Label(label="Steps", xalign=0), 0, row_idx, 1, 1)
        self.steps_spin = Gtk.SpinButton.new_with_range(1, 150, 1)
        self.steps_spin.set_value(20)
        grid.attach(self.steps_spin, 1, row_idx, 1, 1)
        row_idx += 1

        # CFG
        self.cfg_label = Gtk.Label(label="CFG", xalign=0)
        grid.attach(self.cfg_label, 0, row_idx, 1, 1)
        self.cfg_spin = Gtk.SpinButton.new_with_range(1.0, 30.0, 0.5)
        self.cfg_spin.set_value(7.0)
        self.cfg_spin.set_digits(1)
        grid.attach(self.cfg_spin, 1, row_idx, 1, 1)
        row_idx += 1

        # Denoise (img2img, inpaint)
        self.denoise_label = Gtk.Label(label="Denoise", xalign=0)
        grid.attach(self.denoise_label, 0, row_idx, 1, 1)
        self.denoise_spin = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.05)
        self.denoise_spin.set_value(0.75)
        self.denoise_spin.set_digits(2)
        grid.attach(self.denoise_spin, 1, row_idx, 1, 1)
        self.denoise_label.set_visible(False)
        self.denoise_spin.set_visible(False)
        row_idx += 1

        # Duration (audio)
        self.duration_label = Gtk.Label(label="Duration (s)", xalign=0)
        grid.attach(self.duration_label, 0, row_idx, 1, 1)
        self.duration_spin = Gtk.SpinButton.new_with_range(1.0, 60.0, 1.0)
        self.duration_spin.set_value(30.0)
        self.duration_spin.set_digits(0)
        grid.attach(self.duration_spin, 1, row_idx, 1, 1)
        self.duration_label.set_visible(False)
        self.duration_spin.set_visible(False)
        row_idx += 1

        # Elevation (3D)
        self.elev_label = Gtk.Label(label="Elevation", xalign=0)
        grid.attach(self.elev_label, 0, row_idx, 1, 1)
        self.elev_spin = Gtk.SpinButton.new_with_range(-90, 90, 5)
        self.elev_spin.set_value(0)
        grid.attach(self.elev_spin, 1, row_idx, 1, 1)
        self.elev_label.set_visible(False)
        self.elev_spin.set_visible(False)
        row_idx += 1

        # Azimuth (3D)
        self.azim_label = Gtk.Label(label="Azimuth", xalign=0)
        grid.attach(self.azim_label, 0, row_idx, 1, 1)
        self.azim_spin = Gtk.SpinButton.new_with_range(-180, 180, 5)
        self.azim_spin.set_value(0)
        grid.attach(self.azim_spin, 1, row_idx, 1, 1)
        self.azim_label.set_visible(False)
        self.azim_spin.set_visible(False)
        row_idx += 1

        # Seed
        grid.attach(Gtk.Label(label="Seed", xalign=0), 0, row_idx, 1, 1)
        self.seed_entry = Gtk.Entry(placeholder_text="Random")
        grid.attach(self.seed_entry, 1, row_idx, 1, 1)

        box.append(grid)
        return scroll

    def _build_preview(self) -> Gtk.Widget:
        """Build the right preview panel."""
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        box.set_margin_start(8)
        box.set_margin_end(12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)

        # Preview
        frame = Gtk.Frame(vexpand=True)
        self.preview_picture = Gtk.Picture(content_fit=Gtk.ContentFit.CONTAIN)
        placeholder = Gtk.Label(label="Select workflow and generate", css_classes=["dim-label"])
        self.preview_stack = Gtk.Stack()
        self.preview_stack.add_named(placeholder, "placeholder")
        self.preview_stack.add_named(self.preview_picture, "preview")
        self.preview_stack.set_visible_child_name("placeholder")
        frame.set_child(self.preview_stack)
        box.append(frame)

        # Progress
        self.progress_bar = Gtk.ProgressBar(visible=False)
        box.append(self.progress_bar)

        # Gallery
        box.append(Gtk.Label(label="HISTORY", xalign=0, css_classes=["dim-label"]))
        scroll = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.NEVER,
            min_content_height=90
        )
        self.gallery_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        scroll.set_child(self.gallery_box)
        box.append(scroll)

        return box

    def _build_bottom_bar(self) -> Gtk.Widget:
        """Build the bottom bar."""
        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(8)
        box.set_margin_bottom(12)

        self.generate_btn = Gtk.Button(label="GENERATE", css_classes=["suggested-action"])
        self.generate_btn.set_size_request(140, -1)
        self.generate_btn.connect("clicked", self._on_generate)
        box.append(self.generate_btn)

        box.append(Gtk.Box(hexpand=True))  # spacer

        box.append(Gtk.Label(label="VRAM", css_classes=["dim-label"]))
        self.vram_bar = Gtk.LevelBar(min_value=0, max_value=1, value=0)
        self.vram_bar.set_size_request(120, -1)
        box.append(self.vram_bar)
        self.vram_label = Gtk.Label(label="0/0 GB")
        box.append(self.vram_label)

        return box

    def _label(self, text: str) -> Gtk.Label:
        """Create a section label."""
        return Gtk.Label(label=text, xalign=0, css_classes=["dim-label"])

    # =========================================================================
    # Workflow Changes
    # =========================================================================

    def _on_workflow_changed(self, dropdown, _param) -> None:
        """Handle workflow selection change."""
        idx = dropdown.get_selected()
        workflows = list(WorkflowType)
        if idx >= len(workflows):
            return

        self._current_workflow = workflows[idx]
        spec = WORKFLOW_SPECS[self._current_workflow]

        # Filter models for this workflow
        self._filtered_checkpoints = get_models_for_workflow(
            self._all_checkpoints, self._current_workflow
        )
        if self._filtered_checkpoints:
            self.model_dropdown.set_model(Gtk.StringList.new(self._filtered_checkpoints))
        else:
            self.model_dropdown.set_model(Gtk.StringList.new(["No compatible models"]))

        # Show/hide sections based on workflow needs
        self.input_box.set_visible(spec.needs_input_image)
        self.mask_box.set_visible(spec.needs_mask)
        self.prompt_box.set_visible(spec.needs_prompt)
        self.neg_box.set_visible(spec.needs_prompt)

        # Show/hide parameters
        self.size_label.set_visible(spec.needs_size)
        self.size_box.set_visible(spec.needs_size)

        is_img2img = self._current_workflow in (WorkflowType.IMG2IMG, WorkflowType.INPAINT)
        self.denoise_label.set_visible(is_img2img)
        self.denoise_spin.set_visible(is_img2img)

        is_audio = self._current_workflow == WorkflowType.AUDIO
        self.duration_label.set_visible(is_audio)
        self.duration_spin.set_visible(is_audio)

        is_3d = self._current_workflow == WorkflowType.THREE_D
        self.elev_label.set_visible(is_3d)
        self.elev_spin.set_visible(is_3d)
        self.azim_label.set_visible(is_3d)
        self.azim_spin.set_visible(is_3d)

        # Set defaults from spec
        self.steps_spin.set_value(spec.default_steps)
        self.cfg_spin.set_value(spec.default_cfg)
        self.denoise_spin.set_value(spec.default_denoise)

    # =========================================================================
    # File Pickers
    # =========================================================================

    def _pick_input_image(self, btn) -> None:
        """Pick input image."""
        dialog = Gtk.FileDialog(title="Select Input Image")
        f = Gtk.FileFilter()
        f.set_name("Images")
        f.add_mime_type("image/*")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f)
        dialog.set_filters(filters)
        dialog.open(self, None, self._on_input_picked)

    def _on_input_picked(self, dialog, result) -> None:
        try:
            file = dialog.open_finish(result)
            if file:
                self._input_image_path = file.get_path()
                self.input_label.set_text(os.path.basename(self._input_image_path))
        except GLib.Error:
            pass

    def _pick_mask_image(self, btn) -> None:
        """Pick mask image."""
        dialog = Gtk.FileDialog(title="Select Mask Image")
        f = Gtk.FileFilter()
        f.set_name("Images")
        f.add_mime_type("image/*")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f)
        dialog.set_filters(filters)
        dialog.open(self, None, self._on_mask_picked)

    def _on_mask_picked(self, dialog, result) -> None:
        try:
            file = dialog.open_finish(result)
            if file:
                self._mask_image_path = file.get_path()
                self.mask_label.set_text(os.path.basename(self._mask_image_path))
        except GLib.Error:
            pass

    # =========================================================================
    # Generation
    # =========================================================================

    def _on_generate(self, btn) -> None:
        """Handle generate button click."""
        if self._generating or not self._queue:
            return

        # Get model
        idx = self.model_dropdown.get_selected()
        if idx >= len(self._filtered_checkpoints):
            return
        checkpoint = self._filtered_checkpoints[idx]

        # Get prompts
        buf = self.prompt_view.get_buffer()
        prompt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        if not prompt.strip():
            prompt = "high quality, detailed"

        buf = self.neg_view.get_buffer()
        negative = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)

        # Get parameters
        steps = int(self.steps_spin.get_value())
        cfg = self.cfg_spin.get_value()
        seed_txt = self.seed_entry.get_text().strip()
        seed = int(seed_txt) if seed_txt.isdigit() else -1

        # Validate requirements
        spec = WORKFLOW_SPECS[self._current_workflow]
        if spec.needs_input_image and not self._input_image_path:
            return
        if spec.needs_mask and not self._mask_image_path:
            return

        # Start generation
        self._generating = True
        self.generate_btn.set_sensitive(False)
        self.generate_btn.set_label("Generating...")
        self.progress_bar.set_visible(True)
        self.progress_bar.set_fraction(0)

        # Build workflow
        workflow = None
        actual_seed = seed

        if self._current_workflow == WorkflowType.TEXT2IMG:
            workflow, actual_seed = build_text2img_memory_workflow(
                checkpoint=checkpoint, prompt=prompt, negative_prompt=negative,
                width=int(self.width_spin.get_value()),
                height=int(self.height_spin.get_value()),
                steps=steps, cfg=cfg, seed=seed, capture_id="main",
            )
        elif self._current_workflow == WorkflowType.IMG2IMG:
            workflow, actual_seed = build_img2img_memory_workflow(
                checkpoint=checkpoint, image_path=self._input_image_path,
                prompt=prompt, negative_prompt=negative,
                denoise=self.denoise_spin.get_value(),
                steps=steps, cfg=cfg, seed=seed, capture_id="main",
            )
        elif self._current_workflow == WorkflowType.INPAINT:
            workflow, actual_seed = build_inpaint_workflow(
                checkpoint=checkpoint, image_path=self._input_image_path,
                mask_path=self._mask_image_path, prompt=prompt, negative_prompt=negative,
                denoise=self.denoise_spin.get_value(),
                steps=steps, cfg=cfg, seed=seed, capture_id="main",
            )
        elif self._current_workflow == WorkflowType.AUDIO:
            workflow, actual_seed = build_audio_workflow(
                checkpoint=checkpoint, prompt=prompt, negative_prompt=negative,
                seconds=self.duration_spin.get_value(),
                steps=steps, cfg=cfg, seed=seed,
            )
        elif self._current_workflow == WorkflowType.THREE_D:
            workflow, actual_seed = build_3d_zero123_workflow(
                checkpoint=checkpoint, image_path=self._input_image_path,
                elevation=self.elev_spin.get_value(),
                azimuth=self.azim_spin.get_value(),
                steps=steps, cfg=cfg, seed=seed, capture_id="main",
            )

        if not workflow:
            self._reset_ui()
            return

        self._current_seed = actual_seed

        def on_progress(info: ProgressInfo):
            GLib.idle_add(self._update_progress, info)

        def on_complete(job: GenerationJob):
            GLib.idle_add(self._on_complete, job)

        self._queue.submit(workflow=workflow, on_progress=on_progress, on_complete=on_complete)

    def _update_progress(self, info: ProgressInfo) -> bool:
        if info.total_steps > 0:
            self.progress_bar.set_fraction(info.current_step / info.total_steps)
        return False

    def _on_complete(self, job: GenerationJob) -> bool:
        if job.result and job.result.success and job.result.images is not None:
            images = tensor_to_pil(job.result.images)
            if images:
                self._show_image(images[0])
        self._reset_ui()
        return False

    def _show_image(self, pil_image) -> None:
        """Display generated image."""
        import io
        buf = io.BytesIO()
        pil_image.save(buf, format='PNG')
        buf.seek(0)
        loader = GdkPixbuf.PixbufLoader.new_with_type('png')
        loader.write(buf.read())
        loader.close()
        pixbuf = loader.get_pixbuf()

        self.preview_picture.set_paintable(Gdk.Texture.new_for_pixbuf(pixbuf))
        self.preview_stack.set_visible_child_name("preview")
        self._add_to_gallery(pixbuf)

        if self._current_seed is not None:
            self.seed_entry.set_text(str(self._current_seed))

    def _add_to_gallery(self, pixbuf) -> None:
        """Add thumbnail to gallery."""
        size = 80
        w, h = pixbuf.get_width(), pixbuf.get_height()
        scale = size / max(w, h)
        thumb = pixbuf.scale_simple(int(w * scale), int(h * scale), GdkPixbuf.InterpType.BILINEAR)
        pic = Gtk.Picture(paintable=Gdk.Texture.new_for_pixbuf(thumb))
        pic.set_size_request(size, size)
        self.gallery_box.prepend(pic)

    def _reset_ui(self) -> None:
        """Reset UI after generation."""
        self._generating = False
        self.generate_btn.set_sensitive(True)
        self.generate_btn.set_label("GENERATE")
        self.progress_bar.set_visible(False)

    # =========================================================================
    # Model Download
    # =========================================================================

    def _on_download_models_clicked(self, button) -> None:
        """Open the model download dialog."""
        config = get_config()
        dialog = ModelDownloadDialog(config.paths.models_dir)
        dialog.present(self)

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_comfy(self) -> None:
        """Initialize ComfyUI in background."""
        try:
            self._queue = get_queue()
            self._all_checkpoints = get_available_checkpoints()
            GLib.idle_add(self._on_ready)
        except Exception as e:
            GLib.idle_add(self._on_error, str(e))

    def _on_ready(self) -> bool:
        """ComfyUI ready."""
        # Trigger workflow change to filter models
        self._on_workflow_changed(self.workflow_dropdown, None)
        self.generate_btn.set_sensitive(True)
        self.generate_btn.set_label("GENERATE")
        self._update_vram()
        return False

    def _on_error(self, error: str) -> bool:
        """ComfyUI error."""
        self.generate_btn.set_label("Error")
        print(f"ComfyUI error: {error}")
        return False

    def _update_vram(self) -> bool:
        """Update VRAM display."""
        if self._queue and self._queue.engine._initialized:
            used, total = self._queue.engine.get_vram_usage()
            self.vram_bar.set_value(used / total if total > 0 else 0)
            self.vram_label.set_text(f"{used / 1e9:.1f}/{total / 1e9:.1f} GB")
        return True

    def set_vram_usage(self, used_gb: float, total_gb: float):
        """Update VRAM display (external API)."""
        if total_gb > 0:
            self.vram_bar.set_value(used_gb / total_gb)
        self.vram_label.set_text(f"{used_gb:.1f}/{total_gb:.1f} GB")
