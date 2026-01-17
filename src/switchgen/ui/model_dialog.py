"""Model download dialog."""

from pathlib import Path
from typing import Optional

from ..core.logging import get_logger

logger = get_logger(__name__)

try:
    import gi
    gi.require_version('Gtk', '4.0')
    gi.require_version('Adw', '1')
    from gi.repository import Gtk, Adw, GLib, Pango
except (ImportError, ValueError):
    pass

from ..core.models import (
    ModelInfo, ModelType, MODEL_CATALOG, QualityTier,
    get_models_by_type, get_recommended_models, is_model_installed,
)
from ..core.downloader import ModelDownloader, DownloadProgress, DownloadResult


# Section descriptions for beginners
SECTION_DESCRIPTIONS = {
    ModelType.CHECKPOINT: "Main AI models that generate images. You need at least one to get started.",
    ModelType.VAE: "Optional image quality enhancers. Not needed for beginners.",
    ModelType.CLIP_VISION: "Required by some specialized workflows like 3D generation.",
    ModelType.TEXT_ENCODER: "Required by some specialized workflows like audio generation.",
    ModelType.CONTROLNET: "Advanced tools to guide image generation. Learn the basics first.",
    ModelType.UPSCALER: "Make your generated images larger without losing quality.",
}


class ModelDownloadDialog(Adw.Dialog):
    """Dialog for browsing and downloading models."""

    def __init__(self, models_dir: Path, **kwargs):
        super().__init__(**kwargs)

        self.models_dir = models_dir
        self.downloader = ModelDownloader(models_dir)
        self._download_buttons: dict[str, Gtk.Button] = {}
        self._status_labels: dict[str, Gtk.Label] = {}

        self.set_title("Download Models")
        self.set_content_width(500)
        self.set_content_height(600)

        self._build_ui()
        self._refresh_status()

    def _build_ui(self):
        """Build the dialog UI."""
        # Main box
        main_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=0)
        self.set_child(main_box)

        # Header bar
        header = Adw.HeaderBar()
        header.set_show_end_title_buttons(True)
        main_box.append(header)

        # Scrolled content
        scroll = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vexpand=True,
        )
        main_box.append(scroll)

        content_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=16)
        content_box.set_margin_start(16)
        content_box.set_margin_end(16)
        content_box.set_margin_top(16)
        content_box.set_margin_bottom(16)
        scroll.set_child(content_box)

        # Disk space info
        free_mb, total_mb = self.downloader.get_disk_space_mb()
        space_label = Gtk.Label(
            label=f"Disk space: {free_mb/1024:.1f} GB free of {total_mb/1024:.1f} GB",
            xalign=0,
            css_classes=["dim-label"],
        )
        content_box.append(space_label)

        # =====================================================================
        # GETTING STARTED SECTION
        # =====================================================================
        recommended = get_recommended_models()
        if recommended:
            # Check if any recommended models are not installed
            has_uninstalled = any(not is_model_installed(m, self.models_dir) for m in recommended)
            if has_uninstalled:
                self._build_getting_started_section(content_box, recommended)

        # =====================================================================
        # MODEL SECTIONS BY TYPE
        # =====================================================================
        type_order = [
            ModelType.CHECKPOINT,
            ModelType.CLIP_VISION,
            ModelType.TEXT_ENCODER,
            ModelType.CONTROLNET,
            ModelType.VAE,
            ModelType.UPSCALER,
        ]

        for model_type in type_order:
            models = get_models_by_type(model_type)
            if not models:
                continue

            # Section header with description
            type_name = model_type.value.replace("_", " ").upper()
            header_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
            header_box.set_margin_top(12)

            header_label = Gtk.Label(
                label=type_name,
                xalign=0,
                css_classes=["heading"],
            )
            header_box.append(header_label)

            # Add section description for beginners
            if model_type in SECTION_DESCRIPTIONS:
                desc_label = Gtk.Label(
                    label=SECTION_DESCRIPTIONS[model_type],
                    xalign=0,
                    css_classes=["dim-label"],
                    wrap=True,
                    wrap_mode=Pango.WrapMode.WORD_CHAR,
                )
                header_box.append(desc_label)

            content_box.append(header_box)

            # Model list for this type
            list_box = Gtk.ListBox(
                selection_mode=Gtk.SelectionMode.NONE,
                css_classes=["boxed-list"],
            )
            content_box.append(list_box)

            for model in models:
                row = self._create_model_row(model)
                list_box.append(row)

        # Progress section at bottom
        self.progress_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.progress_box.set_margin_top(16)
        self.progress_box.set_visible(False)
        content_box.append(self.progress_box)

        self.progress_label = Gtk.Label(label="", xalign=0)
        self.progress_box.append(self.progress_label)

        self.progress_bar = Gtk.ProgressBar()
        self.progress_box.append(self.progress_bar)

    def _build_getting_started_section(self, content_box: Gtk.Box, recommended: list[ModelInfo]):
        """Build the Getting Started section for new users."""
        # Frame with special styling
        frame = Gtk.Frame()
        frame.add_css_class("view")
        frame_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        frame_box.set_margin_start(16)
        frame_box.set_margin_end(16)
        frame_box.set_margin_top(16)
        frame_box.set_margin_bottom(16)
        frame.set_child(frame_box)

        # Title
        title = Gtk.Label(
            label="Getting Started",
            xalign=0,
            css_classes=["title-2"],
        )
        frame_box.append(title)

        # Welcome text
        welcome = Gtk.Label(
            label="Welcome! To generate images, you need to download at least one model. "
                  "We recommend starting with Stable Diffusion 1.5 - it's fast, works on most computers, "
                  "and produces great results.",
            xalign=0,
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD_CHAR,
        )
        frame_box.append(welcome)

        # Recommended models list
        list_box = Gtk.ListBox(
            selection_mode=Gtk.SelectionMode.NONE,
            css_classes=["boxed-list"],
        )
        frame_box.append(list_box)

        for model in recommended:
            if not is_model_installed(model, self.models_dir):
                row = self._create_model_row(model, highlight=True)
                list_box.append(row)

        content_box.append(frame)

    def _create_model_row(self, model: ModelInfo, highlight: bool = False) -> Gtk.ListBoxRow:
        """Create a row for a model.

        Args:
            model: The model info
            highlight: If True, show with emphasis (for recommended section)
        """
        row = Gtk.ListBoxRow()

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(10)
        box.set_margin_bottom(10)
        row.set_child(box)

        # Info column
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        info_box.set_hexpand(True)
        box.append(info_box)

        # Name row with badges
        name_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        info_box.append(name_row)

        name_label = Gtk.Label(label=model.name, xalign=0)
        name_label.set_css_classes(["heading"])
        name_row.append(name_label)

        # Add badges
        if model.recommended:
            badge = Gtk.Label(label="Recommended")
            badge.add_css_class("success")
            badge.add_css_class("caption")
            name_row.append(badge)

        if model.quality_tier == QualityTier.HIGH:
            badge = Gtk.Label(label="High Quality")
            badge.add_css_class("accent")
            badge.add_css_class("caption")
            name_row.append(badge)
        elif model.quality_tier == QualityTier.STARTER:
            badge = Gtk.Label(label="Beginner Friendly")
            badge.add_css_class("caption")
            name_row.append(badge)

        # Description
        desc_label = Gtk.Label(
            label=model.description,
            xalign=0,
            css_classes=["dim-label"],
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD_CHAR,
        )
        info_box.append(desc_label)

        # Tips (if available)
        if model.tips:
            tips_label = Gtk.Label(
                label=model.tips,
                xalign=0,
                css_classes=["caption"],
                wrap=True,
                wrap_mode=Pango.WrapMode.WORD_CHAR,
            )
            tips_label.set_margin_top(4)
            info_box.append(tips_label)

        # Right side: size, VRAM, status, button
        right_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        right_box.set_valign(Gtk.Align.CENTER)
        box.append(right_box)

        # Size and VRAM info
        specs_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        specs_box.set_halign(Gtk.Align.END)
        right_box.append(specs_box)

        size_label = Gtk.Label(
            label=f"{model.size_mb / 1024:.1f} GB" if model.size_mb >= 1024 else f"{model.size_mb} MB",
            css_classes=["dim-label", "caption"],
        )
        size_label.set_tooltip_text("Download size")
        specs_box.append(size_label)

        vram_label = Gtk.Label(
            label=f"{model.vram_gb:.0f}GB VRAM",
            css_classes=["dim-label", "caption"],
        )
        vram_label.set_tooltip_text("Minimum GPU memory required")
        specs_box.append(vram_label)

        # Status and button row
        button_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        button_box.set_halign(Gtk.Align.END)
        right_box.append(button_box)

        # Status label
        status_label = Gtk.Label(label="")
        status_label.set_width_chars(10)
        button_box.append(status_label)
        self._status_labels[model.id] = status_label

        # Download button
        button = Gtk.Button(label="Download")
        button.connect("clicked", self._on_download_clicked, model)
        if highlight:
            button.add_css_class("suggested-action")
        button_box.append(button)
        self._download_buttons[model.id] = button

        return row

    def _refresh_status(self):
        """Refresh installed status for all models."""
        for model_id, model in MODEL_CATALOG.items():
            if model_id not in self._status_labels:
                continue

            status_label = self._status_labels[model_id]
            button = self._download_buttons[model_id]

            if is_model_installed(model, self.models_dir):
                status_label.set_label("Installed")
                status_label.set_css_classes(["success"])
                button.set_sensitive(False)
                button.set_label("Installed")
            else:
                status_label.set_label("")
                button.set_sensitive(True)
                button.set_label("Download")

    def _on_download_clicked(self, button: Gtk.Button, model: ModelInfo):
        """Handle download button click."""
        if self.downloader.is_downloading:
            return

        # Check disk space
        if not self.downloader.check_disk_space(model.size_mb):
            free_mb, _ = self.downloader.get_disk_space_mb()
            logger.warning(
                "Insufficient disk space for %s (need=%dMB, free=%.0fMB)",
                model.name, model.size_mb, free_mb
            )
            self._show_error(f"Not enough disk space.\nNeed {model.size_mb} MB, have {free_mb:.0f} MB free.")
            return

        logger.info("Download initiated (model=%s, size=%dMB)", model.name, model.size_mb)

        # Disable all download buttons
        for btn in self._download_buttons.values():
            btn.set_sensitive(False)

        # Show progress
        self.progress_box.set_visible(True)
        self.progress_label.set_label(f"Downloading {model.name}...")
        self.progress_bar.set_fraction(0)

        # Start async download
        self.downloader.download_async(
            model,
            progress_callback=lambda p: GLib.idle_add(self._on_progress, p),
            complete_callback=lambda r: GLib.idle_add(self._on_complete, r),
        )

    def _on_progress(self, progress: DownloadProgress):
        """Handle download progress update."""
        self.progress_bar.set_fraction(progress.progress)
        self.progress_label.set_label(
            f"Downloading... {progress.downloaded_mb:.1f} / {progress.total_mb:.1f} MB"
        )
        return False

    def _on_complete(self, result: DownloadResult):
        """Handle download completion."""
        self.progress_box.set_visible(False)

        if result.success:
            logger.info("Download completed successfully (model=%s, path=%s)", result.model_id, result.path)
            self.progress_label.set_label(f"Downloaded successfully!")
        else:
            logger.error("Download failed (model=%s): %s", result.model_id, result.error)
            self._show_error(f"Download failed: {result.error}")

        # Refresh status and re-enable buttons
        self._refresh_status()

        # Re-enable uninstalled model buttons
        for model_id, btn in self._download_buttons.items():
            model = MODEL_CATALOG.get(model_id)
            if model and not is_model_installed(model, self.models_dir):
                btn.set_sensitive(True)

        return False

    def _show_error(self, message: str):
        """Show an error message."""
        logger.error("UI error shown: %s", message)
        dialog = Adw.AlertDialog(
            heading="Error",
            body=message,
        )
        dialog.add_response("ok", "OK")
        dialog.present(self)
