"""Model download dialog."""

from pathlib import Path
from typing import Optional

try:
    import gi
    gi.require_version('Gtk', '4.0')
    gi.require_version('Adw', '1')
    from gi.repository import Gtk, Adw, GLib, Pango
except (ImportError, ValueError):
    pass

from ..core.models import (
    ModelInfo, ModelType, MODEL_CATALOG,
    get_models_by_type, is_model_installed,
)
from ..core.downloader import ModelDownloader, DownloadProgress, DownloadResult


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

        # Group models by type
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

            # Section header
            type_name = model_type.value.replace("_", " ").upper()
            header_label = Gtk.Label(
                label=type_name,
                xalign=0,
                css_classes=["heading"],
            )
            header_label.set_margin_top(8)
            content_box.append(header_label)

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

    def _create_model_row(self, model: ModelInfo) -> Gtk.ListBoxRow:
        """Create a row for a model."""
        row = Gtk.ListBoxRow()

        box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(8)
        box.set_margin_bottom(8)
        row.set_child(box)

        # Info column
        info_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=2)
        info_box.set_hexpand(True)
        box.append(info_box)

        name_label = Gtk.Label(label=model.name, xalign=0)
        name_label.set_css_classes(["heading"])
        info_box.append(name_label)

        desc_label = Gtk.Label(
            label=model.description,
            xalign=0,
            css_classes=["dim-label"],
            ellipsize=Pango.EllipsizeMode.END,
        )
        info_box.append(desc_label)

        # Size
        size_label = Gtk.Label(
            label=f"{model.size_mb} MB",
            css_classes=["dim-label"],
        )
        size_label.set_margin_end(8)
        box.append(size_label)

        # Status label
        status_label = Gtk.Label(label="")
        status_label.set_width_chars(10)
        box.append(status_label)
        self._status_labels[model.id] = status_label

        # Download button
        button = Gtk.Button(label="Download")
        button.connect("clicked", self._on_download_clicked, model)
        box.append(button)
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
            self._show_error(f"Not enough disk space.\nNeed {model.size_mb} MB, have {free_mb:.0f} MB free.")
            return

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
            self.progress_label.set_label(f"Downloaded successfully!")
        else:
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
        dialog = Adw.AlertDialog(
            heading="Error",
            body=message,
        )
        dialog.add_response("ok", "OK")
        dialog.present(self)
