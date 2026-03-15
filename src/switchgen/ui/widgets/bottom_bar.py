"""Bottom bar widget with generate button, VRAM display, and info banner."""

from __future__ import annotations

from collections.abc import Callable

from ...core.logging import get_logger

logger = get_logger(__name__)

try:
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Adw, Gtk
except (ImportError, ValueError):
    pass


class BottomBar(Gtk.Box):
    """Bottom bar containing the generate button, VRAM level bar, and info banner.

    Attributes:
        generate_btn: The main generate Gtk.Button (for external signal wiring).
        on_generate_clicked: Callback invoked when the generate button is pressed.
        on_banner_button_clicked: Callback invoked when the banner action is pressed.
    """

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=0, **kwargs)

        # Callbacks
        self.on_generate_clicked: Callable[[], None] | None = None
        self.on_banner_button_clicked: Callable[[], None] | None = None

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def set_vram(self, used_bytes: int, total_bytes: int) -> None:
        """Update the VRAM level bar and text label.

        Args:
            used_bytes: VRAM in use (bytes).
            total_bytes: Total VRAM available (bytes).
        """
        if total_bytes > 0:
            self._vram_bar.set_value(used_bytes / total_bytes)
        else:
            self._vram_bar.set_value(0)
        used_gb = used_bytes / 1e9
        total_gb = total_bytes / 1e9
        self._vram_label.set_text(f"{used_gb:.1f}/{total_gb:.1f} GB")
        self._vram_text_label.set_text(f"{used_gb:.1f}/{total_gb:.1f} GB")

    def set_generating(self, is_generating: bool) -> None:
        """Toggle the generate button between active and idle states."""
        if is_generating:
            self.generate_btn.set_sensitive(False)
            self.generate_btn.set_label("Generating...")
        else:
            self.generate_btn.set_sensitive(True)
            self.generate_btn.set_label("GENERATE")

    def set_generate_label(self, text: str) -> None:
        """Update the generate button label."""
        self.generate_btn.set_label(text)

    def set_generate_sensitive(self, sensitive: bool) -> None:
        """Enable/disable the generate button."""
        self.generate_btn.set_sensitive(sensitive)

    def show_info(self, message: str) -> None:
        """Show the info banner with *message*. Pass empty string to hide."""
        if message:
            self._info_banner.set_title(message)
            self._info_banner.set_revealed(True)
        else:
            self._info_banner.set_revealed(False)

    @property
    def info_banner(self) -> Adw.Banner:
        """Direct access to the Adw.Banner (for legacy wiring)."""
        return self._info_banner

    # ------------------------------------------------------------------
    # UI Building
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Info banner (sits above the button row)
        self._info_banner = Adw.Banner(
            title="No models installed. Click the download button to get started.",
            button_label="Download Models",
            revealed=False,
        )
        self._info_banner.connect("button-clicked", self._on_banner_clicked)
        self.append(self._info_banner)

        # Button row
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=12)
        row.set_margin_start(12)
        row.set_margin_end(12)
        row.set_margin_top(8)
        row.set_margin_bottom(12)

        self.generate_btn = Gtk.Button(label="GENERATE", css_classes=["suggested-action"])
        self.generate_btn.set_size_request(140, -1)
        self.generate_btn.connect("clicked", self._on_generate_clicked)
        row.append(self.generate_btn)

        row.append(Gtk.Box(hexpand=True))  # spacer

        row.append(Gtk.Label(label="VRAM", css_classes=["dim-label"]))
        self._vram_bar = Gtk.LevelBar(min_value=0, max_value=1, value=0)
        self._vram_bar.set_size_request(120, -1)
        row.append(self._vram_bar)

        # Accessible text label next to the level bar
        self._vram_text_label = Gtk.Label(label="", css_classes=["dim-label", "caption"])
        row.append(self._vram_text_label)

        # Compact numeric label (kept for backward compat)
        self._vram_label = Gtk.Label(label="0/0 GB")
        row.append(self._vram_label)

        self.append(row)

    # ------------------------------------------------------------------
    # Handlers
    # ------------------------------------------------------------------

    def _on_generate_clicked(self, _btn) -> None:
        if self.on_generate_clicked:
            self.on_generate_clicked()

    def _on_banner_clicked(self, _banner) -> None:
        if self.on_banner_button_clicked:
            self.on_banner_button_clicked()
