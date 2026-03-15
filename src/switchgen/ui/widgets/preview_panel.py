"""Right-side preview panel widget."""

from __future__ import annotations

from ...core.logging import get_logger

logger = get_logger(__name__)

try:
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Gdk, GdkPixbuf, Gtk, Pango
except (ImportError, ValueError):
    pass


class PreviewPanel(Gtk.Box):
    """Right-side panel containing the image preview, progress bar, gallery,
    and beginner tips.
    """

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, spacing=8, **kwargs)
        self.set_margin_start(8)
        self.set_margin_end(12)
        self.set_margin_top(12)
        self.set_margin_bottom(12)

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def show_image(self, pixbuf: GdkPixbuf.Pixbuf) -> None:
        """Display a generated image (as a GdkPixbuf)."""
        self._preview_picture.set_paintable(Gdk.Texture.new_for_pixbuf(pixbuf))
        self._preview_stack.set_visible_child_name("preview")
        self.add_to_gallery(pixbuf)

    def set_progress(self, fraction: float, text: str = "") -> None:
        """Update the progress bar.

        Args:
            fraction: Progress between 0.0 and 1.0.
            text: Accessible description for the progress bar.
        """
        self._progress_bar.set_visible(True)
        self._progress_bar.set_fraction(fraction)
        if text:
            self._progress_bar.set_tooltip_text(text)

    def show_placeholder(self) -> None:
        """Show the placeholder instead of a preview image."""
        self._preview_stack.set_visible_child_name("placeholder")

    def add_to_gallery(self, pixbuf: GdkPixbuf.Pixbuf) -> None:
        """Add a thumbnail to the history gallery."""
        size = 80
        w, h = pixbuf.get_width(), pixbuf.get_height()
        scale = size / max(w, h)
        thumb = pixbuf.scale_simple(int(w * scale), int(h * scale), GdkPixbuf.InterpType.BILINEAR)
        pic = Gtk.Picture(paintable=Gdk.Texture.new_for_pixbuf(thumb))
        pic.set_size_request(size, size)
        self._gallery_box.prepend(pic)

    def reset(self) -> None:
        """Hide progress bar and reset state after generation."""
        self._progress_bar.set_visible(False)
        self._progress_bar.set_fraction(0)

    # ------------------------------------------------------------------
    # UI Building
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        # Preview image
        frame = Gtk.Frame(vexpand=True)
        self._preview_picture = Gtk.Picture(content_fit=Gtk.ContentFit.CONTAIN)
        placeholder = Gtk.Label(label="Select workflow and generate", css_classes=["dim-label"])
        self._preview_stack = Gtk.Stack()
        self._preview_stack.add_named(placeholder, "placeholder")
        self._preview_stack.add_named(self._preview_picture, "preview")
        self._preview_stack.set_visible_child_name("placeholder")
        frame.set_child(self._preview_stack)
        self.append(frame)

        # Progress bar (hidden by default)
        self._progress_bar = Gtk.ProgressBar(visible=False)
        self.append(self._progress_bar)

        # Gallery (history thumbnails)
        self.append(Gtk.Label(label="HISTORY", xalign=0, css_classes=["dim-label"]))
        scroll = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            vscrollbar_policy=Gtk.PolicyType.NEVER,
            min_content_height=90,
        )
        self._gallery_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=6)
        scroll.set_child(self._gallery_box)
        self.append(scroll)

        # Tips panel (collapsible)
        tips_expander = Gtk.Expander(label="Tips for Beginners")
        tips_expander.set_expanded(False)
        tips_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        tips_box.set_margin_start(12)
        tips_box.set_margin_end(12)
        tips_box.set_margin_top(8)
        tips_box.set_margin_bottom(8)

        tips = [
            ("Start Simple", "Begin with short, clear prompts. Add details gradually."),
            (
                "Use Presets",
                "Click Quality presets (Fast/Balanced/Quality) to set good defaults.",
            ),
            (
                "Try Templates",
                "Use the Templates dropdown for starter prompts you can customize.",
            ),
            (
                "Add Style",
                "Select a Style to automatically enhance your prompt with artistic terms.",
            ),
            (
                "Iterate",
                "Generate quickly with 'Fast' preset, then increase quality once you like "
                "the result.",
            ),
            (
                "Save Your Seed",
                "The Seed number lets you recreate the exact same image later.",
            ),
        ]

        for title, desc in tips:
            tip_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            tip_title = Gtk.Label(label=f"\u2022 {title}:", xalign=0, css_classes=["heading"])
            tip_title.set_size_request(120, -1)
            tip_row.append(tip_title)
            tip_desc = Gtk.Label(
                label=desc,
                xalign=0,
                wrap=True,
                wrap_mode=Pango.WrapMode.WORD_CHAR,
            )
            tip_desc.set_hexpand(True)
            tip_row.append(tip_desc)
            tips_box.append(tip_row)

        tips_expander.set_child(tips_box)
        self.append(tips_expander)
