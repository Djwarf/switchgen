"""GTK4 Application for SwitchGen.

This module contains the main application class and launch function.
"""

import sys
from typing import Optional

from .core.logging import get_logger

logger = get_logger(__name__)

# Check for GTK4 availability
try:
    import gi
    gi.require_version('Gtk', '4.0')
    gi.require_version('Adw', '1')
    from gi.repository import Gtk, Adw, GLib, Gio, Gdk
    GTK_AVAILABLE = True
    logger.debug("GTK4 and Libadwaita loaded successfully")
except (ImportError, ValueError) as e:
    GTK_AVAILABLE = False
    GTK_ERROR = str(e)
    logger.error("Failed to load GTK4: %s", GTK_ERROR)


def run_app():
    """Run the GTK4 application."""
    if not GTK_AVAILABLE:
        logger.error("GTK4 not available: %s", GTK_ERROR)
        print(f"Error: GTK4 not available: {GTK_ERROR}")
        print("Install with: sudo pacman -S gtk4 libadwaita python-gobject")
        sys.exit(1)

    logger.info("Starting GTK4 application")
    app = SwitchGenApp()
    return app.run(sys.argv)


class SwitchGenApp(Adw.Application):
    """Main GTK4 Application class."""

    def __init__(self):
        super().__init__(
            application_id="com.switchsides.switchgen",
            flags=Gio.ApplicationFlags.FLAGS_NONE
        )
        self.window: Optional["MainWindow"] = None

    def do_startup(self):
        """Called when the application starts."""
        Adw.Application.do_startup(self)
        self._load_css()

    def do_activate(self):
        """Called when the application is activated."""
        logger.debug("Application activated")
        if not self.window:
            logger.debug("Creating main window")
            from .ui.main_window import MainWindow
            self.window = MainWindow(application=self)
        self.window.present()

    def _load_css(self):
        """Load the SwitchSides CSS theme."""
        css_provider = Gtk.CssProvider()

        # Use inline CSS (avoids unsupported properties in external file)
        css_provider.load_from_string(FALLBACK_CSS)

        # Get the default display
        display = Gdk.Display.get_default()
        if display:
            Gtk.StyleContext.add_provider_for_display(
                display,
                css_provider,
                Gtk.STYLE_PROVIDER_PRIORITY_APPLICATION
            )

    def _find_css_path(self):
        """Find the CSS file path."""
        from pathlib import Path
        import switchgen

        # Try package directory
        pkg_dir = Path(switchgen.__file__).parent
        css_path = pkg_dir / "ui" / "styles" / "switchsides.css"
        if css_path.exists():
            return css_path

        return None


# Fallback CSS if file not found
FALLBACK_CSS = """
/* SwitchSides Brand Theme for GTK4 */

/* Color definitions via CSS variables */
@define-color burgundy #2e0000;
@define-color burgundy_dark #1e0000;
@define-color newsprint #FAFAF8;
@define-color black #000000;

/* Base window */
window {
    background-color: @newsprint;
}

/* Typography */
* {
    font-family: "Crimson Text", "Georgia", serif;
    font-size: 18px;
    color: @black;
}

/* Headers */
.title-1 {
    font-weight: 700;
    font-size: 32px;
    letter-spacing: 2px;
    color: @burgundy;
}

.section-header {
    font-size: 14px;
    font-weight: 600;
    letter-spacing: 1px;
    color: @burgundy;
    border-bottom: 2px solid @burgundy;
    padding-bottom: 4px;
    margin-bottom: 8px;
}

/* Buttons */
button.suggested-action,
button.primary {
    background-color: @burgundy;
    color: @newsprint;
    border: none;
    border-radius: 0;
    padding: 12px 24px;
    font-weight: 600;
}

button.suggested-action:hover,
button.primary:hover {
    background-color: @burgundy_dark;
}

/* Text entries */
entry,
textview {
    background-color: white;
    border: 1px solid @burgundy;
    border-radius: 0;
    padding: 8px;
}

entry:focus,
textview:focus {
    border-color: @burgundy_dark;
    border-width: 2px;
}

/* Progress bar */
progressbar trough {
    background-color: #e0e0e0;
    border-radius: 0;
}

progressbar progress {
    background-color: @burgundy;
    border-radius: 0;
}

/* Level bar (for memory gauge) */
levelbar block.filled {
    background-color: @burgundy;
}

levelbar block.empty {
    background-color: #e0e0e0;
}

/* Cards */
.card {
    background-color: white;
    border-top: 3px solid @burgundy;
    padding: 16px;
    margin: 8px;
}
"""
