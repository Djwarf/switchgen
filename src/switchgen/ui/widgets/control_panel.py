"""Left sidebar control panel widget."""

from __future__ import annotations

import os
from collections.abc import Callable
from typing import Any

from ...core.logging import get_logger
from ...core.models import MODEL_CATALOG, ModelType

logger = get_logger(__name__)

try:
    import gi

    gi.require_version("Gtk", "4.0")
    gi.require_version("Adw", "1")
    from gi.repository import Gio, GLib, Gtk, Pango
except (ImportError, ValueError):
    pass

from ...core import WORKFLOW_SPECS, WorkflowType, get_models_for_workflow

# =============================================================================
# Presets and Templates for Beginners
# =============================================================================

# Size presets (width, height, label, tooltip)
SIZE_PRESETS = {
    "square": (512, 512, "Square", "1:1 ratio - good for portraits and icons"),
    "portrait": (512, 768, "Portrait", "2:3 ratio - good for people and characters"),
    "landscape": (768, 512, "Landscape", "3:2 ratio - good for scenes and environments"),
    "wide": (896, 512, "Wide", "16:9 ratio - good for wallpapers and banners"),
}

SIZE_PRESETS_XL = {
    "square": (1024, 1024, "Square", "1:1 ratio - SDXL native resolution"),
    "portrait": (832, 1216, "Portrait", "2:3 ratio - good for people and characters"),
    "landscape": (1216, 832, "Landscape", "3:2 ratio - good for scenes and environments"),
    "wide": (1344, 768, "Wide", "16:9 ratio - good for wallpapers and banners"),
}

# Quality presets (steps, cfg, label, tooltip)
QUALITY_PRESETS = {
    "fast": (12, 5.0, "Fast", "Quick preview - lower quality but very fast"),
    "balanced": (20, 7.0, "Balanced", "Good balance of speed and quality (recommended)"),
    "quality": (35, 7.5, "Quality", "Higher quality - slower but more detailed"),
}

# Style presets (style suffix to add to prompt)
STYLE_PRESETS = {
    "none": ("", "None", "No style modification"),
    "photo": (
        ", professional photograph, photorealistic, 8k, detailed",
        "Photorealistic",
        "Realistic photograph style",
    ),
    "anime": (
        ", anime style, anime art, vibrant colors, detailed",
        "Anime",
        "Japanese anime style",
    ),
    "oil": (
        ", oil painting, painterly, classical art, brushstrokes",
        "Oil Painting",
        "Classical oil painting style",
    ),
    "digital": (
        ", digital art, concept art, artstation, detailed illustration",
        "Digital Art",
        "Modern digital illustration",
    ),
    "watercolor": (
        ", watercolor painting, soft colors, artistic, delicate",
        "Watercolor",
        "Soft watercolor painting style",
    ),
    "3d": (
        ", 3d render, octane render, unreal engine, realistic lighting",
        "3D Render",
        "3D rendered CGI style",
    ),
}

# Prompt templates by category
PROMPT_TEMPLATES = {
    "Portrait": "a portrait of a [person/character], looking at camera, soft lighting, detailed face",
    "Landscape": "a beautiful landscape of [location], golden hour lighting, scenic view, detailed",
    "Fantasy": "a fantasy scene with [subject], magical atmosphere, epic lighting, detailed",
    "Animal": "a [animal] in its natural habitat, wildlife photography, detailed fur/feathers",
    "Architecture": "an architectural photo of [building type], professional photography, detailed",
    "Food": "a delicious plate of [food], food photography, appetizing, professional lighting",
    "Sci-Fi": "a futuristic [subject], science fiction, neon lights, cyberpunk atmosphere",
    "Nature": "a close-up of [natural subject], macro photography, detailed textures, beautiful",
}

# Common negative prompt for beginners
DEFAULT_NEGATIVE = (
    "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text, signature"
)

# Placeholder text shown in the prompt field
_PROMPT_PLACEHOLDER = "Describe what you want to see..."


def _is_xl_checkpoint(model_name: str) -> bool:
    """Check whether a checkpoint name corresponds to an SDXL model.

    Checks the MODEL_CATALOG metadata first (vram_gb >= 8.0 for the matching
    checkpoint entry), then falls back to simple string matching.
    """
    name_lower = model_name.lower()
    # Check catalog entries with type CHECKPOINT
    for model_info in MODEL_CATALOG.values():
        if model_info.type != ModelType.CHECKPOINT:
            continue
        # Match by filename or local_filename
        cat_fname = model_info.get_local_filename().lower()
        cat_orig = model_info.filename.lower()
        if name_lower in (cat_fname, cat_orig):
            return model_info.vram_gb >= 8.0
    # Fallback: string matching
    return "xl" in name_lower or "sdxl" in name_lower


def _section_label(text: str) -> Gtk.Label:
    """Create a section label."""
    return Gtk.Label(label=text, xalign=0, css_classes=["dim-label"])


class ControlPanel(Gtk.Box):
    """Left sidebar containing all generation controls.

    Attributes:
        on_generate_requested: Called when the user wants to generate.
        on_workflow_changed: Called when the workflow selection changes.
    """

    def __init__(self, **kwargs):
        super().__init__(orientation=Gtk.Orientation.VERTICAL, **kwargs)

        # Callbacks (set by MainWindow)
        self.on_generate_requested: Callable[[], None] | None = None
        self.on_workflow_changed: Callable[[], None] | None = None

        # Internal state
        self._current_workflow = WorkflowType.TEXT2IMG
        self._filtered_checkpoints: list[str] = []
        self._all_checkpoints: list[str] = []
        self._is_xl_model: bool = False
        self._input_image_path: str | None = None
        self._mask_image_path: str | None = None
        self._current_style: str = "none"

        self._build_ui()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def get_generation_params(self) -> dict[str, Any]:
        """Return all current control values as a dict.

        Keys: workflow, checkpoint, prompt, negative, style_suffix, width,
        height, steps, cfg, denoise, seed, input_image_path, mask_image_path,
        duration, elevation, azimuth.
        """
        # Prompt
        buf = self._prompt_view.get_buffer()
        prompt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        if prompt == _PROMPT_PLACEHOLDER:
            prompt = ""
        if not prompt.strip():
            prompt = "high quality, detailed"

        # Apply style suffix
        style_suffix = ""
        if self._current_style and self._current_style in STYLE_PRESETS:
            style_suffix = STYLE_PRESETS[self._current_style][0]

        # Negative
        buf = self._neg_view.get_buffer()
        negative = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)

        # Checkpoint
        idx = self._model_dropdown.get_selected()
        checkpoint = (
            self._filtered_checkpoints[idx] if idx < len(self._filtered_checkpoints) else None
        )

        # Seed
        seed_txt = self._seed_entry.get_text().strip()
        seed = int(seed_txt) if seed_txt.isdigit() else -1

        return {
            "workflow": self._current_workflow,
            "checkpoint": checkpoint,
            "prompt": prompt,
            "negative": negative,
            "style_suffix": style_suffix,
            "width": int(self._width_spin.get_value()),
            "height": int(self._height_spin.get_value()),
            "steps": int(self._steps_spin.get_value()),
            "cfg": self._cfg_spin.get_value(),
            "denoise": self._denoise_spin.get_value(),
            "seed": seed,
            "input_image_path": self._input_image_path,
            "mask_image_path": self._mask_image_path,
            "duration": self._duration_spin.get_value(),
            "elevation": self._elev_spin.get_value(),
            "azimuth": self._azim_spin.get_value(),
        }

    def set_checkpoints(self, checkpoints: list[str]) -> None:
        """Populate the full checkpoint list and refresh the dropdown."""
        self._all_checkpoints = checkpoints
        self._refresh_model_dropdown()

    def set_seed_text(self, text: str) -> None:
        """Set the seed entry text (e.g. after generation to show actual seed)."""
        self._seed_entry.set_text(text)

    def restore_session(
        self,
        *,
        workflow: str = "",
        prompt: str = "",
        negative: str = "",
        style: str = "none",
    ) -> None:
        """Restore session state into the controls."""
        # Workflow
        if workflow:
            for i, wt in enumerate(WorkflowType):
                if wt.name.lower() == workflow.lower() or wt.value == workflow:
                    self._workflow_dropdown.set_selected(i)
                    break

        # Prompt
        if prompt:
            self._prompt_view.get_buffer().set_text(prompt)

        # Negative
        if negative:
            self._neg_view.get_buffer().set_text(negative)

        # Style
        if style:
            style_keys = list(STYLE_PRESETS.keys())
            if style in style_keys:
                self._style_dropdown.set_selected(style_keys.index(style))
                self._current_style = style

    def get_session_state(self) -> dict[str, str]:
        """Return session-relevant state for persistence."""
        buf = self._prompt_view.get_buffer()
        prompt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        if prompt == _PROMPT_PLACEHOLDER:
            prompt = ""

        buf = self._neg_view.get_buffer()
        negative = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)

        return {
            "last_workflow": self._current_workflow.value,
            "last_prompt": prompt,
            "last_negative": negative,
            "last_style": self._current_style,
        }

    @property
    def current_workflow(self) -> WorkflowType:
        return self._current_workflow

    @property
    def has_valid_checkpoint(self) -> bool:
        idx = self._model_dropdown.get_selected()
        return idx < len(self._filtered_checkpoints)

    # ------------------------------------------------------------------
    # UI Building
    # ------------------------------------------------------------------

    def _build_ui(self) -> None:
        scroll = Gtk.ScrolledWindow(
            hscrollbar_policy=Gtk.PolicyType.NEVER,
            vscrollbar_policy=Gtk.PolicyType.AUTOMATIC,
            hexpand=False,
            vexpand=True,
        )
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=12)
        box.set_margin_start(12)
        box.set_margin_end(12)
        box.set_margin_top(12)
        box.set_margin_bottom(12)
        scroll.set_child(box)
        self.append(scroll)

        # WORKFLOW selector
        workflow_label = _section_label("WORKFLOW")
        workflow_label.set_tooltip_text(
            "Choose what you want to create:\n"
            "\u2022 Text to Image: Generate from a text description\n"
            "\u2022 Image to Image: Transform an existing image\n"
            "\u2022 Inpainting: Edit parts of an image\n"
            "\u2022 Audio: Generate music or sound effects\n"
            "\u2022 3D Novel View: Rotate around an object"
        )
        box.append(workflow_label)
        workflow_names = [WORKFLOW_SPECS[wt].name for wt in WorkflowType]
        self._workflow_dropdown = Gtk.DropDown.new_from_strings(workflow_names)
        self._workflow_dropdown.connect("notify::selected", self._on_workflow_changed)
        box.append(self._workflow_dropdown)

        # MODEL selector
        model_label = _section_label("MODEL")
        model_label.set_tooltip_text(
            "Choose which AI model to use.\n"
            "Different models have different styles and capabilities.\n"
            "\u2022 SD 1.5: Fast, works on most GPUs (4GB+)\n"
            "\u2022 SDXL: Higher quality, needs 8GB+ VRAM\n"
            "Download models using the button in the header bar."
        )
        box.append(model_label)
        self._model_dropdown = Gtk.DropDown.new_from_strings(["Loading..."])
        self._model_dropdown.connect("notify::selected", self._on_model_changed)
        box.append(self._model_dropdown)

        # INPUT IMAGE (for img2img, inpaint, 3d)
        self._input_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        input_label = _section_label("INPUT IMAGE")
        input_label.set_tooltip_text(
            "Select an image to use as a starting point.\n"
            "The AI will transform this image based on your prompt."
        )
        self._input_box.append(input_label)
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._input_label = Gtk.Label(
            label="None", xalign=0, hexpand=True, ellipsize=Pango.EllipsizeMode.MIDDLE
        )
        row.append(self._input_label)
        btn = Gtk.Button(label="Browse")
        btn.connect("clicked", self._pick_input_image)
        row.append(btn)
        self._input_box.append(row)
        self._input_box.set_visible(False)
        box.append(self._input_box)

        # MASK IMAGE (for inpaint only)
        self._mask_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        mask_label = _section_label("MASK")
        mask_label.set_tooltip_text(
            "Select a mask image.\n"
            "White areas = regions to regenerate\n"
            "Black areas = regions to keep unchanged\n"
            "Create masks in any image editor."
        )
        self._mask_box.append(mask_label)
        row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        self._mask_label = Gtk.Label(
            label="None", xalign=0, hexpand=True, ellipsize=Pango.EllipsizeMode.MIDDLE
        )
        row.append(self._mask_label)
        btn = Gtk.Button(label="Browse")
        btn.connect("clicked", self._pick_mask_image)
        row.append(btn)
        self._mask_box.append(row)
        self._mask_box.set_visible(False)
        box.append(self._mask_box)

        # PROMPT (hidden for 3D)
        self._prompt_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._build_prompt_section(self._prompt_box)
        box.append(self._prompt_box)

        # NEGATIVE (hidden for 3D)
        self._neg_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        self._build_negative_section(self._neg_box)
        box.append(self._neg_box)

        # PARAMETERS grid
        box.append(_section_label("PARAMETERS"))
        self._build_parameters_grid(box)

    def _build_prompt_section(self, container: Gtk.Box) -> None:
        """Build the prompt entry with overlay placeholder and template dropdown."""
        # Header row
        prompt_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        prompt_label = _section_label("PROMPT")
        prompt_label.set_tooltip_text(
            "Describe what you want to generate.\n\n"
            "Tips for better results:\n"
            "\u2022 Be specific: 'golden retriever puppy' not just 'dog'\n"
            "\u2022 Add style: 'oil painting', 'photograph', 'anime'\n"
            "\u2022 Describe quality: 'detailed', 'high resolution'\n"
            "\u2022 Use commas to separate concepts"
        )
        prompt_header.append(prompt_label)
        prompt_header.append(Gtk.Box(hexpand=True))  # spacer

        # Template dropdown
        template_names = ["Templates...", *list(PROMPT_TEMPLATES.keys())]
        self._template_dropdown = Gtk.DropDown.new_from_strings(template_names)
        self._template_dropdown.set_tooltip_text("Click to insert a starter prompt template")
        self._template_dropdown.connect("notify::selected", self._on_template_selected)
        prompt_header.append(self._template_dropdown)
        container.append(prompt_header)

        # Prompt text view with overlay placeholder
        overlay = Gtk.Overlay()
        self._prompt_view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self._prompt_view.set_size_request(-1, 80)
        frame = Gtk.Frame()
        frame.set_child(self._prompt_view)
        overlay.set_child(frame)

        self._prompt_placeholder = Gtk.Label(
            label=_PROMPT_PLACEHOLDER,
            css_classes=["dim-label"],
            halign=Gtk.Align.START,
            valign=Gtk.Align.START,
        )
        self._prompt_placeholder.set_margin_start(10)
        self._prompt_placeholder.set_margin_top(8)
        self._prompt_placeholder.set_can_target(False)
        overlay.add_overlay(self._prompt_placeholder)

        # Hide placeholder when the buffer has text
        buf = self._prompt_view.get_buffer()
        buf.connect("changed", self._on_prompt_buffer_changed)
        container.append(overlay)

        # Style preset buttons
        style_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        style_box.set_margin_top(4)
        style_label = Gtk.Label(label="Style:", css_classes=["dim-label"])
        style_label.set_tooltip_text("Add a style to your image automatically")
        style_box.append(style_label)

        style_names = [STYLE_PRESETS[k][1] for k in STYLE_PRESETS]
        self._style_dropdown = Gtk.DropDown.new_from_strings(style_names)
        self._style_dropdown.set_tooltip_text("Select a style to automatically add to your prompt")
        self._style_dropdown.connect("notify::selected", self._on_style_changed)
        style_box.append(self._style_dropdown)
        container.append(style_box)

    def _build_negative_section(self, container: Gtk.Box) -> None:
        """Build the negative prompt entry with defaults button."""
        neg_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        neg_label = _section_label("NEGATIVE")
        neg_label.set_tooltip_text(
            "Describe what you DON'T want in the image.\n\n"
            "Common negatives:\n"
            "\u2022 'blurry, low quality, distorted'\n"
            "\u2022 'text, watermark, signature'\n"
            "\u2022 'extra fingers, deformed hands'\n"
            "Leave empty if unsure \u2014 it's optional."
        )
        neg_header.append(neg_label)
        neg_header.append(Gtk.Box(hexpand=True))  # spacer

        defaults_btn = Gtk.Button(label="Use Defaults")
        defaults_btn.set_tooltip_text("Fill with recommended negative prompts")
        defaults_btn.connect("clicked", self._on_use_default_negative)
        neg_header.append(defaults_btn)
        container.append(neg_header)

        self._neg_view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self._neg_view.set_size_request(-1, 50)
        frame = Gtk.Frame()
        frame.set_child(self._neg_view)
        container.append(frame)

    def _build_parameters_grid(self, parent: Gtk.Box) -> None:
        """Build the parameter grid (size, steps, cfg, etc.)."""
        grid = Gtk.Grid(column_spacing=8, row_spacing=6)
        row_idx = 0

        # Size
        self._size_label = Gtk.Label(label="Size", xalign=0)
        self._size_label.set_tooltip_text(
            "Output image dimensions in pixels.\n"
            "\u2022 SD 1.5: Best at 512\u00d7512\n"
            "\u2022 SDXL: Best at 1024\u00d71024\n"
            "Larger sizes need more VRAM and time."
        )
        grid.attach(self._size_label, 0, row_idx, 1, 1)
        size_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        self._width_spin = Gtk.SpinButton.new_with_range(256, 2048, 64)
        self._width_spin.set_value(512)
        size_box.append(self._width_spin)
        size_box.append(Gtk.Label(label="\u00d7"))
        self._height_spin = Gtk.SpinButton.new_with_range(256, 2048, 64)
        self._height_spin.set_value(512)
        size_box.append(self._height_spin)
        self._size_box = size_box
        grid.attach(size_box, 1, row_idx, 1, 1)
        row_idx += 1

        # Size presets
        self._size_presets_label = Gtk.Label(label="", xalign=0)
        grid.attach(self._size_presets_label, 0, row_idx, 1, 1)
        size_presets_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        for key, (_w, _h, label, tooltip) in SIZE_PRESETS.items():
            btn = Gtk.Button(label=label)
            btn.set_tooltip_text(tooltip)
            btn.connect("clicked", self._on_size_preset_clicked, key)
            btn.add_css_class("flat")
            size_presets_box.append(btn)
        self._size_presets_box = size_presets_box
        grid.attach(size_presets_box, 1, row_idx, 1, 1)
        row_idx += 1

        # Quality presets
        quality_label = Gtk.Label(label="Quality", xalign=0)
        quality_label.set_tooltip_text("Choose a quality preset to set Steps and CFG automatically")
        grid.attach(quality_label, 0, row_idx, 1, 1)
        quality_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        for key, (_steps, _cfg, label, tooltip) in QUALITY_PRESETS.items():
            btn = Gtk.Button(label=label)
            btn.set_tooltip_text(tooltip)
            btn.connect("clicked", self._on_quality_preset_clicked, key)
            if key == "balanced":
                btn.add_css_class("suggested-action")
            else:
                btn.add_css_class("flat")
            quality_box.append(btn)
        grid.attach(quality_box, 1, row_idx, 1, 1)
        row_idx += 1

        # Steps
        steps_label = Gtk.Label(label="Steps", xalign=0)
        steps_label.set_tooltip_text(
            "Number of denoising iterations.\n"
            "\u2022 15-25: Fast, good for testing\n"
            "\u2022 25-40: Better quality\n"
            "\u2022 40+: Diminishing returns\n"
            "More steps = slower generation."
        )
        grid.attach(steps_label, 0, row_idx, 1, 1)
        self._steps_spin = Gtk.SpinButton.new_with_range(1, 150, 1)
        self._steps_spin.set_value(20)
        self._steps_spin.set_tooltip_text("Start with 20, increase for more detail")
        grid.attach(self._steps_spin, 1, row_idx, 1, 1)
        row_idx += 1

        # CFG
        self._cfg_label = Gtk.Label(label="CFG", xalign=0)
        self._cfg_label.set_tooltip_text(
            "Classifier-Free Guidance scale.\n"
            "Controls how closely the AI follows your prompt.\n"
            "\u2022 1-5: Creative, may ignore prompt\n"
            "\u2022 7-8: Balanced (recommended)\n"
            "\u2022 10+: Strict, can look artificial"
        )
        grid.attach(self._cfg_label, 0, row_idx, 1, 1)
        self._cfg_spin = Gtk.SpinButton.new_with_range(1.0, 30.0, 0.5)
        self._cfg_spin.set_value(7.0)
        self._cfg_spin.set_digits(1)
        self._cfg_spin.set_tooltip_text("7.0 is a good starting point")
        grid.attach(self._cfg_spin, 1, row_idx, 1, 1)
        row_idx += 1

        # Denoise (img2img, inpaint)
        self._denoise_label = Gtk.Label(label="Denoise", xalign=0)
        self._denoise_label.set_tooltip_text(
            "How much to change the input image.\n"
            "\u2022 0.0: No change (useless)\n"
            "\u2022 0.3-0.5: Subtle changes, keeps structure\n"
            "\u2022 0.7-0.8: Significant changes\n"
            "\u2022 1.0: Complete regeneration"
        )
        grid.attach(self._denoise_label, 0, row_idx, 1, 1)
        self._denoise_spin = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.05)
        self._denoise_spin.set_value(0.75)
        self._denoise_spin.set_digits(2)
        self._denoise_spin.set_tooltip_text("0.75 balances creativity and consistency")
        grid.attach(self._denoise_spin, 1, row_idx, 1, 1)
        self._denoise_label.set_visible(False)
        self._denoise_spin.set_visible(False)
        row_idx += 1

        # Duration (audio)
        self._duration_label = Gtk.Label(label="Duration (s)", xalign=0)
        self._duration_label.set_tooltip_text(
            "Length of audio to generate in seconds.\nLonger durations take more time and VRAM."
        )
        grid.attach(self._duration_label, 0, row_idx, 1, 1)
        self._duration_spin = Gtk.SpinButton.new_with_range(1.0, 60.0, 1.0)
        self._duration_spin.set_value(30.0)
        self._duration_spin.set_digits(0)
        self._duration_spin.set_tooltip_text("Start with 10-30 seconds")
        grid.attach(self._duration_spin, 1, row_idx, 1, 1)
        self._duration_label.set_visible(False)
        self._duration_spin.set_visible(False)
        row_idx += 1

        # Elevation (3D)
        self._elev_label = Gtk.Label(label="Elevation", xalign=0)
        self._elev_label.set_tooltip_text(
            "Camera angle up/down from the object.\n"
            "\u2022 Positive: Looking down at object\n"
            "\u2022 Negative: Looking up at object\n"
            "\u2022 0: Eye level"
        )
        grid.attach(self._elev_label, 0, row_idx, 1, 1)
        self._elev_spin = Gtk.SpinButton.new_with_range(-90, 90, 5)
        self._elev_spin.set_value(0)
        self._elev_spin.set_tooltip_text("Vertical camera angle in degrees")
        grid.attach(self._elev_spin, 1, row_idx, 1, 1)
        self._elev_label.set_visible(False)
        self._elev_spin.set_visible(False)
        row_idx += 1

        # Azimuth (3D)
        self._azim_label = Gtk.Label(label="Azimuth", xalign=0)
        self._azim_label.set_tooltip_text(
            "Camera rotation around the object.\n"
            "\u2022 0: Same angle as input\n"
            "\u2022 90: Right side view\n"
            "\u2022 -90: Left side view\n"
            "\u2022 180: Back view"
        )
        grid.attach(self._azim_label, 0, row_idx, 1, 1)
        self._azim_spin = Gtk.SpinButton.new_with_range(-180, 180, 5)
        self._azim_spin.set_value(0)
        self._azim_spin.set_tooltip_text("Horizontal camera angle in degrees")
        grid.attach(self._azim_spin, 1, row_idx, 1, 1)
        self._azim_label.set_visible(False)
        self._azim_spin.set_visible(False)
        row_idx += 1

        # Seed
        seed_label = Gtk.Label(label="Seed", xalign=0)
        seed_label.set_tooltip_text(
            "Random number that determines the output.\n"
            "\u2022 Empty/Random: Different result each time\n"
            "\u2022 Same seed + same settings = same image\n"
            "Use this to recreate or refine results."
        )
        grid.attach(seed_label, 0, row_idx, 1, 1)
        self._seed_entry = Gtk.Entry(placeholder_text="Random")
        self._seed_entry.set_tooltip_text(
            "Leave empty for random, or enter a number to reproduce results"
        )
        grid.attach(self._seed_entry, 1, row_idx, 1, 1)

        parent.append(grid)

    # ------------------------------------------------------------------
    # Event handlers
    # ------------------------------------------------------------------

    def _on_workflow_changed(self, dropdown, _param) -> None:
        """Handle workflow selection change."""
        idx = dropdown.get_selected()
        workflows = list(WorkflowType)
        if idx >= len(workflows):
            return

        self._current_workflow = workflows[idx]
        spec = WORKFLOW_SPECS[self._current_workflow]
        logger.debug("Workflow changed to %s", self._current_workflow.name)

        self._refresh_model_dropdown()

        # Show/hide sections based on workflow needs
        self._input_box.set_visible(spec.needs_input_image)
        self._mask_box.set_visible(spec.needs_mask)
        self._prompt_box.set_visible(spec.needs_prompt)
        self._neg_box.set_visible(spec.needs_prompt)

        # Show/hide parameters
        self._size_label.set_visible(spec.needs_size)
        self._size_box.set_visible(spec.needs_size)

        is_img2img = self._current_workflow in (WorkflowType.IMG2IMG, WorkflowType.INPAINT)
        self._denoise_label.set_visible(is_img2img)
        self._denoise_spin.set_visible(is_img2img)

        is_audio = self._current_workflow == WorkflowType.AUDIO
        self._duration_label.set_visible(is_audio)
        self._duration_spin.set_visible(is_audio)

        is_3d = self._current_workflow == WorkflowType.THREE_D
        self._elev_label.set_visible(is_3d)
        self._elev_spin.set_visible(is_3d)
        self._azim_label.set_visible(is_3d)
        self._azim_spin.set_visible(is_3d)

        # Set defaults from spec
        self._steps_spin.set_value(spec.default_steps)
        self._cfg_spin.set_value(spec.default_cfg)
        self._denoise_spin.set_value(spec.default_denoise)

        if self.on_workflow_changed:
            self.on_workflow_changed()

    def _on_model_changed(self, dropdown, _param) -> None:
        """Detect XL models and update default sizes accordingly."""
        idx = dropdown.get_selected()
        if idx >= len(self._filtered_checkpoints):
            return

        model_name = self._filtered_checkpoints[idx]
        is_xl = _is_xl_checkpoint(model_name)
        if is_xl != self._is_xl_model:
            self._is_xl_model = is_xl
            if is_xl:
                self._width_spin.set_value(1024)
                self._height_spin.set_value(1024)
            else:
                self._width_spin.set_value(512)
                self._height_spin.set_value(512)
            logger.debug("Model changed: XL=%s, updated default size", is_xl)

    def _on_size_preset_clicked(self, _button, preset_key: str) -> None:
        presets = SIZE_PRESETS_XL if self._is_xl_model else SIZE_PRESETS
        if preset_key in presets:
            w, h, _, _ = presets[preset_key]
            self._width_spin.set_value(w)
            self._height_spin.set_value(h)
            logger.debug("Size preset applied: %s (%dx%d)", preset_key, w, h)

    def _on_quality_preset_clicked(self, _button, preset_key: str) -> None:
        if preset_key in QUALITY_PRESETS:
            steps, cfg, _, _ = QUALITY_PRESETS[preset_key]
            self._steps_spin.set_value(steps)
            self._cfg_spin.set_value(cfg)
            logger.debug(
                "Quality preset applied: %s (steps=%d, cfg=%.1f)",
                preset_key,
                steps,
                cfg,
            )

    def _on_template_selected(self, dropdown, _param) -> None:
        idx = dropdown.get_selected()
        if idx == 0:  # "Templates..." placeholder
            return
        template_keys = list(PROMPT_TEMPLATES.keys())
        if idx - 1 < len(template_keys):
            key = template_keys[idx - 1]
            self._prompt_view.get_buffer().set_text(PROMPT_TEMPLATES[key])
            logger.debug("Template inserted: %s", key)
        dropdown.set_selected(0)

    def _on_style_changed(self, dropdown, _param) -> None:
        idx = dropdown.get_selected()
        style_keys = list(STYLE_PRESETS.keys())
        if idx < len(style_keys):
            self._current_style = style_keys[idx]
            logger.debug("Style changed: %s", self._current_style)

    def _on_use_default_negative(self, _button) -> None:
        self._neg_view.get_buffer().set_text(DEFAULT_NEGATIVE)
        logger.debug("Default negative prompt applied")

    def _on_prompt_buffer_changed(self, buf) -> None:
        """Toggle the overlay placeholder based on buffer contents."""
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        self._prompt_placeholder.set_visible(len(text) == 0)

    # ------------------------------------------------------------------
    # File pickers
    # ------------------------------------------------------------------

    def _pick_input_image(self, _btn) -> None:
        dialog = Gtk.FileDialog(title="Select Input Image")
        f = Gtk.FileFilter()
        f.set_name("Images")
        f.add_mime_type("image/*")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f)
        dialog.set_filters(filters)
        dialog.open(self.get_root(), None, self._on_input_picked)

    def _on_input_picked(self, dialog, result) -> None:
        try:
            file = dialog.open_finish(result)
            if file:
                self._input_image_path = file.get_path()
                self._input_label.set_text(os.path.basename(self._input_image_path))
        except GLib.Error:
            pass

    def _pick_mask_image(self, _btn) -> None:
        dialog = Gtk.FileDialog(title="Select Mask Image")
        f = Gtk.FileFilter()
        f.set_name("Images")
        f.add_mime_type("image/*")
        filters = Gio.ListStore.new(Gtk.FileFilter)
        filters.append(f)
        dialog.set_filters(filters)
        dialog.open(self.get_root(), None, self._on_mask_picked)

    def _on_mask_picked(self, dialog, result) -> None:
        try:
            file = dialog.open_finish(result)
            if file:
                self._mask_image_path = file.get_path()
                self._mask_label.set_text(os.path.basename(self._mask_image_path))
        except GLib.Error:
            pass

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _refresh_model_dropdown(self) -> None:
        """Refresh the model dropdown for the current workflow."""
        self._filtered_checkpoints = get_models_for_workflow(
            self._all_checkpoints, self._current_workflow
        )
        if self._filtered_checkpoints:
            self._model_dropdown.set_model(Gtk.StringList.new(self._filtered_checkpoints))
        else:
            self._model_dropdown.set_model(Gtk.StringList.new(["No compatible models"]))
