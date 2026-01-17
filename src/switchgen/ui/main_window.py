"""Main application window."""

import os
import threading
from typing import Optional

from ..core.logging import get_logger

logger = get_logger(__name__)

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
    "photo": (", professional photograph, photorealistic, 8k, detailed", "Photorealistic", "Realistic photograph style"),
    "anime": (", anime style, anime art, vibrant colors, detailed", "Anime", "Japanese anime style"),
    "oil": (", oil painting, painterly, classical art, brushstrokes", "Oil Painting", "Classical oil painting style"),
    "digital": (", digital art, concept art, artstation, detailed illustration", "Digital Art", "Modern digital illustration"),
    "watercolor": (", watercolor painting, soft colors, artistic, delicate", "Watercolor", "Soft watercolor painting style"),
    "3d": (", 3d render, octane render, unreal engine, realistic lighting", "3D Render", "3D rendered CGI style"),
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
DEFAULT_NEGATIVE = "blurry, low quality, distorted, deformed, ugly, bad anatomy, watermark, text, signature"


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
        self._current_style: str = "none"  # Current style preset
        self._is_xl_model: bool = False  # Whether current model is SDXL

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
        workflow_label = self._label("WORKFLOW")
        workflow_label.set_tooltip_text(
            "Choose what you want to create:\n"
            "• Text to Image: Generate from a text description\n"
            "• Image to Image: Transform an existing image\n"
            "• Inpainting: Edit parts of an image\n"
            "• Audio: Generate music or sound effects\n"
            "• 3D Novel View: Rotate around an object"
        )
        box.append(workflow_label)
        workflow_names = [WORKFLOW_SPECS[wt].name for wt in WorkflowType]
        self.workflow_dropdown = Gtk.DropDown.new_from_strings(workflow_names)
        self.workflow_dropdown.connect("notify::selected", self._on_workflow_changed)
        box.append(self.workflow_dropdown)

        # MODEL selector (filtered by workflow)
        model_label = self._label("MODEL")
        model_label.set_tooltip_text(
            "Choose which AI model to use.\n"
            "Different models have different styles and capabilities.\n"
            "• SD 1.5: Fast, works on most GPUs (4GB+)\n"
            "• SDXL: Higher quality, needs 8GB+ VRAM\n"
            "Download models using the button in the header bar."
        )
        box.append(model_label)
        self.model_dropdown = Gtk.DropDown.new_from_strings(["Loading..."])
        self.model_dropdown.connect("notify::selected", self._on_model_changed)
        box.append(self.model_dropdown)

        # INPUT IMAGE (for img2img, inpaint, 3d)
        self.input_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)
        input_label = self._label("INPUT IMAGE")
        input_label.set_tooltip_text(
            "Select an image to use as a starting point.\n"
            "The AI will transform this image based on your prompt."
        )
        self.input_box.append(input_label)
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
        mask_label = self._label("MASK")
        mask_label.set_tooltip_text(
            "Select a mask image.\n"
            "White areas = regions to regenerate\n"
            "Black areas = regions to keep unchanged\n"
            "Create masks in any image editor."
        )
        self.mask_box.append(mask_label)
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

        # Prompt header with template dropdown
        prompt_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        prompt_label = self._label("PROMPT")
        prompt_label.set_tooltip_text(
            "Describe what you want to generate.\n\n"
            "Tips for better results:\n"
            "• Be specific: 'golden retriever puppy' not just 'dog'\n"
            "• Add style: 'oil painting', 'photograph', 'anime'\n"
            "• Describe quality: 'detailed', 'high resolution'\n"
            "• Use commas to separate concepts"
        )
        prompt_header.append(prompt_label)
        prompt_header.append(Gtk.Box(hexpand=True))  # spacer

        # Template dropdown
        template_names = ["Templates..."] + list(PROMPT_TEMPLATES.keys())
        self.template_dropdown = Gtk.DropDown.new_from_strings(template_names)
        self.template_dropdown.set_tooltip_text("Click to insert a starter prompt template")
        self.template_dropdown.connect("notify::selected", self._on_template_selected)
        prompt_header.append(self.template_dropdown)
        self.prompt_box.append(prompt_header)

        self.prompt_view = Gtk.TextView(wrap_mode=Gtk.WrapMode.WORD_CHAR)
        self.prompt_view.set_size_request(-1, 80)
        # Set placeholder-like hint
        buf = self.prompt_view.get_buffer()
        buf.set_text("Describe what you want to see...")
        # GTK4 uses event controllers for focus
        focus_controller = Gtk.EventControllerFocus()
        focus_controller.connect("enter", self._on_prompt_focus_enter)
        self.prompt_view.add_controller(focus_controller)
        frame = Gtk.Frame()
        frame.set_child(self.prompt_view)
        self.prompt_box.append(frame)

        # Style preset buttons
        style_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        style_box.set_margin_top(4)
        style_label = Gtk.Label(label="Style:", css_classes=["dim-label"])
        style_label.set_tooltip_text("Add a style to your image automatically")
        style_box.append(style_label)

        # Create style dropdown
        style_names = [STYLE_PRESETS[k][1] for k in STYLE_PRESETS.keys()]
        self.style_dropdown = Gtk.DropDown.new_from_strings(style_names)
        self.style_dropdown.set_tooltip_text("Select a style to automatically add to your prompt")
        self.style_dropdown.connect("notify::selected", self._on_style_changed)
        style_box.append(self.style_dropdown)
        self.prompt_box.append(style_box)

        box.append(self.prompt_box)

        # NEGATIVE (hidden for 3D)
        self.neg_box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=4)

        # Negative header with defaults button
        neg_header = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
        neg_label = self._label("NEGATIVE")
        neg_label.set_tooltip_text(
            "Describe what you DON'T want in the image.\n\n"
            "Common negatives:\n"
            "• 'blurry, low quality, distorted'\n"
            "• 'text, watermark, signature'\n"
            "• 'extra fingers, deformed hands'\n"
            "Leave empty if unsure - it's optional."
        )
        neg_header.append(neg_label)
        neg_header.append(Gtk.Box(hexpand=True))  # spacer

        defaults_btn = Gtk.Button(label="Use Defaults")
        defaults_btn.set_tooltip_text("Fill with recommended negative prompts")
        defaults_btn.connect("clicked", self._on_use_default_negative)
        neg_header.append(defaults_btn)
        self.neg_box.append(neg_header)

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
        self.size_label.set_tooltip_text(
            "Output image dimensions in pixels.\n"
            "• SD 1.5: Best at 512×512\n"
            "• SDXL: Best at 1024×1024\n"
            "Larger sizes need more VRAM and time."
        )
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

        # Size presets
        self.size_presets_label = Gtk.Label(label="", xalign=0)
        grid.attach(self.size_presets_label, 0, row_idx, 1, 1)
        size_presets_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        for key, (w, h, label, tooltip) in SIZE_PRESETS.items():
            btn = Gtk.Button(label=label)
            btn.set_tooltip_text(tooltip)
            btn.connect("clicked", self._on_size_preset_clicked, key)
            btn.add_css_class("flat")
            size_presets_box.append(btn)
        self.size_presets_box = size_presets_box
        grid.attach(size_presets_box, 1, row_idx, 1, 1)
        row_idx += 1

        # Quality presets
        quality_label = Gtk.Label(label="Quality", xalign=0)
        quality_label.set_tooltip_text("Choose a quality preset to set Steps and CFG automatically")
        grid.attach(quality_label, 0, row_idx, 1, 1)
        quality_box = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=4)
        for key, (steps, cfg, label, tooltip) in QUALITY_PRESETS.items():
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
            "• 15-25: Fast, good for testing\n"
            "• 25-40: Better quality\n"
            "• 40+: Diminishing returns\n"
            "More steps = slower generation."
        )
        grid.attach(steps_label, 0, row_idx, 1, 1)
        self.steps_spin = Gtk.SpinButton.new_with_range(1, 150, 1)
        self.steps_spin.set_value(20)
        self.steps_spin.set_tooltip_text("Start with 20, increase for more detail")
        grid.attach(self.steps_spin, 1, row_idx, 1, 1)
        row_idx += 1

        # CFG
        self.cfg_label = Gtk.Label(label="CFG", xalign=0)
        self.cfg_label.set_tooltip_text(
            "Classifier-Free Guidance scale.\n"
            "Controls how closely the AI follows your prompt.\n"
            "• 1-5: Creative, may ignore prompt\n"
            "• 7-8: Balanced (recommended)\n"
            "• 10+: Strict, can look artificial"
        )
        grid.attach(self.cfg_label, 0, row_idx, 1, 1)
        self.cfg_spin = Gtk.SpinButton.new_with_range(1.0, 30.0, 0.5)
        self.cfg_spin.set_value(7.0)
        self.cfg_spin.set_digits(1)
        self.cfg_spin.set_tooltip_text("7.0 is a good starting point")
        grid.attach(self.cfg_spin, 1, row_idx, 1, 1)
        row_idx += 1

        # Denoise (img2img, inpaint)
        self.denoise_label = Gtk.Label(label="Denoise", xalign=0)
        self.denoise_label.set_tooltip_text(
            "How much to change the input image.\n"
            "• 0.0: No change (useless)\n"
            "• 0.3-0.5: Subtle changes, keeps structure\n"
            "• 0.7-0.8: Significant changes\n"
            "• 1.0: Complete regeneration"
        )
        grid.attach(self.denoise_label, 0, row_idx, 1, 1)
        self.denoise_spin = Gtk.SpinButton.new_with_range(0.0, 1.0, 0.05)
        self.denoise_spin.set_value(0.75)
        self.denoise_spin.set_digits(2)
        self.denoise_spin.set_tooltip_text("0.75 balances creativity and consistency")
        grid.attach(self.denoise_spin, 1, row_idx, 1, 1)
        self.denoise_label.set_visible(False)
        self.denoise_spin.set_visible(False)
        row_idx += 1

        # Duration (audio)
        self.duration_label = Gtk.Label(label="Duration (s)", xalign=0)
        self.duration_label.set_tooltip_text(
            "Length of audio to generate in seconds.\n"
            "Longer durations take more time and VRAM."
        )
        grid.attach(self.duration_label, 0, row_idx, 1, 1)
        self.duration_spin = Gtk.SpinButton.new_with_range(1.0, 60.0, 1.0)
        self.duration_spin.set_value(30.0)
        self.duration_spin.set_digits(0)
        self.duration_spin.set_tooltip_text("Start with 10-30 seconds")
        grid.attach(self.duration_spin, 1, row_idx, 1, 1)
        self.duration_label.set_visible(False)
        self.duration_spin.set_visible(False)
        row_idx += 1

        # Elevation (3D)
        self.elev_label = Gtk.Label(label="Elevation", xalign=0)
        self.elev_label.set_tooltip_text(
            "Camera angle up/down from the object.\n"
            "• Positive: Looking down at object\n"
            "• Negative: Looking up at object\n"
            "• 0: Eye level"
        )
        grid.attach(self.elev_label, 0, row_idx, 1, 1)
        self.elev_spin = Gtk.SpinButton.new_with_range(-90, 90, 5)
        self.elev_spin.set_value(0)
        self.elev_spin.set_tooltip_text("Vertical camera angle in degrees")
        grid.attach(self.elev_spin, 1, row_idx, 1, 1)
        self.elev_label.set_visible(False)
        self.elev_spin.set_visible(False)
        row_idx += 1

        # Azimuth (3D)
        self.azim_label = Gtk.Label(label="Azimuth", xalign=0)
        self.azim_label.set_tooltip_text(
            "Camera rotation around the object.\n"
            "• 0: Same angle as input\n"
            "• 90: Right side view\n"
            "• -90: Left side view\n"
            "• 180: Back view"
        )
        grid.attach(self.azim_label, 0, row_idx, 1, 1)
        self.azim_spin = Gtk.SpinButton.new_with_range(-180, 180, 5)
        self.azim_spin.set_value(0)
        self.azim_spin.set_tooltip_text("Horizontal camera angle in degrees")
        grid.attach(self.azim_spin, 1, row_idx, 1, 1)
        self.azim_label.set_visible(False)
        self.azim_spin.set_visible(False)
        row_idx += 1

        # Seed
        seed_label = Gtk.Label(label="Seed", xalign=0)
        seed_label.set_tooltip_text(
            "Random number that determines the output.\n"
            "• Empty/Random: Different result each time\n"
            "• Same seed + same settings = same image\n"
            "Use this to recreate or refine results."
        )
        grid.attach(seed_label, 0, row_idx, 1, 1)
        self.seed_entry = Gtk.Entry(placeholder_text="Random")
        self.seed_entry.set_tooltip_text("Leave empty for random, or enter a number to reproduce results")
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

        # Info banner for no models
        self.info_banner = Adw.Banner(
            title="No models installed. Click the download button to get started.",
            button_label="Download Models",
            revealed=False,
        )
        self.info_banner.connect("button-clicked", self._on_download_models_clicked)
        box.append(self.info_banner)

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
            ("Use Presets", "Click Quality presets (Fast/Balanced/Quality) to set good defaults."),
            ("Try Templates", "Use the Templates dropdown for starter prompts you can customize."),
            ("Add Style", "Select a Style to automatically enhance your prompt with artistic terms."),
            ("Iterate", "Generate quickly with 'Fast' preset, then increase quality once you like the result."),
            ("Save Your Seed", "The Seed number lets you recreate the exact same image later."),
        ]

        for title, desc in tips:
            tip_row = Gtk.Box(orientation=Gtk.Orientation.HORIZONTAL, spacing=8)
            tip_title = Gtk.Label(label=f"• {title}:", xalign=0, css_classes=["heading"])
            tip_title.set_size_request(120, -1)
            tip_row.append(tip_title)
            tip_desc = Gtk.Label(label=desc, xalign=0, wrap=True, wrap_mode=Pango.WrapMode.WORD_CHAR)
            tip_desc.set_hexpand(True)
            tip_row.append(tip_desc)
            tips_box.append(tip_row)

        tips_expander.set_child(tips_box)
        box.append(tips_expander)

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
        logger.debug("Workflow changed to %s", self._current_workflow.name)

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

    def _on_model_changed(self, dropdown, _param) -> None:
        """Handle model selection change - detect XL models and update sizes."""
        idx = dropdown.get_selected()
        if idx >= len(self._filtered_checkpoints):
            return

        model_name = self._filtered_checkpoints[idx].lower()
        # Detect if this is an SDXL model
        is_xl = "xl" in model_name or "sdxl" in model_name
        if is_xl != self._is_xl_model:
            self._is_xl_model = is_xl
            # Update to appropriate default size
            if is_xl:
                self.width_spin.set_value(1024)
                self.height_spin.set_value(1024)
            else:
                self.width_spin.set_value(512)
                self.height_spin.set_value(512)
            logger.debug("Model changed: XL=%s, updated default size", is_xl)

    def _on_size_preset_clicked(self, button, preset_key: str) -> None:
        """Apply a size preset."""
        presets = SIZE_PRESETS_XL if self._is_xl_model else SIZE_PRESETS
        if preset_key in presets:
            w, h, _, _ = presets[preset_key]
            self.width_spin.set_value(w)
            self.height_spin.set_value(h)
            logger.debug("Size preset applied: %s (%dx%d)", preset_key, w, h)

    def _on_quality_preset_clicked(self, button, preset_key: str) -> None:
        """Apply a quality preset."""
        if preset_key in QUALITY_PRESETS:
            steps, cfg, _, _ = QUALITY_PRESETS[preset_key]
            self.steps_spin.set_value(steps)
            self.cfg_spin.set_value(cfg)
            logger.debug("Quality preset applied: %s (steps=%d, cfg=%.1f)", preset_key, steps, cfg)

    def _on_template_selected(self, dropdown, _param) -> None:
        """Insert a prompt template."""
        idx = dropdown.get_selected()
        if idx == 0:  # "Templates..." placeholder
            return
        template_keys = list(PROMPT_TEMPLATES.keys())
        if idx - 1 < len(template_keys):
            key = template_keys[idx - 1]
            template = PROMPT_TEMPLATES[key]
            buf = self.prompt_view.get_buffer()
            buf.set_text(template)
            logger.debug("Template inserted: %s", key)
        # Reset dropdown to placeholder
        dropdown.set_selected(0)

    def _on_style_changed(self, dropdown, _param) -> None:
        """Handle style preset change."""
        idx = dropdown.get_selected()
        style_keys = list(STYLE_PRESETS.keys())
        if idx < len(style_keys):
            self._current_style = style_keys[idx]
            logger.debug("Style changed: %s", self._current_style)

    def _on_use_default_negative(self, button) -> None:
        """Fill negative prompt with defaults."""
        buf = self.neg_view.get_buffer()
        buf.set_text(DEFAULT_NEGATIVE)
        logger.debug("Default negative prompt applied")

    def _on_prompt_focus_enter(self, controller) -> None:
        """Clear placeholder text when prompt gets focus."""
        buf = self.prompt_view.get_buffer()
        text = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        if text == "Describe what you want to see...":
            buf.set_text("")

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
            # No valid model selected - show banner
            self.info_banner.set_title("Please download a model first to generate images.")
            self.info_banner.set_revealed(True)
            logger.warning("Generate clicked but no model selected")
            return
        checkpoint = self._filtered_checkpoints[idx]
        logger.info("Generate clicked (workflow=%s, model=%s)", self._current_workflow.name, checkpoint)

        # Get prompts
        buf = self.prompt_view.get_buffer()
        prompt = buf.get_text(buf.get_start_iter(), buf.get_end_iter(), False)
        # Clear placeholder text if still there
        if prompt == "Describe what you want to see...":
            prompt = ""
        if not prompt.strip():
            prompt = "high quality, detailed"

        # Apply style suffix
        if self._current_style and self._current_style in STYLE_PRESETS:
            style_suffix, _, _ = STYLE_PRESETS[self._current_style]
            if style_suffix:
                prompt = prompt.rstrip() + style_suffix
                logger.debug("Style applied: %s", self._current_style)

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
                steps=steps, cfg=cfg, seed=seed, capture_id="default",
            )
        elif self._current_workflow == WorkflowType.IMG2IMG:
            workflow, actual_seed = build_img2img_memory_workflow(
                checkpoint=checkpoint, image_path=self._input_image_path,
                prompt=prompt, negative_prompt=negative,
                denoise=self.denoise_spin.get_value(),
                steps=steps, cfg=cfg, seed=seed, capture_id="default",
            )
        elif self._current_workflow == WorkflowType.INPAINT:
            workflow, actual_seed = build_inpaint_workflow(
                checkpoint=checkpoint, image_path=self._input_image_path,
                mask_path=self._mask_image_path, prompt=prompt, negative_prompt=negative,
                denoise=self.denoise_spin.get_value(),
                steps=steps, cfg=cfg, seed=seed, capture_id="default",
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
                steps=steps, cfg=cfg, seed=seed, capture_id="default",
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
            self._current_workflow.name, checkpoint, steps, cfg, actual_seed
        )
        self._queue.submit(workflow=workflow, on_progress=on_progress, on_complete=on_complete)

    def _update_progress(self, info: ProgressInfo) -> bool:
        if info.total_steps > 0:
            progress = info.current_step / info.total_steps
            self.progress_bar.set_fraction(progress)
            # Update button text with progress
            pct = int(progress * 100)
            self.generate_btn.set_label(f"Generating... {pct}%")
        return False

    def _on_complete(self, job: GenerationJob) -> bool:
        logger.info("_on_complete called: result=%s", job.result is not None)
        if job.result:
            logger.info("  success=%s, images=%s, error=%s",
                       job.result.success,
                       job.result.images is not None,
                       job.result.error)
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
        self.progress_bar.set_visible(False)
        # Only re-enable if we have models
        if self._all_checkpoints:
            self.generate_btn.set_sensitive(True)
            self.generate_btn.set_label("GENERATE")
        else:
            self.generate_btn.set_sensitive(False)
            self.generate_btn.set_label("No Models")

    # =========================================================================
    # Model Download
    # =========================================================================

    def _on_download_models_clicked(self, button) -> None:
        """Open the model download dialog."""
        config = get_config()
        dialog = ModelDownloadDialog(config.paths.models_dir)
        dialog.connect("closed", self._on_download_dialog_closed)
        dialog.present(self)

    def _refresh_models(self) -> None:
        """Refresh the model list after downloading."""
        # Re-scan for checkpoints
        self._all_checkpoints = get_available_checkpoints()

        # Update workflow dropdown to refresh filtered models
        self._on_workflow_changed(self.workflow_dropdown, None)

        # Update UI based on whether we now have models
        if self._all_checkpoints:
            self.generate_btn.set_sensitive(True)
            self.generate_btn.set_label("GENERATE")
            self.info_banner.set_revealed(False)
        else:
            self.generate_btn.set_sensitive(False)
            self.generate_btn.set_label("No Models")
            self.info_banner.set_revealed(True)

    # =========================================================================
    # Initialization
    # =========================================================================

    def _init_comfy(self) -> None:
        """Initialize ComfyUI in background."""
        logger.info("Initializing ComfyUI...")
        try:
            self._queue = get_queue()
            self._all_checkpoints = get_available_checkpoints()
            logger.info("ComfyUI initialized (checkpoints=%d)", len(self._all_checkpoints))
            GLib.idle_add(self._on_ready)
        except Exception as e:
            logger.error("ComfyUI initialization failed: %s", e, exc_info=True)
            GLib.idle_add(self._on_error, str(e))

    def _on_ready(self) -> bool:
        """ComfyUI ready."""
        logger.info("Main window ready")
        # Trigger workflow change to filter models
        self._on_workflow_changed(self.workflow_dropdown, None)
        self._update_vram()

        # Check if any models are available
        if not self._all_checkpoints:
            # No models installed - show banner, keep button disabled
            self.generate_btn.set_sensitive(False)
            self.generate_btn.set_label("No Models")
            self.info_banner.set_revealed(True)
            # Auto-open download dialog on first launch with no models
            GLib.idle_add(self._show_welcome_download_dialog)
        else:
            # Models available - enable generate
            self.generate_btn.set_sensitive(True)
            self.generate_btn.set_label("GENERATE")
            self.info_banner.set_revealed(False)

        return False

    def _show_welcome_download_dialog(self) -> bool:
        """Show download dialog for first-time users with no models."""
        config = get_config()
        dialog = ModelDownloadDialog(config.paths.models_dir)
        dialog.connect("closed", self._on_download_dialog_closed)
        dialog.present(self)
        return False

    def _on_download_dialog_closed(self, dialog) -> None:
        """Handle download dialog being closed - refresh model list."""
        self._refresh_models()

    def _on_error(self, error: str) -> bool:
        """ComfyUI error."""
        self.generate_btn.set_label("Error")
        logger.error("ComfyUI error: %s", error)
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
