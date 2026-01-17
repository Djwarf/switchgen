# SwitchGen

A Linux-native AI image generator with a GTK4 interface. Uses ComfyUI as an embedded library for Stable Diffusion workflows.

## Features

- Text-to-image generation
- Image-to-image transformation
- Inpainting with mask support
- Audio generation (Stable Audio)
- 3D novel view synthesis (Zero123)
- Built-in model downloader for HuggingFace models
- Real-time generation progress
- VRAM monitoring and cleanup

## Installation

### Arch Linux (AUR)

```bash
yay -S switchgen-git
```

### Manual Installation

#### 1. Clone with submodules

```bash
git clone --recursive https://github.com/Djwarf/switchgen.git
cd switchgen
```

If already cloned without submodules:

```bash
git submodule update --init --recursive
```

#### 2. Install system dependencies

```bash
# Arch Linux
sudo pacman -S gtk4 libadwaita python-gobject python-pytorch python-torchvision

# Other distributions
# Install GTK4, libadwaita, PyGObject, and PyTorch with CUDA support
```

#### 3. Install Python package

```bash
pip install -e .
```

## Usage

Launch the application:

```bash
switchgen
```

Or run as a module:

```bash
python -m switchgen
```

### First Run

1. Click the download button in the header bar to open the model manager
2. Download required models (checkpoints, VAE, CLIP, etc.)
3. Select a workflow type and checkpoint
4. Enter a prompt and click Generate

## Project Structure

```
switchgen/
├── src/switchgen/
│   ├── core/
│   │   ├── comfy_init.py    # ComfyUI initialization
│   │   ├── config.py        # Path and app configuration
│   │   ├── downloader.py    # HuggingFace model downloader
│   │   ├── engine.py        # Generation engine
│   │   ├── models.py        # Model catalog
│   │   ├── queue.py         # Job queue system
│   │   └── workflows.py     # Workflow builders
│   │
│   ├── ui/
│   │   ├── main_window.py   # Main GTK4 window
│   │   ├── model_dialog.py  # Model download dialog
│   │   └── styles/          # CSS themes
│   │
│   └── resources/
│       └── fonts/           # Bundled fonts
│
├── vendor/
│   └── ComfyUI/             # Bundled ComfyUI engine (submodule)
│
├── models/                  # Downloaded models
├── custom_nodes/            # User custom nodes
├── output/                  # Generated images
└── temp/                    # Temporary files
```

## Supported Workflows

| Workflow | Description | Required Models |
|----------|-------------|-----------------|
| Text2Img | Generate images from text prompts | Checkpoint, VAE |
| Img2Img | Transform existing images | Checkpoint, VAE |
| Inpaint | Edit specific regions with masks | Checkpoint, VAE |
| Audio | Generate audio from text | Stable Audio checkpoint |
| 3D | Generate novel views of objects | Zero123 checkpoint, CLIP ViT-L |

## Model Storage

Models are stored in the `models/` directory:

```
models/
├── checkpoints/      # Main model files (.safetensors, .ckpt)
├── vae/              # VAE models
├── clip_vision/      # CLIP vision encoders
├── text_encoders/    # Text encoders (T5, etc.)
├── controlnet/       # ControlNet models
├── loras/            # LoRA adapters
└── upscale_models/   # Upscaling models
```

## Requirements

- Python 3.10 or later
- GTK4 and libadwaita
- PyTorch with CUDA support
- NVIDIA GPU with 8GB+ VRAM recommended

## Configuration

The application auto-detects paths. To override:

- Set `COMFYUI_PATH` environment variable to use an external ComfyUI installation
- Models are stored in `./models/` relative to the application root

## Troubleshooting

### Missing models error

Open the model download dialog and download the required models for your workflow.

### VRAM out of memory

- Close other GPU applications
- Use smaller image dimensions
- Try a different checkpoint (some use less VRAM)

### Custom nodes not loading

Place custom node folders in the `custom_nodes/` directory and restart the application.

## License

MIT License. See LICENSE file for details.

## Credits

- ComfyUI by comfyanonymous
- Stable Diffusion by Stability AI
- GTK4 and libadwaita by GNOME
