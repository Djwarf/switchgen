# SwitchGen

A Linux-native image and video generator using ComfyUI as an embedded library, with SwitchSides branding.

## Prerequisites

- Python 3.10+
- GTK4 and libadwaita
- PyTorch with CUDA support
- ComfyUI (already installed at `/mnt/storage/repos/ComfyUI`)

## Installation

### 1. Install System Dependencies (Arch Linux)

```bash
sudo pacman -S gtk4 libadwaita python-gobject
```

### 2. Create Python Environment

```bash
cd /mnt/storage/repos/switchgen

# Create virtual environment
python -m venv .venv
source .venv/bin/activate

# Install PyTorch (adjust for your CUDA version)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

# Install SwitchGen dependencies
pip install -e .
```

### 3. Configure ComfyUI Path

The default configuration expects ComfyUI at `/mnt/storage/repos/ComfyUI`. If your installation is elsewhere, update `src/switchgen/core/config.py`.

## Usage

### Run the GUI

```bash
source .venv/bin/activate
python -m switchgen
```

### Test Headless Mode

```bash
source .venv/bin/activate
python -m switchgen --test
```

## Project Structure

```
switchgen/
├── src/switchgen/
│   ├── core/
│   │   ├── comfy_init.py     # ComfyUI 4-subsystem initialization
│   │   ├── engine.py         # Generation engine with VRAM cleanup
│   │   ├── queue.py          # Job queuing system
│   │   ├── workflows.py      # Workflow builder
│   │   └── config.py         # Configuration
│   │
│   ├── ui/
│   │   ├── main_window.py    # GTK4 main window
│   │   └── styles/           # SwitchSides CSS theme
│   │
│   └── resources/
│       └── fonts/            # Crimson Text font
│
├── workflows/                # API-format workflow templates
├── output/                   # Generated images
└── temp/                     # Temporary files
```

## ComfyUI Integration

SwitchGen uses ComfyUI as a library by initializing 4 subsystems:

1. **Path Management** (`folder_paths`) - Configure model directories
2. **Node Registry** (`nodes`) - Load core and custom nodes
3. **Execution Engine** (`PromptExecutor`) - Execute workflows via MockServer
4. **Memory Manager** (`model_management`) - GPU memory handling

### Key Features

- **VRAM Cleanup**: Automatic cleanup after each generation prevents memory leaks
- **Interrupt Handling**: Graceful Ctrl+C support during generation
- **Job Queue**: Queue multiple generation requests
- **Sunshine Support**: Reserves VRAM for streaming encoder

## Branding

SwitchGen uses the SwitchSides brand design:

- **Primary Color**: Deep burgundy `#2e0000`
- **Background**: Newsprint `#FAFAF8`
- **Font**: Crimson Text (serif)
- **Style**: Classical newspaper aesthetic with sharp corners

## License

MIT
