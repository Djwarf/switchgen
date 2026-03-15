# Changelog

## [0.2.0] - 2026-03-15

### Fixed
- Priority queue now respects job priority ordering (was using FIFO Queue instead of PriorityQueue)
- Race condition in job cancellation (status transition now atomic under lock)
- Captured images collision when multiple jobs use same capture_id (now uses unique prompt_id)
- Unbounded job history memory leak (auto-purges beyond configurable limit)
- Interrupt handling now clears captured images to prevent stale data
- Silent failures in async download callbacks now logged
- Version mismatch between __init__.py and pyproject.toml

### Added
- ruff and mypy configuration in pyproject.toml
- GitHub Actions CI (lint, typecheck, test)
- This changelog

## [0.1.1] - 2025-01-17

### Fixed
- AUR packaging: use tar pipe instead of cp/rsync for fakeroot compatibility
- Model downloader: use requests instead of deprecated tqdm_class

## [0.1.0] - 2025-01-10

### Added
- Initial release
- GTK4/Libadwaita GUI with ComfyUI as headless library
- Text2Img, Img2Img, Inpaint, Audio, and 3D workflow support
- Model download manager with HuggingFace Hub integration
- In-memory image capture via ReturnToApp custom node
- VRAM monitoring and display
- Job queue with priority support and cancellation
- AUR packaging (PKGBUILD)
