"""Model download manager using HuggingFace Hub."""

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass, field

from tqdm.auto import tqdm as tqdm_auto

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_download, HfApi
    from huggingface_hub.utils import EntryNotFoundError
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False
    logger.debug("huggingface_hub not available")

from .models import ModelInfo, ModelType, MODEL_CATALOG, is_model_installed


class DownloadCancelledException(Exception):
    """Raised when a download is cancelled by the user."""
    pass


@dataclass
class DownloadProgress:
    """Progress information for a download."""
    model_id: str
    downloaded_bytes: int
    total_bytes: int
    speed_bps: float  # bytes per second

    @property
    def progress(self) -> float:
        """Progress as a fraction (0.0 to 1.0)."""
        if self.total_bytes <= 0:
            return 0.0
        return min(1.0, self.downloaded_bytes / self.total_bytes)

    @property
    def downloaded_mb(self) -> float:
        return self.downloaded_bytes / (1024 * 1024)

    @property
    def total_mb(self) -> float:
        return self.total_bytes / (1024 * 1024)

    @property
    def speed_mbps(self) -> float:
        """Speed in MB/s."""
        return self.speed_bps / (1024 * 1024)

    @property
    def eta_seconds(self) -> Optional[float]:
        """Estimated time remaining in seconds."""
        if self.speed_bps <= 0:
            return None
        remaining_bytes = self.total_bytes - self.downloaded_bytes
        if remaining_bytes <= 0:
            return 0.0
        return remaining_bytes / self.speed_bps

    @property
    def eta_formatted(self) -> str:
        """Human-readable ETA string."""
        eta = self.eta_seconds
        if eta is None:
            return "calculating..."
        if eta <= 0:
            return "done"
        if eta < 60:
            return f"{int(eta)}s"
        elif eta < 3600:
            return f"{int(eta // 60)}m {int(eta % 60)}s"
        else:
            hours = int(eta // 3600)
            minutes = int((eta % 3600) // 60)
            return f"{hours}h {minutes}m"


@dataclass
class DownloadResult:
    """Result of a download operation."""
    success: bool
    model_id: str
    path: Optional[Path] = None
    error: Optional[str] = None


class DownloadProgressTqdm(tqdm_auto):
    """Custom tqdm that reports progress via callback and supports cancellation."""

    def __init__(self, *args, progress_callback=None, model_id=None,
                 cancel_check=None, **kwargs):
        self._progress_callback = progress_callback
        self._model_id = model_id
        self._cancel_check = cancel_check
        self._start_time = time.time()
        self._last_update_time = self._start_time
        self._last_bytes = 0
        self._smoothed_speed = 0.0
        super().__init__(*args, **kwargs)

    def update(self, n=1):
        """Override update to call progress callback and check cancellation."""
        # Check for cancellation
        if self._cancel_check and self._cancel_check():
            raise DownloadCancelledException("Download cancelled by user")

        super().update(n)

        if self._progress_callback and self.total:
            current_time = time.time()
            elapsed = current_time - self._last_update_time

            # Calculate speed (smoothed with exponential moving average)
            if elapsed >= 0.25:  # Update speed every 250ms
                bytes_delta = self.n - self._last_bytes
                instant_speed = bytes_delta / elapsed if elapsed > 0 else 0

                # Smooth the speed with EMA (alpha=0.3 for responsiveness)
                if self._smoothed_speed == 0:
                    self._smoothed_speed = instant_speed
                else:
                    self._smoothed_speed = 0.3 * instant_speed + 0.7 * self._smoothed_speed

                self._last_update_time = current_time
                self._last_bytes = self.n

                progress = DownloadProgress(
                    model_id=self._model_id,
                    downloaded_bytes=self.n,
                    total_bytes=self.total,
                    speed_bps=self._smoothed_speed,
                )
                self._progress_callback(progress)


class ModelDownloader:
    """Download manager for HuggingFace models."""

    def __init__(self, models_dir: Path):
        self.models_dir = models_dir
        self._current_download: Optional[str] = None
        self._cancel_requested = False

    def is_available(self) -> bool:
        """Check if HuggingFace Hub is available."""
        return HF_AVAILABLE

    def get_disk_space_mb(self) -> tuple[float, float]:
        """Get free and total disk space in MB."""
        stat = shutil.disk_usage(self.models_dir)
        free_mb = stat.free / (1024 * 1024)
        total_mb = stat.total / (1024 * 1024)
        return free_mb, total_mb

    def check_disk_space(self, size_mb: int) -> bool:
        """Check if enough disk space is available."""
        free_mb, _ = self.get_disk_space_mb()
        # Require 500MB extra headroom
        return free_mb >= (size_mb + 500)

    def get_installed_models(self) -> list[str]:
        """Get list of installed model IDs from the catalog."""
        installed = []
        for model_id, model_info in MODEL_CATALOG.items():
            if is_model_installed(model_info, self.models_dir):
                installed.append(model_id)
        return installed

    def download(
        self,
        model_info: ModelInfo,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
    ) -> DownloadResult:
        """Download a model from HuggingFace Hub.

        Args:
            model_info: The model to download
            progress_callback: Called with progress updates

        Returns:
            DownloadResult with success status and path or error
        """
        if not HF_AVAILABLE:
            logger.error("huggingface_hub not installed")
            return DownloadResult(
                success=False,
                model_id=model_info.id,
                error="huggingface_hub not installed. Run: pip install huggingface_hub"
            )

        # Check disk space
        if not self.check_disk_space(model_info.size_mb):
            free_mb, _ = self.get_disk_space_mb()
            logger.error("Not enough disk space for %s: need %dMB, have %.0fMB",
                        model_info.name, model_info.size_mb, free_mb)
            return DownloadResult(
                success=False,
                model_id=model_info.id,
                error=f"Not enough disk space. Need {model_info.size_mb}MB, have {free_mb:.0f}MB free"
            )

        # Prepare target directory
        target_dir = self.models_dir / model_info.type.value
        target_dir.mkdir(parents=True, exist_ok=True)

        local_filename = model_info.get_local_filename()
        target_path = target_dir / local_filename

        self._current_download = model_info.id
        self._cancel_requested = False

        logger.info("Starting download: %s (%dMB) from %s",
                   model_info.name, model_info.size_mb, model_info.repo_id)

        # Create tqdm class factory with bound parameters
        downloader_self = self

        def create_tqdm_class():
            class BoundProgressTqdm(DownloadProgressTqdm):
                def __init__(inner_self, *args, **kwargs):
                    super().__init__(
                        *args,
                        progress_callback=progress_callback,
                        model_id=model_info.id,
                        cancel_check=lambda: downloader_self._cancel_requested,
                        **kwargs
                    )
            return BoundProgressTqdm

        try:
            # Download using huggingface_hub with custom tqdm
            downloaded_path = hf_hub_download(
                repo_id=model_info.repo_id,
                filename=model_info.filename,
                local_dir=target_dir,
                local_dir_use_symlinks=False,
                tqdm_class=create_tqdm_class(),
            )

            downloaded_path = Path(downloaded_path)

            # Rename if needed
            if downloaded_path.name != local_filename:
                if target_path.exists():
                    target_path.unlink()
                downloaded_path.rename(target_path)
                downloaded_path = target_path

            # Final progress update
            if progress_callback:
                file_size = downloaded_path.stat().st_size
                progress_callback(DownloadProgress(
                    model_id=model_info.id,
                    downloaded_bytes=file_size,
                    total_bytes=file_size,
                    speed_bps=0,
                ))

            logger.info("Download completed: %s -> %s", model_info.name, downloaded_path)

            return DownloadResult(
                success=True,
                model_id=model_info.id,
                path=downloaded_path,
            )

        except DownloadCancelledException:
            logger.info("Download cancelled: %s", model_info.name)
            return DownloadResult(
                success=False,
                model_id=model_info.id,
                error="Download cancelled"
            )
        except EntryNotFoundError:
            logger.error("Model file not found: %s/%s", model_info.repo_id, model_info.filename)
            return DownloadResult(
                success=False,
                model_id=model_info.id,
                error=f"Model not found on HuggingFace. It may have been moved or removed."
            )
        except Exception as e:
            logger.error("Download failed for %s: %s", model_info.name, e, exc_info=True)
            error_msg = str(e)
            # Provide helpful guidance for common errors
            if "401" in error_msg or "unauthorized" in error_msg.lower():
                error_msg = "Authentication required. This model may need a HuggingFace account."
            elif "403" in error_msg or "forbidden" in error_msg.lower():
                error_msg = "Access denied. You may need to accept the model's license on HuggingFace."
            elif "connection" in error_msg.lower() or "timeout" in error_msg.lower():
                error_msg = "Network error. Check your internet connection and try again."
            elif "resolve" in error_msg.lower() or "dns" in error_msg.lower():
                error_msg = "Could not connect to HuggingFace. Check your internet connection."
            return DownloadResult(
                success=False,
                model_id=model_info.id,
                error=error_msg
            )
        finally:
            self._current_download = None

    def download_async(
        self,
        model_info: ModelInfo,
        progress_callback: Optional[Callable[[DownloadProgress], None]] = None,
        complete_callback: Optional[Callable[[DownloadResult], None]] = None,
    ) -> threading.Thread:
        """Download a model asynchronously.

        Args:
            model_info: The model to download
            progress_callback: Called with progress updates (from download thread)
            complete_callback: Called when download completes (from download thread)

        Returns:
            The download thread
        """
        def run():
            result = self.download(model_info, progress_callback)
            if complete_callback:
                complete_callback(result)

        thread = threading.Thread(target=run, daemon=True)
        thread.start()
        return thread

    def cancel_download(self):
        """Request cancellation of current download."""
        self._cancel_requested = True

    @property
    def is_downloading(self) -> bool:
        """Check if a download is in progress."""
        return self._current_download is not None

    @property
    def current_download_id(self) -> Optional[str]:
        """Get the ID of the model currently being downloaded."""
        return self._current_download
