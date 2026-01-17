"""Model download manager using HuggingFace Hub."""

import logging
import shutil
import threading
import time
from pathlib import Path
from typing import Callable, Optional
from dataclasses import dataclass

import requests

logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_url, HfApi
    from huggingface_hub.utils import EntryNotFoundError, build_hf_headers
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

        # Use temp file for download, then rename on success
        temp_path = target_path.with_suffix('.tmp')

        try:
            # Get the download URL from HuggingFace
            url = hf_hub_url(
                repo_id=model_info.repo_id,
                filename=model_info.filename,
            )

            # Get headers (for authentication if needed)
            headers = build_hf_headers()

            # Start download with streaming
            # Use tuple timeout: (connect_timeout, read_timeout)
            # Read timeout is per-chunk, not total, so 60s is plenty
            response = requests.get(url, headers=headers, stream=True, timeout=(30, 60), allow_redirects=True)
            response.raise_for_status()

            # Get total size from headers
            total_size = int(response.headers.get('content-length', 0))
            if total_size == 0:
                total_size = model_info.size_mb * 1024 * 1024  # Fallback to catalog size

            # Track progress
            downloaded = 0
            start_time = time.time()
            last_update_time = start_time
            last_bytes = 0
            smoothed_speed = 0.0
            chunk_size = 1024 * 1024  # 1MB chunks for faster downloads

            logger.info("Download starting: %s (%.1f MB)", model_info.name, total_size / (1024*1024))

            # Initial progress callback
            if progress_callback:
                progress_callback(DownloadProgress(
                    model_id=model_info.id,
                    downloaded_bytes=0,
                    total_bytes=total_size,
                    speed_bps=0,
                ))

            with open(temp_path, 'wb') as f:
                for chunk in response.iter_content(chunk_size=chunk_size):
                    # Check for cancellation
                    if self._cancel_requested:
                        raise DownloadCancelledException("Download cancelled by user")

                    if chunk:
                        f.write(chunk)
                        downloaded += len(chunk)

                        # Update progress every 250ms
                        current_time = time.time()
                        elapsed = current_time - last_update_time

                        if elapsed >= 0.25 and progress_callback:
                            bytes_delta = downloaded - last_bytes
                            instant_speed = bytes_delta / elapsed if elapsed > 0 else 0

                            # Smooth speed with EMA
                            if smoothed_speed == 0:
                                smoothed_speed = instant_speed
                            else:
                                smoothed_speed = 0.3 * instant_speed + 0.7 * smoothed_speed

                            last_update_time = current_time
                            last_bytes = downloaded

                            progress_callback(DownloadProgress(
                                model_id=model_info.id,
                                downloaded_bytes=downloaded,
                                total_bytes=total_size,
                                speed_bps=smoothed_speed,
                            ))

            # Verify download completed
            if not temp_path.exists():
                raise RuntimeError(f"Download failed: temp file not created at {temp_path}")

            actual_size = temp_path.stat().st_size
            if actual_size == 0:
                temp_path.unlink()
                raise RuntimeError("Download failed: file is empty")

            if total_size > 0 and actual_size < total_size * 0.99:  # Allow 1% tolerance
                temp_path.unlink()
                raise RuntimeError(f"Download incomplete: got {actual_size} bytes, expected {total_size}")

            # Rename temp file to final path
            if target_path.exists():
                target_path.unlink()
            temp_path.rename(target_path)

            # Final progress update
            if progress_callback:
                progress_callback(DownloadProgress(
                    model_id=model_info.id,
                    downloaded_bytes=total_size,
                    total_bytes=total_size,
                    speed_bps=0,
                ))

            logger.info("Download completed: %s -> %s", model_info.name, target_path)

            return DownloadResult(
                success=True,
                model_id=model_info.id,
                path=target_path,
            )

        except DownloadCancelledException:
            logger.info("Download cancelled: %s", model_info.name)
            # Clean up temp file
            if temp_path.exists():
                temp_path.unlink()
            return DownloadResult(
                success=False,
                model_id=model_info.id,
                error="Download cancelled"
            )
        except requests.exceptions.HTTPError as e:
            logger.error("HTTP error downloading %s: %s", model_info.name, e)
            if temp_path.exists():
                temp_path.unlink()
            status_code = e.response.status_code if e.response else 0
            if status_code == 401:
                error_msg = "Authentication required. This model may need a HuggingFace account."
            elif status_code == 403:
                error_msg = "Access denied. You may need to accept the model's license on HuggingFace."
            elif status_code == 404:
                error_msg = "Model not found on HuggingFace. It may have been moved or removed."
            else:
                error_msg = f"HTTP error {status_code}: {str(e)}"
            return DownloadResult(
                success=False,
                model_id=model_info.id,
                error=error_msg
            )
        except requests.exceptions.ConnectionError:
            logger.error("Connection error downloading %s", model_info.name)
            if temp_path.exists():
                temp_path.unlink()
            return DownloadResult(
                success=False,
                model_id=model_info.id,
                error="Network error. Check your internet connection and try again."
            )
        except requests.exceptions.Timeout:
            logger.error("Timeout downloading %s", model_info.name)
            if temp_path.exists():
                temp_path.unlink()
            return DownloadResult(
                success=False,
                model_id=model_info.id,
                error="Download timed out. Try again later."
            )
        except Exception as e:
            logger.error("Download failed for %s: %s", model_info.name, e, exc_info=True)
            if temp_path.exists():
                temp_path.unlink()
            return DownloadResult(
                success=False,
                model_id=model_info.id,
                error=str(e)
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
