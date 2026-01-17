"""Unit tests for switchgen.core.downloader module."""

import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock


class TestDownloadProgress:
    """Tests for DownloadProgress dataclass."""

    def test_progress_calculation(self):
        """progress should be downloaded/total."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=500,
            total_bytes=1000,
            speed_bps=100.0,
        )

        assert progress.progress == 0.5

    def test_progress_with_zero_total(self):
        """progress should be 0.0 when total is zero."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=500,
            total_bytes=0,
            speed_bps=0.0,
        )

        assert progress.progress == 0.0

    def test_progress_caps_at_one(self):
        """progress should not exceed 1.0."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=1500,
            total_bytes=1000,
            speed_bps=100.0,
        )

        assert progress.progress == 1.0

    def test_downloaded_mb(self):
        """downloaded_mb should convert bytes to MB."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=1024 * 1024 * 5,  # 5MB
            total_bytes=1024 * 1024 * 10,
            speed_bps=100.0,
        )

        assert progress.downloaded_mb == 5.0

    def test_total_mb(self):
        """total_mb should convert bytes to MB."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=0,
            total_bytes=1024 * 1024 * 10,  # 10MB
            speed_bps=0.0,
        )

        assert progress.total_mb == 10.0

    def test_speed_mbps(self):
        """speed_mbps should convert bytes/s to MB/s."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=0,
            total_bytes=1000,
            speed_bps=1024 * 1024 * 5,  # 5 MB/s
        )

        assert progress.speed_mbps == 5.0

    def test_eta_seconds_calculation(self):
        """eta_seconds should calculate remaining time correctly."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=500,
            total_bytes=1000,
            speed_bps=100.0,  # 100 bytes/sec, 500 remaining = 5 seconds
        )

        assert progress.eta_seconds == 5.0

    def test_eta_seconds_zero_speed(self):
        """eta_seconds should return None when speed is zero."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=500,
            total_bytes=1000,
            speed_bps=0.0,
        )

        assert progress.eta_seconds is None

    def test_eta_formatted_seconds(self):
        """eta_formatted should show seconds for short times."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=900,
            total_bytes=1000,
            speed_bps=10.0,  # 10 seconds remaining
        )

        assert progress.eta_formatted == "10s"

    def test_eta_formatted_minutes(self):
        """eta_formatted should show minutes and seconds."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=0,
            total_bytes=9000,
            speed_bps=100.0,  # 90 seconds = 1m 30s
        )

        assert progress.eta_formatted == "1m 30s"

    def test_eta_formatted_hours(self):
        """eta_formatted should show hours and minutes for long times."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=0,
            total_bytes=7200,
            speed_bps=1.0,  # 7200 seconds = 2h 0m
        )

        assert progress.eta_formatted == "2h 0m"

    def test_eta_formatted_calculating(self):
        """eta_formatted should show 'calculating...' when speed is zero."""
        from switchgen.core.downloader import DownloadProgress

        progress = DownloadProgress(
            model_id="test",
            downloaded_bytes=0,
            total_bytes=1000,
            speed_bps=0.0,
        )

        assert progress.eta_formatted == "calculating..."


class TestDownloadCancelledException:
    """Tests for DownloadCancelledException."""

    def test_exception_message(self):
        """Should store the provided message."""
        from switchgen.core.downloader import DownloadCancelledException

        exc = DownloadCancelledException("User cancelled")
        assert str(exc) == "User cancelled"


class TestDownloadResult:
    """Tests for DownloadResult dataclass."""

    def test_success_result(self, tmp_path):
        """Should represent successful download."""
        from switchgen.core.downloader import DownloadResult

        result = DownloadResult(
            success=True,
            model_id="test_model",
            path=tmp_path / "model.safetensors",
        )

        assert result.success is True
        assert result.model_id == "test_model"
        assert result.error is None

    def test_failure_result(self):
        """Should represent failed download."""
        from switchgen.core.downloader import DownloadResult

        result = DownloadResult(
            success=False,
            model_id="test_model",
            error="Network error",
        )

        assert result.success is False
        assert result.error == "Network error"
        assert result.path is None


class TestModelDownloader:
    """Tests for ModelDownloader class."""

    def test_is_available(self, tmp_models_dir):
        """is_available should return HF_AVAILABLE status."""
        from switchgen.core.downloader import ModelDownloader, HF_AVAILABLE

        downloader = ModelDownloader(tmp_models_dir)

        assert downloader.is_available() == HF_AVAILABLE

    def test_get_disk_space_mb(self, tmp_models_dir):
        """get_disk_space_mb should return (free, total) tuple."""
        from switchgen.core.downloader import ModelDownloader

        downloader = ModelDownloader(tmp_models_dir)

        free_mb, total_mb = downloader.get_disk_space_mb()

        assert isinstance(free_mb, float)
        assert isinstance(total_mb, float)
        assert free_mb >= 0
        assert total_mb >= free_mb

    def test_check_disk_space_sufficient(self, tmp_models_dir):
        """check_disk_space should return True when space available."""
        from switchgen.core.downloader import ModelDownloader

        downloader = ModelDownloader(tmp_models_dir)

        # 1MB should definitely be available
        assert downloader.check_disk_space(1) is True

    def test_check_disk_space_insufficient(self, tmp_models_dir):
        """check_disk_space should return False when space insufficient."""
        from switchgen.core.downloader import ModelDownloader

        downloader = ModelDownloader(tmp_models_dir)

        # Request impossibly large size
        assert downloader.check_disk_space(10_000_000_000) is False

    def test_get_installed_models_empty(self, tmp_models_dir):
        """get_installed_models should return empty list when no models."""
        from switchgen.core.downloader import ModelDownloader

        downloader = ModelDownloader(tmp_models_dir)

        result = downloader.get_installed_models()

        assert result == []

    def test_get_installed_models_finds_installed(self, tmp_models_dir):
        """get_installed_models should find installed models."""
        from switchgen.core.downloader import ModelDownloader
        from switchgen.core.models import MODEL_CATALOG

        downloader = ModelDownloader(tmp_models_dir)

        # Install a model from catalog
        first_model = next(iter(MODEL_CATALOG.values()))
        model_dir = tmp_models_dir / first_model.type.value
        model_dir.mkdir(parents=True, exist_ok=True)
        (model_dir / first_model.get_local_filename()).touch()

        result = downloader.get_installed_models()

        assert first_model.id in result

    def test_is_downloading_initially_false(self, tmp_models_dir):
        """is_downloading should be False initially."""
        from switchgen.core.downloader import ModelDownloader

        downloader = ModelDownloader(tmp_models_dir)

        assert downloader.is_downloading is False

    def test_current_download_id_initially_none(self, tmp_models_dir):
        """current_download_id should be None initially."""
        from switchgen.core.downloader import ModelDownloader

        downloader = ModelDownloader(tmp_models_dir)

        assert downloader.current_download_id is None

    def test_download_without_hf_hub(self, tmp_models_dir, sample_model_info):
        """download should fail gracefully when huggingface_hub unavailable."""
        from switchgen.core import downloader as downloader_module
        from switchgen.core.downloader import ModelDownloader

        # Mock HF_AVAILABLE as False
        with patch.object(downloader_module, 'HF_AVAILABLE', False):
            dl = ModelDownloader(tmp_models_dir)
            result = dl.download(sample_model_info)

        assert result.success is False
        assert "huggingface_hub not installed" in result.error

    def test_download_checks_disk_space(self, tmp_models_dir, sample_model_info):
        """download should check disk space before downloading."""
        from switchgen.core.downloader import ModelDownloader

        downloader = ModelDownloader(tmp_models_dir)

        # Request huge file
        sample_model_info.size_mb = 10_000_000_000

        result = downloader.download(sample_model_info)

        assert result.success is False
        assert "disk space" in result.error.lower()


class TestModelDownloaderAsync:
    """Tests for async download functionality."""

    def test_download_async_returns_thread(self, tmp_models_dir, sample_model_info):
        """download_async should return a Thread object."""
        from switchgen.core.downloader import ModelDownloader
        import threading

        downloader = ModelDownloader(tmp_models_dir)

        # Mock the download to complete quickly
        with patch.object(downloader, 'download') as mock_download:
            mock_download.return_value = MagicMock(success=True)

            thread = downloader.download_async(sample_model_info)

            assert isinstance(thread, threading.Thread)
            thread.join(timeout=1.0)

    def test_download_async_calls_complete_callback(self, tmp_models_dir, sample_model_info):
        """download_async should call complete_callback when done."""
        from switchgen.core.downloader import ModelDownloader, DownloadResult

        downloader = ModelDownloader(tmp_models_dir)
        callback_called = []

        def on_complete(result):
            callback_called.append(result)

        with patch.object(downloader, 'download') as mock_download:
            mock_result = DownloadResult(success=True, model_id="test")
            mock_download.return_value = mock_result

            thread = downloader.download_async(
                sample_model_info,
                complete_callback=on_complete,
            )
            thread.join(timeout=1.0)

        assert len(callback_called) == 1
        assert callback_called[0] == mock_result

    def test_cancel_download_sets_flag(self, tmp_models_dir):
        """cancel_download should set the cancel flag."""
        from switchgen.core.downloader import ModelDownloader

        downloader = ModelDownloader(tmp_models_dir)

        downloader.cancel_download()

        assert downloader._cancel_requested is True
