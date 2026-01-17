"""Unit tests for switchgen.core.engine module."""

import pytest
from unittest.mock import MagicMock, patch
import numpy as np


class TestGenerationResult:
    """Tests for GenerationResult dataclass."""

    def test_success_result(self):
        """Should represent successful generation."""
        from switchgen.core.engine import GenerationResult

        result = GenerationResult(
            prompt_id="test-123",
            success=True,
            outputs={"images": []},
        )

        assert result.success is True
        assert result.error is None

    def test_failure_result(self):
        """Should represent failed generation."""
        from switchgen.core.engine import GenerationResult

        result = GenerationResult(
            prompt_id="test-123",
            success=False,
            outputs={},
            error="Out of memory",
        )

        assert result.success is False
        assert result.error == "Out of memory"


class TestProgressInfo:
    """Tests for ProgressInfo class."""

    def test_initial_values(self):
        """Should have zero initial values."""
        from switchgen.core.engine import ProgressInfo

        info = ProgressInfo()

        assert info.current_step == 0
        assert info.total_steps == 0
        assert info.current_node == ""
        assert info.preview_image is None

    def test_update(self):
        """update should set all values."""
        from switchgen.core.engine import ProgressInfo

        info = ProgressInfo()
        info.update(5, 20, "KSampler", "preview_data")

        assert info.current_step == 5
        assert info.total_steps == 20
        assert info.current_node == "KSampler"
        assert info.preview_image == "preview_data"

    def test_update_partial(self):
        """update should work with partial values."""
        from switchgen.core.engine import ProgressInfo

        info = ProgressInfo()
        info.update(10, 30)

        assert info.current_step == 10
        assert info.total_steps == 30
        assert info.current_node == ""
        assert info.preview_image is None


class TestMockServer:
    """Tests for MockServer class."""

    def test_client_id(self):
        """Should have switchgen client ID."""
        from switchgen.core.engine import MockServer

        server = MockServer()

        assert server.client_id == "switchgen_client"

    def test_send_sync_progress_event(self):
        """send_sync should handle progress events."""
        from switchgen.core.engine import MockServer

        callback_called = []

        def callback(progress):
            callback_called.append(progress)

        server = MockServer(progress_callback=callback)
        server.send_sync("progress", {"value": 5, "max": 20})

        assert len(callback_called) == 1
        assert callback_called[0].current_step == 5
        assert callback_called[0].total_steps == 20

    def test_send_sync_executing_event(self):
        """send_sync should handle executing events."""
        from switchgen.core.engine import MockServer

        server = MockServer()
        server.send_sync("executing", {"node": "KSampler"})

        assert server.last_node_id == "KSampler"
        assert server.progress.current_node == "KSampler"

    def test_send_sync_without_callback(self):
        """send_sync should work without callback."""
        from switchgen.core.engine import MockServer

        server = MockServer()
        # Should not raise
        server.send_sync("progress", {"value": 1, "max": 10})

    def test_queue_updated(self):
        """queue_updated should be callable."""
        from switchgen.core.engine import MockServer

        server = MockServer()
        # Should not raise
        server.queue_updated()


class TestGenerationEngine:
    """Tests for GenerationEngine class."""

    def test_initial_state(self):
        """Should have correct initial state."""
        from switchgen.core.engine import GenerationEngine

        engine = GenerationEngine()

        assert engine._initialized is False
        assert engine._executor is None
        assert engine._server is None
        assert engine._interrupted is False

    def test_get_vram_usage_uninitialized(self):
        """get_vram_usage should return (0, 0) when uninitialized."""
        from switchgen.core.engine import GenerationEngine

        engine = GenerationEngine()

        used, total = engine.get_vram_usage()

        assert used == 0
        assert total == 0

    def test_get_vram_usage_percent_uninitialized(self):
        """get_vram_usage_percent should return 0.0 when uninitialized."""
        from switchgen.core.engine import GenerationEngine

        engine = GenerationEngine()

        result = engine.get_vram_usage_percent()

        assert result == 0.0

    def test_get_vram_usage_percent_calculation(self):
        """get_vram_usage_percent should calculate correctly."""
        from switchgen.core.engine import GenerationEngine

        engine = GenerationEngine()

        # Mock get_vram_usage to return specific values
        with patch.object(engine, 'get_vram_usage', return_value=(500, 1000)):
            result = engine.get_vram_usage_percent()

        assert result == 50.0

    def test_get_vram_usage_percent_zero_total(self):
        """get_vram_usage_percent should handle zero total."""
        from switchgen.core.engine import GenerationEngine

        engine = GenerationEngine()

        with patch.object(engine, 'get_vram_usage', return_value=(0, 0)):
            result = engine.get_vram_usage_percent()

        assert result == 0.0

    def test_interrupt_sets_flag(self):
        """interrupt should set interrupted flag."""
        from switchgen.core.engine import GenerationEngine

        engine = GenerationEngine()
        engine._memory_manager = MagicMock()

        engine.interrupt()

        assert engine._interrupted is True
        engine._memory_manager.interrupt_current_processing.assert_called_once()

    def test_set_progress_callback(self):
        """set_progress_callback should store callback."""
        from switchgen.core.engine import GenerationEngine
        from switchgen.core import engine as engine_module

        engine = GenerationEngine()
        engine._server = MagicMock()

        callback = MagicMock()

        # Mock the comfy_init functions to avoid ComfyUI dependency
        with patch.object(engine_module, 'set_comfy_progress_callback'):
            engine.set_progress_callback(callback)

        assert engine._progress_callback == callback
        assert engine._server.progress_callback == callback

    def test_clear_progress_callback(self):
        """set_progress_callback(None) should clear callback."""
        from switchgen.core.engine import GenerationEngine
        from switchgen.core import engine as engine_module

        engine = GenerationEngine()
        engine._server = MagicMock()
        engine._progress_callback = MagicMock()

        # Mock the comfy_init functions to avoid ComfyUI dependency
        with patch.object(engine_module, 'clear_progress_callback'):
            engine.set_progress_callback(None)

        assert engine._progress_callback is None

    def test_cleanup_vram(self):
        """cleanup_vram should call memory manager and gc."""
        from switchgen.core.engine import GenerationEngine
        import gc

        engine = GenerationEngine()
        engine._memory_manager = MagicMock()

        with patch.object(gc, 'collect') as mock_gc:
            engine.cleanup_vram()

        engine._memory_manager.soft_empty_cache.assert_called_once()
        mock_gc.assert_called_once()


class TestTensorToPil:
    """Tests for tensor_to_pil function."""

    @pytest.fixture
    def pil_available(self):
        """Check if PIL is available."""
        try:
            from PIL import Image
            return True
        except ImportError:
            return False

    def test_returns_empty_for_none(self, pil_available):
        """Should return empty list for None input."""
        if not pil_available:
            pytest.skip("PIL not available")

        from switchgen.core.engine import tensor_to_pil

        result = tensor_to_pil(None)

        assert result == []

    def test_converts_tensor_to_pil(self, pil_available):
        """Should convert tensor to PIL Images."""
        if not pil_available:
            pytest.skip("PIL not available")

        from switchgen.core.engine import tensor_to_pil
        from PIL import Image

        # Create a mock tensor (B, H, W, C) in [0, 1] range
        tensor = MagicMock()
        tensor.cpu.return_value = tensor
        tensor.numpy.return_value = np.random.rand(2, 64, 64, 3)  # 2 images
        tensor.shape = (2, 64, 64, 3)

        result = tensor_to_pil(tensor)

        assert len(result) == 2
        assert all(isinstance(img, Image.Image) for img in result)

    def test_handles_numpy_array(self, pil_available):
        """Should handle numpy array input."""
        if not pil_available:
            pytest.skip("PIL not available")

        from switchgen.core.engine import tensor_to_pil
        from PIL import Image

        # Create numpy array directly
        array = np.random.rand(1, 32, 32, 3)

        result = tensor_to_pil(array)

        assert len(result) == 1
        assert isinstance(result[0], Image.Image)


class TestGetEngine:
    """Tests for get_engine singleton function."""

    def test_returns_engine_instance(self):
        """Should return a GenerationEngine instance."""
        from switchgen.core import engine as engine_module

        # Reset the global
        original = engine_module._engine
        engine_module._engine = None

        try:
            result = engine_module.get_engine()
            assert isinstance(result, engine_module.GenerationEngine)
        finally:
            engine_module._engine = original

    def test_returns_same_instance(self):
        """Should return the same instance on subsequent calls."""
        from switchgen.core import engine as engine_module

        original = engine_module._engine
        engine_module._engine = None

        try:
            result1 = engine_module.get_engine()
            result2 = engine_module.get_engine()
            assert result1 is result2
        finally:
            engine_module._engine = original
