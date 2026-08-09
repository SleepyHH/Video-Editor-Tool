"""
Automated regression tests for the step-2 media search engine: config.py's
portable-path helpers, media_index.py, media_search.py, media_search_server.py.

Fast, no GPU/model load, no real photo library or index needed - run
before/after any change to any of those four files:

    python3 test_media_search.py

Anything that needs the real CLIP model downloaded/loaded, a real library to
scan, or the actual search GUI in a browser isn't here - see TESTING.md's
media search section for that manual checklist.
"""
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import config
import media_index
import media_search
import media_search_server


class TestPortablePaths(unittest.TestCase):
    """DRIVE::-portable path round-tripping - the mechanism that keeps the
    shared index valid across the Mac's /Volumes/... and the PC's D:\\...
    mount points. See media-search-shared-drive-architecture memory."""

    def test_round_trip_when_drive_connected(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive_root = Path(tmp) / "FakeDrive"
            (drive_root / "sub").mkdir(parents=True)
            real_file = drive_root / "sub" / "clip.mov"
            real_file.write_text("x")

            with patch.object(config, "find_volume_by_label", return_value=drive_root):
                portable = config.to_portable_path(str(real_file), "FakeDrive")
                self.assertEqual(portable, "DRIVE::sub/clip.mov")
                resolved = config.resolve_portable_path(portable, "FakeDrive")
                self.assertEqual(resolved, real_file)

    def test_path_not_on_drive_passes_through_unchanged(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive_root = Path(tmp) / "FakeDrive"
            drive_root.mkdir()
            with patch.object(config, "find_volume_by_label", return_value=drive_root):
                elsewhere = str(Path(tmp) / "not_on_drive.jpg")
                self.assertEqual(config.to_portable_path(elsewhere, "FakeDrive"), elsewhere)

    def test_resolve_returns_none_when_drive_not_connected(self):
        with patch.object(config, "find_volume_by_label", return_value=None):
            self.assertIsNone(config.resolve_portable_path("DRIVE::foo.jpg", "FakeDrive"))

    def test_resolve_local_only_path_passes_through(self):
        # Content that was never on the shared drive (e.g. the Mac's native
        # Photos library) has no DRIVE:: prefix - must resolve to itself.
        local = "/Users/x/Photos/img.heic"
        self.assertEqual(config.resolve_portable_path(local, "FakeDrive"), Path(local))


class TestIndexSelfExclusion(unittest.TestCase):
    def test_search_index_folder_is_ignored(self):
        # Real bug (06/07-08-2026): the drive walk recursed into its own
        # thumbnail cache and indexed the generated thumbnails as library
        # photos - regression guard, see media-search-windows-first-full-index.
        self.assertIn("HuysVideoEditor_SearchIndex", media_index.IGNORED_DIR_NAMES)


class TestWalkMediaFiles(unittest.TestCase):
    """Real bug found live 08-08-2026 against the actual external drive: macOS
    writes an AppleDouble sidecar file ("._IMG_1234.mov") next to every real
    file on exFAT, carrying the same extension as the real one - an
    extension-only filter lets these through as if they were real media."""

    def test_skips_appledouble_sidecar_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "IMG_1234.mov").write_text("real video bytes")
            (root / "._IMG_1234.mov").write_text("resource fork stub")
            names = sorted(p.name for p in config.walk_media_files(root))
            self.assertEqual(names, ["IMG_1234.mov"])

    def test_skips_ignored_and_hidden_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "System Volume Information").mkdir()
            (root / "System Volume Information" / "skip.mov").write_text("x")
            (root / ".hidden").mkdir()
            (root / ".hidden" / "skip2.mov").write_text("x")
            (root / "keep.mov").write_text("x")
            names = [p.name for p in config.walk_media_files(root)]
            self.assertEqual(names, ["keep.mov"])


class TestFileTypeFiltering(unittest.TestCase):
    def test_video_shorthand_expands_to_video_extensions(self):
        types = media_search.resolve_file_types("video")
        self.assertEqual(types, media_index.VIDEO_EXTENSIONS)

    def test_image_shorthand_expands_to_image_extensions(self):
        types = media_search.resolve_file_types("image")
        self.assertEqual(types, media_index.IMAGE_EXTENSIONS)

    def test_specific_extension_passthrough(self):
        self.assertEqual(media_search.resolve_file_types("mov"), {".mov"})
        self.assertEqual(media_search.resolve_file_types(".mov"), {".mov"})

    def test_comma_separated_mixes_types(self):
        types = media_search.resolve_file_types("mov,png")
        self.assertEqual(types, {".mov", ".png"})

    def test_none_or_empty_means_no_filter(self):
        self.assertIsNone(media_search.resolve_file_types(None))
        self.assertIsNone(media_search.resolve_file_types(""))


class TestDeviceSelection(unittest.TestCase):
    def test_returns_a_supported_backend(self):
        self.assertIn(media_index.get_device(), ("mps", "cuda", "cpu"))


class TestClipModelCache(unittest.TestCase):
    def test_unload_before_load_is_a_safe_noop(self):
        media_index._model_cache.clear()
        media_index.unload_clip_model()  # must not raise
        self.assertNotIn("model", media_index._model_cache)


class TestClipIdleUnloadTimer(unittest.TestCase):
    """Doesn't wait through the real 10-minute production window - swaps in a
    near-zero interval and checks the timer mechanics themselves (schedule,
    cancel-and-reschedule, actually fires), see media_search_server.py."""

    def tearDown(self):
        if media_search_server._unload_timer is not None:
            media_search_server._unload_timer.cancel()

    def test_reschedule_cancels_the_previous_timer(self):
        with patch.object(media_search_server, "CLIP_IDLE_UNLOAD_SECONDS", 999):
            media_search_server._schedule_clip_unload()
            first_timer = media_search_server._unload_timer
            media_search_server._schedule_clip_unload()
            second_timer = media_search_server._unload_timer
        self.assertIsNot(first_timer, second_timer)
        self.assertFalse(first_timer.is_alive())

    def test_timer_fires_and_calls_unload(self):
        with patch.object(media_search_server, "CLIP_IDLE_UNLOAD_SECONDS", 0.05), \
             patch.object(media_search_server, "unload_clip_model") as mock_unload:
            media_search_server._schedule_clip_unload()
            time.sleep(0.2)
        mock_unload.assert_called_once()


if __name__ == "__main__":
    unittest.main(verbosity=2)
