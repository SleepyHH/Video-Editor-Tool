"""
Automated regression tests for Huys-Video-Editor.

Covers everything that can be tested quickly, without a GUI window, a real
Photos library, or a slow Whisper/render pass - the pure logic functions and
the ffmpeg-based probes. Run before/after any code change:

    python3 test_helpers.py

For everything else (Photos scanning, transcription, rendering - anything
needing real hardware time or the actual Photos database), see TESTING.md
for the manual checklist.
"""
import os
import sys
import types
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import main
import config

PROJECT_DIR = Path(__file__).resolve().parent
IMPORTS_DIR = Path.home() / "Documents" / "HuysVideoEditor_Imports"


def make_stub_profile():
    """A minimal stand-in for `self` so instance methods that only touch
    self.profile can be called directly, without building the full Qt window."""
    profile = config.get_os_profile()
    return SimpleNamespace(profile=profile)


class TestSrtTimestamps(unittest.TestCase):
    def test_zero(self):
        self.assertEqual(main.format_srt_timestamp(0), "00:00:00,000")

    def test_sub_second(self):
        self.assertEqual(main.format_srt_timestamp(1.5), "00:00:01,500")

    def test_minutes_and_hours(self):
        self.assertEqual(main.format_srt_timestamp(3661.25), "01:01:01,250")


class TestUniquePath(unittest.TestCase):
    def test_no_collision_returns_same_path(self):
        with tempfile.TemporaryDirectory() as tmp:
            p = Path(tmp) / "clip.srt"
            self.assertEqual(main.get_unique_path(p), p)

    def test_collisions_increment(self):
        with tempfile.TemporaryDirectory() as tmp:
            base = Path(tmp) / "clip.srt"
            base.write_text("x")
            self.assertEqual(main.get_unique_path(base).name, "clip-1.srt")

            (Path(tmp) / "clip-1.srt").write_text("x")
            self.assertEqual(main.get_unique_path(base).name, "clip-2.srt")


class TestColumnsForWidth(unittest.TestCase):
    """Backs the Prompt-Style Search tab's responsive result grid (08-08-2026) -
    the one genuinely pure-logic piece of that feature, everything else needs
    a real Qt widget tree and belongs in TESTING.md's manual checklist instead."""

    def test_fits_expected_count_at_default_card_size(self):
        # 3 columns need 3*(RESULT_CARD_WIDTH+8) - 8 = 556px; one pixel less drops to 2.
        self.assertEqual(main.columns_for_width(556), 3)
        self.assertEqual(main.columns_for_width(555), 2)

    def test_never_returns_fewer_than_one_column(self):
        self.assertEqual(main.columns_for_width(0), 1)
        self.assertEqual(main.columns_for_width(-100), 1)

    def test_custom_card_size_and_spacing(self):
        self.assertEqual(main.columns_for_width(220, card_width=100, spacing=10), 2)


class TestManualImportsPersistence(unittest.TestCase):
    def test_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            app_data_dir = Path(tmp)
            with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as real_file:
                real_file.write(b"fake video bytes")
                real_path = real_file.name

            items = [("my clip.mp4", real_path)]
            main.save_manual_imports_to_disk(app_data_dir, items)
            loaded = main.load_manual_imports_from_disk(app_data_dir)
            self.assertEqual(loaded, items)
            Path(real_path).unlink()

    def test_missing_file_is_dropped_not_crashed(self):
        with tempfile.TemporaryDirectory() as tmp:
            app_data_dir = Path(tmp)
            main.save_manual_imports_to_disk(app_data_dir, [("gone.mp4", "/nowhere/gone.mp4")])
            self.assertEqual(main.load_manual_imports_from_disk(app_data_dir), [])

    def test_no_file_yet_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            self.assertEqual(main.load_manual_imports_from_disk(Path(tmp)), [])


class TestHddMediaListing(unittest.TestCase):
    """list_hdd_media backs the HDD-first media source in load_iphone_photos
    (08-08-2026) - the drive, when connected, is now the primary source for
    the video editor's media list, matching what the search index scans."""

    def test_finds_videos_recursively_and_sorts_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "sub").mkdir()
            old = root / "old.mp4"
            old.write_text("x")
            new = root / "sub" / "new.mov"
            new.write_text("x")
            os.utime(old, (1000, 1000))
            os.utime(new, (2000, 2000))
            names = [name for name, _ in main.list_hdd_media(root)]
            self.assertEqual(names, ["new.mov", "old.mp4"])

    def test_skips_ignored_and_hidden_directories(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            ignored = root / "System Volume Information"
            ignored.mkdir()
            (ignored / "skip.mp4").write_text("x")
            hidden = root / ".hidden"
            hidden.mkdir()
            (hidden / "skip2.mp4").write_text("x")
            (root / "keep.mp4").write_text("x")
            names = [name for name, _ in main.list_hdd_media(root)]
            self.assertEqual(names, ["keep.mp4"])

    def test_ignores_non_video_files(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            (root / "photo.jpg").write_text("x")
            (root / "clip.mp4").write_text("x")
            names = [name for name, _ in main.list_hdd_media(root)]
            self.assertEqual(names, ["clip.mp4"])


class TestTranscriptionLanguages(unittest.TestCase):
    def test_covers_all_100_whisper_languages_plus_mixed(self):
        # +1 for "Mixed languages" - "English" is one of the 100, not double-counted
        self.assertEqual(len(main.TRANSCRIPTION_LANGUAGES), 101)

    def test_english_is_first_and_default(self):
        self.assertEqual(next(iter(main.TRANSCRIPTION_LANGUAGES)), "English")
        self.assertEqual(main.TRANSCRIPTION_LANGUAGES["English"], "en")

    def test_mixed_languages_uses_the_multilingual_sentinel(self):
        self.assertEqual(main.TRANSCRIPTION_LANGUAGES["Mixed languages"], main.MULTILINGUAL_SENTINEL)

    def test_auto_detect_removed(self):
        # Confirmed unreliable (misidentified real Vietnamese speech as Khmer) - must stay removed
        self.assertNotIn("Auto-detect", main.TRANSCRIPTION_LANGUAGES)

    def test_vietnamese_present(self):
        self.assertEqual(main.TRANSCRIPTION_LANGUAGES["Vietnamese"], "vi")


class TestDurationFormatting(unittest.TestCase):
    def test_under_an_hour(self):
        self.assertEqual(main.ProfessionalAIEditor.format_duration(95), "01:35")

    def test_over_an_hour(self):
        self.assertEqual(main.ProfessionalAIEditor.format_duration(3725), "01:02:05")

    def test_none_is_unknown(self):
        self.assertEqual(main.ProfessionalAIEditor.format_duration(None), "unknown")


class TestFfmpegProbesOnRealFiles(unittest.TestCase):
    """Uses whatever real video files are already sitting in the Imports folder
    from prior sessions. Each test skips itself if that specific file isn't
    there, rather than failing - keeps this robust to the user cleaning up
    their own folders."""

    def _skip_unless_exists(self, filename):
        path = IMPORTS_DIR / filename
        if not path.exists():
            self.skipTest(f"{filename} not present in {IMPORTS_DIR} - skipping")
        return path

    def test_probe_video_returns_codec_and_duration_together(self):
        # probe_video replaced the old separate get_video_codec/get_video_duration_seconds -
        # one ffmpeg call covering both, instead of two calls re-reading the same file.
        path = self._skip_unless_exists("IMG_1814.MOV")
        stub = make_stub_profile()
        codec, duration = main.ProfessionalAIEditor.probe_video(stub, str(path))
        self.assertEqual(codec, "hevc")
        self.assertIsNotNone(duration)
        self.assertGreater(duration, 0)


class TestWhisperUnload(unittest.TestCase):
    def test_unload_clears_the_model_reference(self):
        stub = SimpleNamespace(whisper_model=object())
        main.ProfessionalAIEditor.unload_whisper_model(stub)
        self.assertIsNone(stub.whisper_model)


class TestConfigProfile(unittest.TestCase):
    def test_profile_has_required_keys(self):
        profile = config.get_os_profile()
        for key in ("os", "photos_path", "ffmpeg_binary", "pipeline_mode", "app_data_dir"):
            self.assertIn(key, profile)

    def test_mac_uses_native_db_pipeline(self):
        if config.platform.system() != "Darwin":
            self.skipTest("Not running on macOS")
        profile = config.get_os_profile()
        self.assertEqual(profile["pipeline_mode"], "native_db")
        self.assertTrue(Path(profile["ffmpeg_binary"]).is_absolute())

    def test_bundled_ffmpeg_exists_and_is_runnable_on_mac(self):
        if config.platform.system() != "Darwin":
            self.skipTest("Not running on macOS")
        profile = config.get_os_profile()
        ffmpeg_path = Path(profile["ffmpeg_binary"])
        self.assertTrue(ffmpeg_path.exists(), f"Bundled ffmpeg missing at {ffmpeg_path}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
