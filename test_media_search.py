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


class TestPruneMissingFiles(unittest.TestCase):
    """prune_missing_files (08-08-2026) - built per the user's request, but
    deliberately never run against the real project index in this session.
    Everything here runs against an isolated fake drive/index instead."""

    def _build_fake_index(self, drive_root):
        import numpy as np
        (drive_root / "existing").mkdir(parents=True)
        real_file = drive_root / "existing" / "kept.mp4"
        real_file.write_text("x")

        # Both patched: get_index_dir() (defined in media_index.py) resolves
        # find_volume_by_label via media_index's own namespace, but
        # resolve_portable_path (defined in config.py) resolves it via
        # config's own namespace internally - patching only one leaves the
        # other still pointed at the real drive.
        with patch.object(media_index, "find_volume_by_label", return_value=drive_root), \
             patch.object(config, "find_volume_by_label", return_value=drive_root):
            conn = media_index.init_db()
            # Row 0: still resolvable - must survive.
            conn.execute(
                "INSERT INTO indexed_files (file_path, mtime) VALUES (?, ?)",
                ("DRIVE::existing/kept.mp4", 1.0))
            conn.execute(
                "INSERT INTO items (vector_index, file_path, media_type, timestamp_seconds) VALUES (0, ?, 'video', 0)",
                ("DRIVE::existing/kept.mp4",))
            # Row 1: DRIVE::-portable but the file's gone - genuine orphan, must be pruned.
            conn.execute(
                "INSERT INTO indexed_files (file_path, mtime) VALUES (?, ?)",
                ("DRIVE::gone/deleted.mp4", 2.0))
            conn.execute(
                "INSERT INTO items (vector_index, file_path, media_type, timestamp_seconds) VALUES (1, ?, 'video', 0)",
                ("DRIVE::gone/deleted.mp4",))
            # Row 2: local-only path (e.g. a PC's own iCloud folder) - expected
            # unresolvable from here, must NOT be pruned even though it "looks missing".
            conn.execute(
                "INSERT INTO indexed_files (file_path, mtime) VALUES (?, ?)",
                (r"C:\Users\someone\Pictures\iCloud Photos\Photos\img.jpg", 3.0))
            conn.execute(
                "INSERT INTO items (vector_index, file_path, media_type, timestamp_seconds) VALUES (2, ?, 'image', NULL)",
                (r"C:\Users\someone\Pictures\iCloud Photos\Photos\img.jpg",))
            conn.commit()
            conn.close()

            embeddings = np.array([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]], dtype="float32")
            np.save(media_index.get_embeddings_path(), embeddings)

    def test_dry_run_reports_without_touching_anything(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            drive_root = Path(tmp) / "FakeDrive"
            self._build_fake_index(drive_root)
            with patch.object(media_index, "find_volume_by_label", return_value=drive_root), \
                 patch.object(config, "find_volume_by_label", return_value=drive_root):
                count = media_index.prune_missing_files(dry_run=True)
                conn = media_index.init_db()
                total = conn.execute("SELECT COUNT(*) FROM indexed_files").fetchone()[0]
                conn.close()
                embeddings = np.load(media_index.get_embeddings_path())

            self.assertEqual(count, 1)
            self.assertEqual(total, 3)  # nothing deleted
            self.assertEqual(len(embeddings), 3)  # nothing compacted

    def test_real_run_prunes_only_the_genuine_orphan(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            drive_root = Path(tmp) / "FakeDrive"
            self._build_fake_index(drive_root)
            with patch.object(media_index, "find_volume_by_label", return_value=drive_root), \
                 patch.object(config, "find_volume_by_label", return_value=drive_root):
                count = media_index.prune_missing_files(dry_run=False)

                conn = media_index.init_db()
                remaining_paths = {row[0] for row in conn.execute("SELECT file_path FROM indexed_files")}
                remaining_items = conn.execute("SELECT file_path, vector_index FROM items ORDER BY vector_index").fetchall()
                conn.close()
                embeddings = np.load(media_index.get_embeddings_path())

            self.assertEqual(count, 1)
            self.assertEqual(remaining_paths, {
                "DRIVE::existing/kept.mp4",
                r"C:\Users\someone\Pictures\iCloud Photos\Photos\img.jpg",
            })
            # Renumbered contiguously: 2 survivors -> vector_index 0 and 1.
            self.assertEqual([vi for _, vi in remaining_items], [0, 1])
            self.assertEqual(len(embeddings), 2)
            # The surviving vectors' actual content followed them, not just their count.
            kept_vi = dict(remaining_items)["DRIVE::existing/kept.mp4"]
            np.testing.assert_array_equal(embeddings[kept_vi], [1.0, 0.0])

    def test_refuses_to_run_when_drive_not_connected(self):
        with patch.object(media_index, "find_volume_by_label", return_value=None):
            with self.assertRaises(RuntimeError):
                media_index.prune_missing_files(dry_run=True)


class TestSmartSearch(unittest.TestCase):
    """smart_search() (2026-08-24) - the natural-language person-detection
    wrapper around search(). Only the paths that don't need the real ~1.6GB
    CLIP model are unit-tested directly here (empty-remainder, and the
    no-people-mentioned delegation) - same reason search() itself is never
    directly unit-tested. The has-remainder/multi-person ranking path is
    covered by TESTING.md's manual checklist + live verification instead."""

    def _fake_conn(self, tmp):
        drive_root = Path(tmp) / "FakeDrive"
        drive_root.mkdir()
        patches = (
            patch.object(media_index, "find_volume_by_label", return_value=drive_root),
            patch.object(config, "find_volume_by_label", return_value=drive_root),
        )
        for p in patches:
            p.start()
            self.addCleanup(p.stop)
        conn = media_index.init_db()
        self.addCleanup(conn.close)
        return conn, drive_root

    def _add_indexed_file(self, conn, drive_root, rel_path, vector_index, date_taken):
        real_file = drive_root / rel_path
        real_file.parent.mkdir(parents=True, exist_ok=True)
        real_file.write_text("x")
        portable = f"DRIVE::{rel_path}"
        conn.execute("INSERT INTO indexed_files (file_path, mtime, date_taken) VALUES (?, ?, ?)",
                     (portable, 1.0, date_taken))
        conn.execute(
            "INSERT INTO items (vector_index, file_path, media_type, timestamp_seconds) VALUES (?, ?, 'image', NULL)",
            (vector_index, portable))
        return portable

    def _label_face(self, conn, face_vector_index, portable_path, person_id, person_name):
        conn.execute("INSERT OR IGNORE INTO people (person_id, name) VALUES (?, ?)", (person_id, person_name))
        conn.execute(
            "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
            "passes_filter, person_id, discarded, crop_filename) "
            "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, 0, ?)",
            (face_vector_index, portable_path, person_id, f"crop_{face_vector_index}.jpg"),
        )

    def test_empty_remainder_returns_persons_files_newest_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn, drive_root = self._fake_conn(tmp)
            older = self._add_indexed_file(conn, drive_root, "older.jpg", 0, "2026-01-01")
            newer = self._add_indexed_file(conn, drive_root, "newer.jpg", 1, "2026-06-01")
            self._label_face(conn, 100, older, 1, "Alice")
            self._label_face(conn, 101, newer, 1, "Alice")
            conn.commit()
            conn.close()  # smart_search() opens its own connections internally -
                           # EXCLUSIVE locking mode means this one must release first.

            results = media_search.smart_search("Alice")

        self.assertEqual(len(results), 2)
        self.assertTrue(all(r["score"] is None for r in results))
        self.assertTrue(all(r["matched_people"] == ["Alice"] for r in results))
        self.assertEqual([Path(r["file_path"]).name for r in results], ["newer.jpg", "older.jpg"])

    def test_empty_remainder_multiple_people_prioritizes_by_match_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn, drive_root = self._fake_conn(tmp)
            both = self._add_indexed_file(conn, drive_root, "both.jpg", 0, "2026-01-01")
            just_alice = self._add_indexed_file(conn, drive_root, "just_alice.jpg", 1, "2026-06-01")
            self._label_face(conn, 100, both, 1, "Alice")
            self._label_face(conn, 101, both, 2, "Bob")
            self._label_face(conn, 102, just_alice, 1, "Alice")
            conn.commit()
            conn.close()

            results = media_search.smart_search("Alice and Bob")

        # "both.jpg" matches 2 mentioned people, "just_alice.jpg" only 1 - the
        # 2-match file must rank first even though it's the OLDER of the two
        # (match_count is the primary sort key, date only applies within a tier).
        self.assertEqual([Path(r["file_path"]).name for r in results], ["both.jpg", "just_alice.jpg"])
        self.assertEqual(results[0]["match_count"], 2)
        self.assertEqual(results[1]["match_count"], 1)

    def test_no_people_mentioned_delegates_straight_to_search(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn, drive_root = self._fake_conn(tmp)
            conn.close()
            with patch.object(media_search, "search", return_value=["sentinel"]) as mock_search:
                results = media_search.smart_search("cats and dogs", top_k=5, after="2026-01-01")

        mock_search.assert_called_once_with("cats and dogs", top_k=5, after="2026-01-01", before=None, file_types=None)
        self.assertEqual(results, ["sentinel"])

    def test_mentioned_person_with_no_files_returns_empty(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn, drive_root = self._fake_conn(tmp)
            conn.execute("INSERT INTO people (person_id, name) VALUES (1, 'Alice')")
            conn.commit()
            conn.close()

            results = media_search.smart_search("Alice")

        self.assertEqual(results, [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
