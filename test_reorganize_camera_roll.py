"""
Automated regression tests for reorganize_camera_roll.py - all against an
isolated temp-directory fixture, never the real Camera Roll folder. Run
before/after any change:

    python3 test_reorganize_camera_roll.py
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from PIL import Image

import reorganize_camera_roll as rcr


class TestCanonicalFolderName(unittest.TestCase):
    def test_derives_year_and_month(self):
        self.assertEqual(rcr.canonical_folder_name("2023-06-23 18:41:11"), "202306__")

    def test_single_digit_month_stays_zero_padded(self):
        self.assertEqual(rcr.canonical_folder_name("2024-01-05 09:00:00"), "202401__")


class TestClassifyExistingFolders(unittest.TestCase):
    def test_splits_canonical_variant_and_ignores_non_date_folders(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for name in ["202204__", "202309_a", "202309_b", "202412_g", "202501_", "Vn", "italy"]:
                (root / name).mkdir()

            canonical, variants = rcr.classify_existing_folders(root)

            self.assertEqual(set(canonical.keys()), {"202204"})
            self.assertEqual(canonical["202204"], root / "202204__")
            self.assertEqual(set(variants.keys()), {"202309", "202412", "202501"})
            self.assertEqual({p.name for p in variants["202309"]}, {"202309_a", "202309_b"})


class TestFindAaePairs(unittest.TestCase):
    def test_separates_sidecars_from_media_and_pairs_by_stem(self):
        paths = [Path("IMG_1.MOV"), Path("IMG_1.AAE"), Path("IMG_2.HEIC")]
        media, aae_by_stem = rcr.find_aae_pairs(paths)
        self.assertEqual(media, [Path("IMG_1.MOV"), Path("IMG_2.HEIC")])
        self.assertEqual(aae_by_stem, {"IMG_1": Path("IMG_1.AAE")})


class TestSuffixedName(unittest.TestCase):
    def test_inserts_before_extension(self):
        self.assertEqual(rcr.suffixed_name("IMG_1234.jpg", 1), "IMG_1234 (dup1).jpg")
        self.assertEqual(rcr.suffixed_name("clip.mov", 2), "clip (dup2).mov")


class TestPickCollisionWinner(unittest.TestCase):
    def _make_image(self, path, size):
        img = Image.new("RGB", size)
        img.save(path, "JPEG")

    def test_larger_valid_file_wins(self):
        with tempfile.TemporaryDirectory() as tmp:
            small = Path(tmp) / "small.jpg"
            large = Path(tmp) / "large.jpg"
            self._make_image(small, (10, 10))
            self._make_image(large, (500, 500))

            winner, losers = rcr.pick_collision_winner([small, large], ffmpeg_binary="ffmpeg")
            self.assertEqual(winner, large)
            self.assertEqual(losers, [small])

    def test_corrupted_file_loses_regardless_of_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            valid = Path(tmp) / "valid.jpg"
            corrupted = Path(tmp) / "corrupted.jpg"
            self._make_image(valid, (10, 10))
            corrupted.write_bytes(b"not a real jpeg" * 1000)  # bigger in bytes, but unopenable

            winner, losers = rcr.pick_collision_winner([valid, corrupted], ffmpeg_binary="ffmpeg")
            self.assertEqual(winner, valid)
            self.assertEqual(losers, [corrupted])


class TestContentHash(unittest.TestCase):
    def test_identical_content_same_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jpg"
            b = Path(tmp) / "b.jpg"
            a.write_bytes(b"identical bytes")
            b.write_bytes(b"identical bytes")
            self.assertEqual(rcr.content_hash(a), rcr.content_hash(b))

    def test_different_content_different_hash(self):
        with tempfile.TemporaryDirectory() as tmp:
            a = Path(tmp) / "a.jpg"
            b = Path(tmp) / "b.jpg"
            a.write_bytes(b"these bytes")
            b.write_bytes(b"other bytes")
            self.assertNotEqual(rcr.content_hash(a), rcr.content_hash(b))


class TestFindDuplicateGroups(unittest.TestCase):
    def test_groups_identical_content_regardless_of_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "IMG_1234.jpg"
            b = root / "9f8c7a2b-uuid-export.jpg"
            unique = root / "IMG_5678.jpg"
            a.write_bytes(b"same photo")
            b.write_bytes(b"same photo")
            unique.write_bytes(b"a different photo")

            groups = rcr.find_duplicate_groups([a, b, unique])

            self.assertEqual(len(groups), 1)
            (members,) = groups.values()
            self.assertEqual(set(members), {a, b})

    def test_never_hashes_files_with_a_unique_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            a = root / "a.jpg"
            b = root / "b.jpg"
            a.write_bytes(b"x" * 10)
            b.write_bytes(b"y" * 20)  # different size - can never be a's duplicate

            with patch.object(rcr, "content_hash") as mock_hash:
                groups = rcr.find_duplicate_groups([a, b])

        mock_hash.assert_not_called()
        self.assertEqual(groups, {})


class TestPickDuplicateKeeper(unittest.TestCase):
    def test_prefers_file_already_in_a_canonical_folder(self):
        canonical_folder = Path("/CameraRoll/202306__")
        already_filed = canonical_folder / "IMG_1234.jpg"
        loose_duplicate = Path("/CameraRoll/Photos back up/export-uuid.jpg")

        keeper = rcr.pick_duplicate_keeper(
            [loose_duplicate, already_filed], canonical_folder_paths={canonical_folder})

        self.assertEqual(keeper, already_filed)

    def test_falls_back_to_shortest_filename(self):
        short = Path("/CameraRoll/Vn/IMG_1234.jpg")
        long = Path("/CameraRoll/Photos back up/9f8c7a2b-uuid-export.jpg")

        keeper = rcr.pick_duplicate_keeper([long, short], canonical_folder_paths=set())

        self.assertEqual(keeper, short)


class TestBuildPlanDeduplication(unittest.TestCase):
    """build_plan integration: duplicates across different source folders
    are detected, exactly one copy is kept, and the rest are routed to
    Duplicates/ rather than the normal date-sorted destination."""

    def test_duplicate_across_folders_keeps_one_routes_rest_to_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            already_filed = root / "202306__" / "IMG_1234.jpg"
            already_filed.parent.mkdir(parents=True)
            already_filed.write_bytes(b"same photo content")

            loose_duplicate = root / "Photos back up" / "export-uuid.jpg"
            loose_duplicate.parent.mkdir(parents=True)
            loose_duplicate.write_bytes(b"same photo content")

            with patch.object(rcr, "extract_true_date", return_value=None):
                plan = rcr.build_plan(root, ffmpeg_binary="ffmpeg")

            by_source = {source: (dest, reason) for source, dest, reason in plan}

            # The already-correctly-filed copy needs no action at all.
            self.assertNotIn(already_filed, by_source)
            # The loose duplicate is routed to Duplicates/, not date-sorted.
            dest, reason = by_source[loose_duplicate]
            self.assertEqual(dest, root / rcr.DUPLICATES_FOLDER / "export-uuid.jpg")
            self.assertIn("duplicate of", reason)

    def test_duplicates_own_aae_sidecar_follows_it_to_duplicates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            already_filed = root / "202306__" / "IMG_1234.jpg"
            already_filed.parent.mkdir(parents=True)
            already_filed.write_bytes(b"same photo content")

            loose_duplicate = root / "Photos back up" / "IMG_9999.jpg"
            loose_duplicate.parent.mkdir(parents=True)
            loose_duplicate.write_bytes(b"same photo content")
            loose_aae = root / "Photos back up" / "IMG_9999.AAE"
            loose_aae.write_bytes(b"edit metadata")

            with patch.object(rcr, "extract_true_date", return_value=None):
                plan = rcr.build_plan(root, ffmpeg_binary="ffmpeg")

            by_source = {source: dest for source, dest, _ in plan}

            # The .AAE follows its duplicate media file to Duplicates/, not
            # Unknown date/ - it shouldn't be treated as orphaned just
            # because its pair got filtered out of the normal date-sort.
            self.assertEqual(by_source[loose_aae], root / rcr.DUPLICATES_FOLDER / "IMG_9999.AAE")


class TestSkipsAppleDoubleFiles(unittest.TestCase):
    """Real bug found live 10-08-2026: the dry run against the actual drive
    put 6,779 macOS AppleDouble sidecar files ("._IMG_1234.mov") into the
    plan, mostly landing in Unknown date/ since they carry no real metadata
    of their own - config.walk_media_files() already filters these out
    elsewhere in the project, but this script's own directory listings
    didn't use it. Regression guard for all three fixed call sites."""

    def _touch(self, path, content=b"x"):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)

    def test_loose_root_and_named_subfolders_skip_appledouble(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._touch(root / "IMG_0001.HEIC")
            self._touch(root / "._IMG_0001.HEIC")
            self._touch(root / "Vn" / "IMG_0002.MOV")
            self._touch(root / "Vn" / "._IMG_0002.MOV")

            files = rcr.collect_loose_sources(root)
            names = {p.name for p in files}
            self.assertIn("IMG_0001.HEIC", names)
            self.assertIn("IMG_0002.MOV", names)
            self.assertNotIn("._IMG_0001.HEIC", names)
            self.assertNotIn("._IMG_0002.MOV", names)

    def test_variant_folder_amalgamation_skips_appledouble(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            self._touch(root / "202309_a" / "IMG_0002.MOV")
            self._touch(root / "202309_a" / "._IMG_0002.MOV")

            with patch.object(rcr, "extract_true_date", return_value=None):
                plan = rcr.build_plan(root, ffmpeg_binary="ffmpeg")

            sources = {source.name for source, _, _ in plan}
            self.assertIn("IMG_0002.MOV", sources)
            self.assertNotIn("._IMG_0002.MOV", sources)


class TestBuildPlanIntegration(unittest.TestCase):
    """End-to-end against a small fake Camera Roll - real files on disk,
    but date extraction is mocked (writing real EXIF is unnecessary
    complexity for what this is testing) and nothing here ever touches the
    actual project drive."""

    def _touch(self, path, content=None):
        # Distinct content per file by default (the path itself) - this
        # suite isn't testing duplicate detection, so fixture files must
        # not accidentally collide with each other now that build_plan
        # includes a real content-based dedup pass.
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content if content is not None else str(path).encode())

    def test_amalgamates_and_sorts_with_no_real_moves_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            # Existing canonical + variant folders for the same month.
            self._touch(root / "202309__" / "IMG_0001.MOV")
            self._touch(root / "202309_a" / "IMG_0002.MOV")
            # A variant-only month with no canonical folder yet.
            self._touch(root / "202412_g" / "IMG_0003.MOV")
            # Loose files at the root, with a matching .AAE sidecar.
            self._touch(root / "IMG_0010.HEIC")
            self._touch(root / "IMG_0010.AAE")
            self._touch(root / "IMG_0011.HEIC")  # will get "no date" from the mock below

            def fake_extract_true_date(path, ffmpeg_binary):
                return {
                    "IMG_0002.MOV": "2023-09-10 10:00:00",
                    "IMG_0003.MOV": "2024-12-01 10:00:00",
                    "IMG_0010.HEIC": "2026-05-01 10:00:00",
                }.get(path.name)

            with patch.object(rcr, "extract_true_date", side_effect=fake_extract_true_date):
                plan = rcr.build_plan(root, ffmpeg_binary="ffmpeg")

            by_source_name = {source.name: dest for source, dest, _ in plan}

            # Variant folder's file joins the existing canonical folder.
            self.assertEqual(by_source_name["IMG_0002.MOV"], root / "202309__" / "IMG_0002.MOV")
            # A variant-only month gets a freshly-created canonical folder.
            self.assertEqual(by_source_name["IMG_0003.MOV"], root / "202412__" / "IMG_0003.MOV")
            # A dated loose file is sorted into its month.
            self.assertEqual(by_source_name["IMG_0010.HEIC"], root / "202605__" / "IMG_0010.HEIC")
            # Its .AAE sidecar follows it to the same folder.
            self.assertEqual(by_source_name["IMG_0010.AAE"], root / "202605__" / "IMG_0010.AAE")
            # No date -> Unknown date.
            self.assertEqual(by_source_name["IMG_0011.HEIC"], root / "Unknown date" / "IMG_0011.HEIC")
            # Already-correctly-placed file (IMG_0001.MOV) needs no action at all.
            self.assertNotIn("IMG_0001.MOV", by_source_name)

            # Still nothing actually moved on disk - build_plan is planning-only.
            self.assertTrue((root / "202309_a" / "IMG_0002.MOV").exists())
            self.assertTrue((root / "IMG_0010.HEIC").exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
