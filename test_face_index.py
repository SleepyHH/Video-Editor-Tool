"""
Automated regression tests for face_index.py - the face-recognition indexing
engine (insightface detection, the quality filter, HDBSCAN clustering).

Fast, no real model load, no real photo library or index needed - everything
here runs against an isolated fake drive/index/detector, same pattern as
TestPruneMissingFiles in test_media_search.py. Real-model behavior (does
insightface actually detect/embed correctly) is instead covered by the
smoke-test + regression-check against face_bakeoff_output described in the
2026-08-19 diary entry, not re-derived here.

    python3 test_face_index.py
"""
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
from PIL import Image

import config
import face_index
import media_index


def _fake_face(bbox=(0, 0, 10, 10), embedding=(1.0, 0.0), det_score=0.9, blur=500.0, width=100, height=100):
    return {
        "bbox": bbox,
        "embedding": np.array(list(embedding) + [0.0] * (face_index.FACE_EMBEDDING_DIM - len(embedding)), dtype=np.float32),
        "crop": Image.new("RGB", (width, height), color=(120, 120, 120)),
        "det_score": det_score,
        "blur": blur,
        "width": width,
        "height": height,
    }


class TestPassesQualityFilter(unittest.TestCase):
    """Thresholds picked 2026-08-19 from real diagnostics review - pinning
    the exact boundary behavior so a future tweak doesn't silently drift."""

    def test_low_blur_fails_regardless_of_score_or_size(self):
        face = _fake_face(blur=face_index.BLUR_MIN, det_score=0.99, width=1000, height=1000)
        self.assertFalse(face_index.passes_quality_filter(face))

    def test_good_score_passes_regardless_of_size(self):
        face = _fake_face(det_score=face_index.SCORE_MIN + 0.01, width=10, height=10)
        self.assertTrue(face_index.passes_quality_filter(face))

    def test_low_score_and_small_fails(self):
        face = _fake_face(det_score=face_index.SCORE_MIN, width=face_index.SIZE_RESCUE_MIN - 1, height=1000)
        self.assertFalse(face_index.passes_quality_filter(face))

    def test_low_score_but_large_is_rescued(self):
        face = _fake_face(det_score=face_index.SCORE_MIN, width=face_index.SIZE_RESCUE_MIN, height=face_index.SIZE_RESCUE_MIN)
        self.assertTrue(face_index.passes_quality_filter(face))


class TestBlurVariance(unittest.TestCase):
    def test_flat_image_scores_lower_than_noisy_image(self):
        flat = Image.new("RGB", (200, 200), color=(128, 128, 128))
        rng = np.random.default_rng(0)
        noisy = Image.fromarray(rng.integers(0, 255, (200, 200, 3), dtype="uint8"))
        self.assertLess(face_index.blur_variance(flat), face_index.blur_variance(noisy))

    def test_size_normalization_makes_equally_sharp_crops_comparable(self):
        rng = np.random.default_rng(1)
        small = Image.fromarray(rng.integers(0, 255, (40, 40, 3), dtype="uint8"))
        large = small.resize((300, 300), Image.LANCZOS)
        # Not exactly equal (resizing changes edge content a little), but should
        # land in the same ballpark rather than differing by orders of magnitude
        # the way the pre-fix (non-size-normalized) metric did.
        small_blur, large_blur = face_index.blur_variance(small), face_index.blur_variance(large)
        self.assertLess(abs(small_blur - large_blur), max(small_blur, large_blur))


class TestSchema(unittest.TestCase):
    def test_people_faces_face_indexed_files_tables_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = media_index.init_db(Path(tmp) / "index.sqlite3")
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            self.assertTrue({"people", "faces", "face_indexed_files"}.issubset(tables))
            face_cols = {row[1] for row in conn.execute("PRAGMA table_info(faces)")}
            self.assertTrue({"face_vector_index", "person_id", "discarded", "passes_filter", "cluster_id"}.issubset(face_cols))
            conn.close()

    def test_init_db_is_idempotent_on_an_already_migrated_db(self):
        with tempfile.TemporaryDirectory() as tmp:
            db_path = Path(tmp) / "index.sqlite3"
            media_index.init_db(db_path).close()
            conn = media_index.init_db(db_path)  # second call must not raise
            conn.execute("SELECT * FROM faces").fetchall()
            conn.close()


class TestBuildFaceIndex(unittest.TestCase):
    def test_new_files_get_indexed_and_embeddings_grow(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive_root = Path(tmp) / "FakeDrive"
            (drive_root / "Camera Roll").mkdir(parents=True)
            img_path = drive_root / "Camera Roll" / "a.jpg"
            Image.new("RGB", (50, 50)).save(img_path)

            with patch.object(media_index, "find_volume_by_label", return_value=drive_root), \
                 patch.object(config, "find_volume_by_label", return_value=drive_root), \
                 patch.object(face_index, "list_library_media", return_value=[img_path]), \
                 patch.object(face_index, "load_image_rgb", side_effect=lambda p: str(p)), \
                 patch.object(face_index, "load_insightface_backend", return_value=None), \
                 patch.object(face_index, "detect_insightface", return_value=[_fake_face(), _fake_face(embedding=(0.0, 1.0))]):
                new_files, new_faces = face_index.build_face_index()

                conn = media_index.init_db()
                face_rows = conn.execute("SELECT file_path, passes_filter FROM faces").fetchall()
                indexed_rows = conn.execute("SELECT COUNT(*) FROM face_indexed_files").fetchone()[0]
                conn.close()
                embeddings = np.load(face_index.get_face_embeddings_path())

            self.assertEqual((new_files, new_faces), (1, 2))
            self.assertEqual(len(face_rows), 2)
            self.assertEqual(indexed_rows, 1)
            self.assertEqual(embeddings.shape, (2, face_index.FACE_EMBEDDING_DIM))

    def test_unchanged_file_is_skipped_on_rerun(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive_root = Path(tmp) / "FakeDrive"
            (drive_root / "Camera Roll").mkdir(parents=True)
            img_path = drive_root / "Camera Roll" / "a.jpg"
            Image.new("RGB", (50, 50)).save(img_path)

            with patch.object(media_index, "find_volume_by_label", return_value=drive_root), \
                 patch.object(config, "find_volume_by_label", return_value=drive_root), \
                 patch.object(face_index, "list_library_media", return_value=[img_path]), \
                 patch.object(face_index, "load_image_rgb", side_effect=lambda p: str(p)), \
                 patch.object(face_index, "load_insightface_backend", return_value=None), \
                 patch.object(face_index, "detect_insightface") as mock_detect:
                mock_detect.return_value = [_fake_face()]
                face_index.build_face_index()
                self.assertEqual(mock_detect.call_count, 1)

                new_files, new_faces = face_index.build_face_index()  # rerun, nothing changed on disk
                self.assertEqual((new_files, new_faces), (0, 0))
                self.assertEqual(mock_detect.call_count, 1)  # not called again for the unchanged file

    def test_one_bad_file_does_not_abort_the_whole_run(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive_root = Path(tmp) / "FakeDrive"
            (drive_root / "Camera Roll").mkdir(parents=True)
            good_path = drive_root / "Camera Roll" / "good.jpg"
            bad_path = drive_root / "Camera Roll" / "bad.jpg"
            Image.new("RGB", (50, 50)).save(good_path)
            Image.new("RGB", (50, 50)).save(bad_path)

            def fake_load(p):
                if p == bad_path:
                    raise ValueError("simulated corrupt file")
                return str(p)

            with patch.object(media_index, "find_volume_by_label", return_value=drive_root), \
                 patch.object(config, "find_volume_by_label", return_value=drive_root), \
                 patch.object(face_index, "list_library_media", return_value=[bad_path, good_path]), \
                 patch.object(face_index, "load_image_rgb", side_effect=fake_load), \
                 patch.object(face_index, "load_insightface_backend", return_value=None), \
                 patch.object(face_index, "detect_insightface", return_value=[_fake_face()]):
                new_files, new_faces = face_index.build_face_index()

            self.assertEqual((new_files, new_faces), (1, 1))  # only the good file counted


class TestReclusterFaces(unittest.TestCase):
    """HDBSCAN needs real density contrast to find structure - empirically
    confirmed (including against real, previously-validated same-person
    embeddings from face_bakeoff_output) that even 5-6 points run through it
    in isolation reliably come back all-noise, regardless of how tight they
    are. So these tests mock HDBSCAN's own output and check recluster_faces'
    wiring around it (right subset selected, labels written to the right
    rows, labeled/discarded faces left untouched) rather than relying on
    real clustering behavior at toy dataset sizes."""

    def _seed_faces(self, conn, rows):
        """rows: list of (embedding, person_id, discarded, passes_filter)."""
        embeddings = []
        for i, (embedding, person_id, discarded, passes_filter) in enumerate(rows):
            embeddings.append(embedding)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, person_id, discarded) VALUES (?, 'f.jpg', 'image', 0.9, 500, 100, 100, ?, ?, ?)",
                (i, int(passes_filter), person_id, int(discarded)),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_clusters_unlabeled_faces_and_skips_labeled_or_discarded(self):
        with tempfile.TemporaryDirectory() as tmp:
            drive_root = Path(tmp) / "FakeDrive"
            drive_root.mkdir()
            with patch.object(media_index, "find_volume_by_label", return_value=drive_root), \
                 patch.object(config, "find_volume_by_label", return_value=drive_root):
                conn = media_index.init_db()
                # Rows 0-2: unlabeled, should be the ones handed to HDBSCAN (in that order).
                # Row 3: already labeled -> must be excluded from clustering entirely.
                # Row 4: discarded -> must also be excluded.
                dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
                self._seed_faces(conn, [
                    ([1.0, 0.0] + dummy, None, False, True),
                    ([0.9, 0.1] + dummy, None, False, True),
                    ([0.0, 1.0] + dummy, None, False, True),
                    ([1.0, 0.0] + dummy, 1, False, True),
                    ([1.0, 0.0] + dummy, None, True, True),
                ])

                with patch("hdbscan.HDBSCAN") as mock_hdbscan_cls:
                    mock_hdbscan_cls.return_value.fit_predict.return_value = np.array([0, 0, -1])
                    cluster_count = face_index.recluster_faces(conn)

                # Only the 3 eligible rows' embeddings were handed to HDBSCAN, in face_vector_index order.
                passed_embeddings = mock_hdbscan_cls.return_value.fit_predict.call_args[0][0]
                np.testing.assert_array_equal(passed_embeddings, np.load(face_index.get_face_embeddings_path())[[0, 1, 2]])

                rows = dict(conn.execute("SELECT face_vector_index, cluster_id FROM faces").fetchall())
                conn.close()

            self.assertEqual(cluster_count, 1)
            self.assertEqual((rows[0], rows[1], rows[2]), (0, 0, -1))
            self.assertIsNone(rows[3])  # labeled face untouched (never had a cluster_id set)
            self.assertIsNone(rows[4])  # discarded face untouched


class TestLabeling(unittest.TestCase):
    """label_faces / discard_faces / the list_* browsing helpers - the
    functions the People tab calls into. No Qt involved, matching the
    project's existing DB-logic-stays-testable-without-a-window precedent
    (TestManualImportsPersistence in test_helpers.py)."""

    def _seed(self, conn, faces):
        """faces: list of (embedding_first_2_dims, cluster_id, person_id, discarded)."""
        dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
        embeddings = []
        for i, (e2, cluster_id, person_id, discarded) in enumerate(faces):
            embeddings.append(e2 + dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, cluster_id, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, ?, ?, ?)",
                (i, f"photo_{i}.jpg", cluster_id, person_id, int(discarded), f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

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
        return conn

    def test_list_unlabeled_clusters_excludes_labeled_discarded_and_filtered(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 0, None, False),   # cluster 0
                ([1.0, 0.0], 0, None, False),   # cluster 0
                ([0.0, 1.0], 1, None, False),   # cluster 1
                ([0.0, 1.0], 1, None, False),   # cluster 1
                ([0.0, 1.0], 1, None, False),   # cluster 1
                ([1.0, 1.0], -1, None, False),  # noise
                ([1.0, 1.0], 2, 7, False),      # already labeled - excluded even though cluster_id is set
                ([1.0, 1.0], 3, None, True),    # discarded - excluded
            ])
            clusters = face_index.list_unlabeled_clusters(conn)
            conn.close()

        # Biggest first, noise (-1) always last regardless of size.
        self.assertEqual(clusters, [
            {"cluster_id": 1, "count": 3},
            {"cluster_id": 0, "count": 2},
            {"cluster_id": -1, "count": 1},
        ])

    def test_get_faces_for_cluster_and_for_person(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 0, None, False), ([1.0, 0.0], None, 5, False)])
            cluster_faces = face_index.get_faces_for_cluster(0, conn=conn)
            person_faces = face_index.get_faces_for_person(5, conn=conn)
            conn.close()

        self.assertEqual([f["face_vector_index"] for f in cluster_faces], [0])
        self.assertEqual([f["face_vector_index"] for f in person_faces], [1])

    def test_label_faces_only_touches_the_explicit_list_not_the_whole_cluster(self):
        # The bug this guards against, 2026-08-20: a UI that only displayed/reviewed
        # some of a cluster (main.py caps rendering at MAX_FACES_TO_DISPLAY) must never
        # have MORE faces labeled than what it actually showed. The original approach
        # re-derived "everything in this cluster_id" from the database and was unsafe
        # for that case; label_faces() takes an explicit list and must only ever touch
        # exactly that.
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 0, None, False), ([1.0, 0.0], 0, None, False), ([1.0, 0.0], 0, None, False),
            ])  # a 3-face cluster, but only faces 0 and 1 are ever "shown" to this call
            person_id, count = face_index.label_faces([0, 1], "Alice", conn=conn)
            rows = {r[0]: r[1] for r in conn.execute("SELECT face_vector_index, person_id FROM faces")}
            conn.close()

        self.assertEqual(count, 2)
        self.assertEqual((rows[0], rows[1]), (person_id, person_id))
        self.assertIsNone(rows[2])  # never in the explicit list - must stay untouched

    def test_discard_faces_only_touches_the_explicit_list_not_the_whole_cluster(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 0, None, False), ([1.0, 0.0], 0, None, False), ([1.0, 0.0], 0, None, False),
            ])
            count = face_index.discard_faces([0, 1], conn=conn)
            rows = {r[0]: r[1] for r in conn.execute("SELECT face_vector_index, discarded FROM faces")}
            conn.close()

        self.assertEqual(count, 2)
        self.assertEqual((rows[0], rows[1]), (1, 1))
        self.assertEqual(rows[2], 0)  # never in the explicit list - must stay untouched

    def test_label_faces_rejects_blank_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 0, None, False)])
            with self.assertRaises(ValueError):
                face_index.label_faces([0], "   ", conn=conn)

    def test_list_people_is_alphabetical_not_by_face_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], None, None, False), ([1.0, 0.0], None, None, False), ([1.0, 0.0], None, None, False),
                ([0.0, 1.0], None, None, False),
                ([1.0, 1.0], None, None, False), ([1.0, 1.0], None, None, False),
            ])
            face_index.label_faces([0, 1, 2], "Zach", conn=conn)   # most faces, should sort last
            face_index.label_faces([3], "alice", conn=conn)         # lowercase - still sorts first
            face_index.label_faces([4, 5], "Bob", conn=conn)
            names = [p["name"] for p in face_index.list_people(conn)]
            conn.close()

        self.assertEqual(names, ["alice", "Bob", "Zach"])

    def test_rename_person_renames_when_new_name_is_free(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], None, None, False)])
            person_id, _ = face_index.label_faces([0], "Alise", conn=conn)
            result_id = face_index.rename_person(person_id, "  Alice  ", conn=conn)
            people = face_index.list_people(conn)
            conn.close()

        self.assertEqual(result_id, person_id)  # same person, just renamed
        self.assertEqual(people, [{"person_id": person_id, "name": "Alice", "face_count": 1}])

    def test_rename_person_merges_into_existing_person_on_name_collision(self):
        # people.name is UNIQUE - renaming to a name someone else already has
        # must merge rather than raise, mirroring label_faces()'s own
        # reuse-an-existing-name behavior.
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], None, None, False), ([0.0, 1.0], None, None, False)])
            alice_id, _ = face_index.label_faces([0], "Alice", conn=conn)
            bob_id, _ = face_index.label_faces([1], "Bob", conn=conn)
            result_id = face_index.rename_person(bob_id, "Alice", conn=conn)
            people = face_index.list_people(conn)
            remaining_person_ids = {r[0] for r in conn.execute("SELECT person_id FROM people")}
            face_person_ids = {r[0] for r in conn.execute("SELECT person_id FROM faces")}
            conn.close()

        self.assertEqual(result_id, alice_id)  # merged into the pre-existing "Alice", not "Bob"
        self.assertEqual(people, [{"person_id": alice_id, "name": "Alice", "face_count": 2}])
        self.assertNotIn(bob_id, remaining_person_ids)  # the now-empty old row is gone
        self.assertEqual(face_person_ids, {alice_id})  # both faces now point at the surviving person

    def test_rename_person_rejects_blank_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], None, None, False)])
            person_id, _ = face_index.label_faces([0], "Alice", conn=conn)
            with self.assertRaises(ValueError):
                face_index.rename_person(person_id, "   ", conn=conn)


class TestRecoveryMatching(unittest.TestCase):
    """Phase 2 - compute_person_centroids / propose_matches / apply_matches.
    propose_matches must never write anything (same plan/apply split as
    reorganize_camera_roll.py's build_plan()/execute_plan()); apply_matches
    is the only writer, and only for what's explicitly accepted."""

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
        return conn

    def _seed(self, conn, faces, people=None):
        """faces: list of (embedding_first_2_dims, person_id, discarded, passes_filter).
        people: optional {person_id: name}, inserted first so a face's person_id
        refers to a real row - matching how real labeling always creates the
        person before pointing faces at them."""
        for person_id, name in (people or {}).items():
            conn.execute("INSERT INTO people (person_id, name) VALUES (?, ?)", (person_id, name))
        dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
        embeddings = []
        for i, (e2, person_id, discarded, passes_filter) in enumerate(faces):
            embeddings.append(e2 + dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, ?, ?, ?, ?)",
                (i, f"photo_{i}.jpg", int(passes_filter), person_id, int(discarded), f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_compute_person_centroids_averages_and_renormalizes(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 1, False, True),
                ([0.0, 1.0], 1, False, True),  # same person, 2nd example -> centroid is their normalized mean
                ([0.0, 1.0], 2, False, True),  # a different person
            ], people={1: "Alice", 2: "Bob"})
            centroids = face_index.compute_person_centroids(conn)

        self.assertEqual(set(centroids.keys()), {1, 2})
        expected_1 = np.array([1.0, 1.0] + [0.0] * (face_index.FACE_EMBEDDING_DIM - 2))
        expected_1 /= np.linalg.norm(expected_1)
        np.testing.assert_allclose(centroids[1], expected_1, atol=1e-6)
        self.assertAlmostEqual(float(np.linalg.norm(centroids[1])), 1.0, places=5)

    def test_compute_person_centroids_empty_when_nobody_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], None, False, True)])
            self.assertEqual(face_index.compute_person_centroids(conn), {})

    def test_propose_matches_finds_close_face_ignores_far_face(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 1, False, True),        # Alice's one labeled example
                ([0.99, 0.01], None, False, True),   # close to Alice - should be proposed
                ([0.0, 1.0], None, False, True),     # far from Alice - should not be proposed
            ], people={1: "Alice"})
            plan = face_index.propose_matches(conn=conn)

        self.assertEqual([m["face_vector_index"] for m in plan], [1])
        self.assertEqual(plan[0]["proposed_person_id"], 1)
        self.assertEqual(plan[0]["proposed_person_name"], "Alice")
        self.assertGreater(plan[0]["similarity"], face_index.DEFAULT_MATCH_THRESHOLD)

    def test_propose_matches_returns_empty_with_nobody_labeled(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], None, False, True)])
            self.assertEqual(face_index.propose_matches(conn=conn), [])

    def test_propose_matches_excludes_labeled_discarded_and_filtered_candidates(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 1, False, True),       # Alice's example
                ([1.0, 0.0], 2, False, True),       # already labeled as someone else - not a candidate
                ([1.0, 0.0], None, True, True),     # discarded - not a candidate
                ([1.0, 0.0], None, False, False),   # fails the quality filter - not a candidate
            ], people={1: "Alice", 2: "Bob"})
            plan = face_index.propose_matches(conn=conn)
        self.assertEqual(plan, [])  # nothing left eligible to propose

    def test_propose_matches_never_writes_to_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 1, False, True),
                ([0.99, 0.01], None, False, True),
            ], people={1: "Alice"})
            face_index.propose_matches(conn=conn)
            person_id = conn.execute("SELECT person_id FROM faces WHERE face_vector_index = 1").fetchone()[0]
        self.assertIsNone(person_id)

    def test_propose_matches_sorts_most_confident_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 1, False, True),        # Alice's one labeled example
                ([0.7, 0.71], None, False, True),    # a weaker match
                ([1.0, 0.02], None, False, True),    # a near-perfect match
                ([0.72, 0.69], None, False, True),   # another weak-ish match
            ], people={1: "Alice"})
            plan = face_index.propose_matches(similarity_threshold=0.0, conn=conn)

        scores = [m["similarity"] for m in plan]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_apply_matches_writes_only_accepted_entries(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 1, False, True),
                ([0.99, 0.01], None, False, True),   # will be accepted
                ([0.98, 0.02], None, False, True),   # will be proposed but NOT accepted
            ], people={1: "Alice"})
            plan = face_index.propose_matches(conn=conn)
            self.assertEqual({m["face_vector_index"] for m in plan}, {1, 2})

            applied_count = face_index.apply_matches(plan, accepted_face_vector_indices=[1], conn=conn)
            rows = {r[0]: r[1] for r in conn.execute("SELECT face_vector_index, person_id FROM faces")}

        self.assertEqual(applied_count, 1)
        self.assertEqual(rows[1], 1)     # accepted - now Alice
        self.assertIsNone(rows[2])       # proposed but not accepted - left alone


class TestMatchScoreSeparation(unittest.TestCase):
    """analyze_match_score_separation - real-data groundwork for deciding
    whether an auto-apply tier is safe, using existing clusters (not real
    labels, since nobody's labeled yet) as an identity proxy."""

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
        return conn

    def _seed(self, conn, faces):
        """faces: list of (embedding_first_2_dims, cluster_id)."""
        dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
        embeddings = []
        for i, (e2, cluster_id) in enumerate(faces):
            embeddings.append(e2 + dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, cluster_id, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, NULL, 0, ?)",
                (i, f"photo_{i}.jpg", cluster_id, f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_positive_scores_higher_than_negative_for_well_separated_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.02], 10), ([0.99, -0.01], 10), ([0.98, 0.03], 10), ([1.0, -0.02], 10),
                ([0.02, 1.0], 20), ([-0.01, 0.99], 20), ([0.01, 0.98], 20), ([0.0, 1.0], 20),
            ])
            result = face_index.analyze_match_score_separation(min_cluster_size=2, conn=conn)

        self.assertEqual(len(result["positive"]), 8)
        self.assertEqual(len(result["negative"]), 8)  # 1 other cluster per face
        self.assertGreater(result["positive"].min(), result["negative"].max())  # clean separation

    def test_respects_min_cluster_size(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 10), ([0.99, 0.01], 10), ([0.98, 0.02], 10),  # size 3, qualifies
                ([0.0, 1.0], 20),                                          # size 1, below min
            ])
            result = face_index.analyze_match_score_separation(min_cluster_size=2, conn=conn)

        self.assertEqual(result["cluster_ids"], [10])

    def test_empty_with_fewer_than_two_qualifying_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 10), ([0.99, 0.01], 10)])
            result = face_index.analyze_match_score_separation(min_cluster_size=2, conn=conn)

        self.assertEqual(len(result["positive"]), 0)
        self.assertEqual(len(result["negative"]), 0)

    def test_leave_one_out_excludes_the_face_itself_from_its_own_centroid(self):
        # Two identical points in cluster A, two in cluster B, far apart - each
        # face's leave-one-out centroid is exactly the other identical point,
        # so the positive score should be exactly 1.0, not inflated by partly
        # measuring a face against itself.
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 10), ([1.0, 0.0], 10),
                ([0.0, 1.0], 20), ([0.0, 1.0], 20),
            ])
            result = face_index.analyze_match_score_separation(min_cluster_size=2, conn=conn)

        np.testing.assert_allclose(result["positive"], 1.0, atol=1e-6)


class TestMatchThresholdReport(unittest.TestCase):
    """build_match_threshold_report / write_match_threshold_report - the
    per-face, visual-review companion to analyze_match_score_separation."""

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
        return conn

    def _seed(self, conn, faces):
        """faces: list of (embedding_first_2_dims, cluster_id)."""
        dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
        embeddings = []
        for i, (e2, cluster_id) in enumerate(faces):
            embeddings.append(e2 + dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, cluster_id, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, NULL, 0, ?)",
                (i, f"photo_{i}.jpg", cluster_id, f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_records_have_own_and_worst_cross_cluster_score(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.02], 10), ([0.99, -0.01], 10), ([0.98, 0.03], 10),
                ([0.02, 1.0], 20), ([-0.01, 0.99], 20), ([0.01, 0.98], 20),
            ])
            records = face_index.build_match_threshold_report(min_cluster_size=2, conn=conn)

        self.assertEqual(len(records), 6)
        r = next(r for r in records if r["face_vector_index"] == 0)
        self.assertEqual(r["crop_filename"], "crop_0.jpg")
        self.assertEqual(r["cluster_id"], 10)
        self.assertIsNotNone(r["positive_score"])
        self.assertEqual(r["negative_cluster_id"], 20)  # the only other cluster
        self.assertGreater(r["positive_score"], r["negative_score"])  # well-separated data

    def test_empty_with_fewer_than_two_qualifying_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 10), ([0.99, 0.01], 10)])
            records = face_index.build_match_threshold_report(min_cluster_size=2, conn=conn)
        self.assertEqual(records, [])

    def test_write_report_produces_sorted_html_referencing_real_crop_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.02], 10), ([0.99, -0.01], 10), ([0.5, 0.5], 10),  # last one is a weak/ambiguous member
                ([0.02, 1.0], 20), ([-0.01, 0.99], 20), ([0.5, 0.5], 20),
            ])
            records = face_index.build_match_threshold_report(min_cluster_size=2, conn=conn)
            out_path = Path(tmp) / "report.html"
            face_index.write_match_threshold_report(records, out_path, crops_rel="face_crops")
            html = out_path.read_text()

        self.assertIn("Lowest own-person scores", html)
        self.assertIn("Highest different-person scores", html)
        self.assertIn('src="face_crops/crop_2.jpg"', html)  # the ambiguous face should appear somewhere


class TestSuggestPersonForCluster(unittest.TestCase):
    """suggest_person_for_cluster - the cluster-level counterpart to
    propose_matches(), used to pre-fill the People tab's name box. Pure
    suggestion - these tests also confirm nothing ever gets written."""

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
        return conn

    def _seed(self, conn, faces, people=None):
        """faces: list of (embedding_first_2_dims, cluster_id, person_id)."""
        for person_id, name in (people or {}).items():
            conn.execute("INSERT INTO people (person_id, name) VALUES (?, ?)", (person_id, name))
        dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
        embeddings = []
        for i, (e2, cluster_id, person_id) in enumerate(faces):
            embeddings.append(e2 + dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, cluster_id, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, ?, 0, ?)",
                (i, f"photo_{i}.jpg", cluster_id, person_id, f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_suggests_the_closest_labeled_person_above_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], None, 1),           # Alice's one labeled example
                ([0.0, 1.0], None, 2),           # Bob's one labeled example
                ([0.99, 0.01], 10, None),        # pending cluster 10 - close to Alice
                ([0.98, 0.02], 10, None),
            ], people={1: "Alice", 2: "Bob"})
            suggestion = face_index.suggest_person_for_cluster(10, similarity_threshold=0.5, conn=conn)

        self.assertEqual(suggestion["name"], "Alice")
        self.assertEqual(suggestion["person_id"], 1)
        self.assertGreater(suggestion["similarity"], 0.5)

    def test_no_suggestion_below_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], None, 1),
                ([0.0, 1.0], 10, None),  # nothing like Alice
            ], people={1: "Alice"})
            suggestion = face_index.suggest_person_for_cluster(10, similarity_threshold=0.5, conn=conn)

        self.assertIsNone(suggestion)

    def test_none_when_nobody_labeled_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 10, None)])
            suggestion = face_index.suggest_person_for_cluster(10, conn=conn)
        self.assertIsNone(suggestion)

    def test_writes_nothing_to_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], None, 1),
                ([0.99, 0.01], 10, None),
            ], people={1: "Alice"})
            face_index.suggest_person_for_cluster(10, similarity_threshold=0.5, conn=conn)
            rows = conn.execute("SELECT person_id FROM faces WHERE cluster_id = 10").fetchall()

        self.assertTrue(all(r == (None,) for r in rows))


class TestSuggestPeopleForAllClusters(unittest.TestCase):
    """Batch version of suggest_person_for_cluster() - must give the exact
    same per-cluster answers as calling the single-cluster version
    repeatedly, just computed more efficiently (centroids/embeddings loaded
    once). See main.py's _ordered_pending_clusters, which is why this
    exists - grouping the People tab's whole list needs an answer for every
    pending cluster, not just one at a time."""

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
        return conn

    def _seed(self, conn, faces, people=None):
        """faces: list of (embedding_first_2_dims, cluster_id, person_id)."""
        for person_id, name in (people or {}).items():
            conn.execute("INSERT INTO people (person_id, name) VALUES (?, ?)", (person_id, name))
        dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
        embeddings = []
        for i, (e2, cluster_id, person_id) in enumerate(faces):
            embeddings.append(e2 + dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, cluster_id, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, ?, 0, ?)",
                (i, f"photo_{i}.jpg", cluster_id, person_id, f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_matches_single_cluster_version_for_multiple_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], None, 1),         # Alice's example
                ([0.0, 1.0], None, 2),         # Bob's example
                ([0.99, 0.01], 10, None),      # cluster 10 - close to Alice
                ([0.01, 0.99], 20, None),      # cluster 20 - close to Bob
                ([-1.0, -1.0], 30, None),      # cluster 30 - points away from both, close to neither
            ], people={1: "Alice", 2: "Bob"})

            batch = face_index.suggest_people_for_all_clusters(similarity_threshold=0.5, conn=conn)
            single_10 = face_index.suggest_person_for_cluster(10, similarity_threshold=0.5, conn=conn)
            single_20 = face_index.suggest_person_for_cluster(20, similarity_threshold=0.5, conn=conn)
            single_30 = face_index.suggest_person_for_cluster(30, similarity_threshold=0.5, conn=conn)

        self.assertEqual(batch[10]["name"], single_10["name"])
        self.assertAlmostEqual(batch[10]["similarity"], single_10["similarity"], places=6)
        self.assertEqual(batch[20]["name"], single_20["name"])
        self.assertNotIn(30, batch)
        self.assertIsNone(single_30)

    def test_empty_when_nobody_labeled_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 10, None)])
            self.assertEqual(face_index.suggest_people_for_all_clusters(conn=conn), {})

    def test_writes_nothing_to_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], None, 1),
                ([0.99, 0.01], 10, None),
            ], people={1: "Alice"})
            face_index.suggest_people_for_all_clusters(similarity_threshold=0.5, conn=conn)
            rows = conn.execute("SELECT person_id FROM faces WHERE cluster_id = 10").fetchall()
        self.assertTrue(all(r == (None,) for r in rows))


class TestClusterGroupingSuggestions(unittest.TestCase):
    """suggest_cluster_groupings / write_cluster_grouping_report - the
    cluster-vs-cluster version of match suggestions, for spotting small
    fragments of the same person BEFORE anyone's been labeled. Purely
    informational - these tests also confirm nothing ever gets written."""

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
        return conn

    def _seed(self, conn, faces):
        """faces: list of (embedding_first_2_dims, cluster_id)."""
        dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
        embeddings = []
        for i, (e2, cluster_id) in enumerate(faces):
            embeddings.append(e2 + dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, cluster_id, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, NULL, 0, ?)",
                (i, f"photo_{i}.jpg", cluster_id, f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_two_similar_clusters_suggest_each_other(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 10), ([0.99, 0.01], 10), ([0.98, -0.01], 10),   # cluster 10, "main" cluster
                ([0.97, 0.02], 20), ([0.96, -0.02], 20),                    # cluster 20, a nearby fragment
            ])
            suggestions = face_index.suggest_cluster_groupings(min_cluster_size=2, similarity_threshold=0.5, conn=conn)

        by_cluster = {s["cluster_id"]: s for s in suggestions}
        self.assertEqual(by_cluster[10]["suggested_cluster_id"], 20)
        self.assertEqual(by_cluster[20]["suggested_cluster_id"], 10)
        self.assertEqual(by_cluster[10]["cluster_size"], 3)
        self.assertEqual(by_cluster[20]["suggested_cluster_size"], 3)

    def test_dissimilar_cluster_gets_no_suggestion(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 10), ([0.99, 0.01], 10),
                ([0.0, 1.0], 20), ([-0.01, 0.99], 20),  # a genuinely different person
            ])
            suggestions = face_index.suggest_cluster_groupings(min_cluster_size=2, similarity_threshold=0.5, conn=conn)

        self.assertEqual(suggestions, [])

    def test_sorted_most_confident_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 10), ([1.0, 0.0], 10),
                ([0.99, 0.02], 20), ([0.99, 0.02], 20),   # very close to 10
                ([0.9, 0.1], 30), ([0.9, 0.1], 30),       # closest to 10, but less close than 20
            ])
            suggestions = face_index.suggest_cluster_groupings(min_cluster_size=2, similarity_threshold=0.0, conn=conn)

        scores = [s["similarity"] for s in suggestions]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_writes_nothing_to_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 10), ([0.99, 0.01], 10),
                ([0.98, -0.01], 20), ([0.97, 0.02], 20),
            ])
            face_index.suggest_cluster_groupings(min_cluster_size=2, similarity_threshold=0.5, conn=conn)
            rows = conn.execute("SELECT person_id, discarded FROM faces").fetchall()

        self.assertTrue(all(r == (None, 0) for r in rows))

    def test_empty_with_fewer_than_two_qualifying_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 10), ([0.99, 0.01], 10)])
            suggestions = face_index.suggest_cluster_groupings(min_cluster_size=2, conn=conn)
        self.assertEqual(suggestions, [])

    def test_write_report_references_real_crop_filenames(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 10), ([0.99, 0.01], 10),
                ([0.98, -0.01], 20), ([0.97, 0.02], 20),
            ])
            suggestions = face_index.suggest_cluster_groupings(min_cluster_size=2, similarity_threshold=0.5, conn=conn)
            out_path = Path(tmp) / "grouping.html"
            face_index.write_cluster_grouping_report(suggestions, out_path, crops_rel="face_crops")
            html = out_path.read_text()

        self.assertIn("Cluster grouping suggestions", html)
        self.assertIn('src="face_crops/crop_0.jpg"', html)


class TestCandidateSuggestionsReport(unittest.TestCase):
    """build_candidate_suggestions_report / write_candidate_suggestions_report
    - the full ranked shortlist of candidate people per pending cluster, not
    just suggest_person_for_cluster()'s single #1 pick. Pure review aid -
    these tests also confirm nothing ever gets written."""

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
        return conn

    def _seed(self, conn, faces, people=None):
        """faces: list of (embedding_first_2_dims, cluster_id, person_id)."""
        for person_id, name in (people or {}).items():
            conn.execute("INSERT INTO people (person_id, name) VALUES (?, ?)", (person_id, name))
        dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
        embeddings = []
        for i, (e2, cluster_id, person_id) in enumerate(faces):
            embeddings.append(e2 + dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, cluster_id, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, ?, 0, ?)",
                (i, f"photo_{i}.jpg", cluster_id, person_id, f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_returns_a_ranked_shortlist_not_just_the_best(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 10, None), ([0.99, 0.01], 10, None),   # pending cluster, near [1, 0]
                ([1.0, 0.02], None, 1),   # Alice - near-perfect match
                ([0.7, 0.7], None, 2),    # Bob - a real but weaker runner-up
                ([0.0, 1.0], None, 3),    # Carol - orthogonal, no real match
            ], people={1: "Alice", 2: "Bob", 3: "Carol"})
            records = face_index.build_candidate_suggestions_report(top_k=5, min_similarity=0.3, conn=conn)

        self.assertEqual(len(records), 1)
        names = [c["name"] for c in records[0]["candidates"]]
        self.assertEqual(names, ["Alice", "Bob"])  # Carol's ~0 similarity falls below the floor
        self.assertGreater(records[0]["candidates"][0]["similarity"], records[0]["candidates"][1]["similarity"])

    def test_respects_top_k_cap(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 10, None), ([0.99, 0.01], 10, None),
                ([1.0, 0.0], None, 1), ([0.9, 0.1], None, 2), ([0.8, 0.2], None, 3),
            ], people={1: "Alice", 2: "Bob", 3: "Carol"})
            records = face_index.build_candidate_suggestions_report(top_k=2, min_similarity=0.0, conn=conn)

        self.assertEqual(len(records[0]["candidates"]), 2)
        self.assertEqual([c["name"] for c in records[0]["candidates"]], ["Alice", "Bob"])

    def test_excludes_unclustered_noise_bucket(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], -1, None), ([0.99, 0.01], -1, None),  # noise, not a real cluster
                ([1.0, 0.0], None, 1),
            ], people={1: "Alice"})
            records = face_index.build_candidate_suggestions_report(conn=conn)

        self.assertEqual(records, [])

    def test_empty_when_nobody_labeled_yet(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 10, None), ([0.99, 0.01], 10, None)])
            records = face_index.build_candidate_suggestions_report(conn=conn)

        self.assertEqual(records, [])

    def test_writes_nothing_to_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 10, None), ([0.99, 0.01], 10, None), ([1.0, 0.0], None, 1),
            ], people={1: "Alice"})
            face_index.build_candidate_suggestions_report(conn=conn)
            rows = conn.execute("SELECT person_id, cluster_id, discarded FROM faces WHERE cluster_id = 10").fetchall()

        self.assertTrue(all(r == (None, 10, 0) for r in rows))

    def test_write_report_references_real_crops_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 10, None), ([0.99, 0.01], 10, None), ([1.0, 0.0], None, 1),
            ], people={1: "Alice"})
            records = face_index.build_candidate_suggestions_report(min_similarity=0.3, conn=conn)
            out_path = Path(tmp) / "candidates.html"
            face_index.write_candidate_suggestions_report(records, out_path, crops_rel="face_crops")
            html = out_path.read_text()

        self.assertIn("Candidate suggestions report", html)
        self.assertIn("Alice", html)
        self.assertIn('src="face_crops/crop_0.jpg"', html)  # cluster 10's representative crop
        self.assertIn('src="face_crops/crop_2.jpg"', html)  # Alice's representative crop


class TestProposeAndApplyClusterLabels(unittest.TestCase):
    """propose_cluster_labels / apply_cluster_labels - batch-accepting many
    confident cluster suggestions at once (2026-08-21), while holding out
    genuine close calls for a human. Same propose-then-apply split as
    propose_matches()/apply_matches() - these tests also confirm propose
    writes nothing, and apply only touches what was explicitly accepted."""

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
        return conn

    def _seed(self, conn, faces, people=None):
        """faces: list of (embedding_first_2_dims, cluster_id, person_id)."""
        for person_id, name in (people or {}).items():
            conn.execute("INSERT INTO people (person_id, name) VALUES (?, ?)", (person_id, name))
        dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
        embeddings = []
        for i, (e2, cluster_id, person_id) in enumerate(faces):
            embeddings.append(e2 + dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, cluster_id, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, ?, 0, ?)",
                (i, f"photo_{i}.jpg", cluster_id, person_id, f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_close_call_is_excluded_from_the_plan(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 20, None), ([0.999, 0.045], 20, None),
                ([0.45, 0.893], None, 1),   # Dan - sim ~0.45
                ([0.40, 0.917], None, 2),   # Eve - sim ~0.40, within 0.08 of Dan
            ], people={1: "Dan", 2: "Eve"})
            plan = face_index.propose_cluster_labels(min_similarity=0.3, min_margin=0.08, conn=conn)

        self.assertEqual([p["cluster_id"] for p in plan], [])

    def test_confident_cluster_with_a_clear_margin_is_included(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 30, None), ([0.999, 0.045], 30, None),
                ([1.0, 0.0], None, 3),      # Frank - sim ~1.0
                ([0.35, 0.937], None, 4),   # Grace - sim ~0.35, a healthy 0.65 gap from Frank
            ], people={3: "Frank", 4: "Grace"})
            plan = face_index.propose_cluster_labels(min_similarity=0.3, min_margin=0.08, conn=conn)

        self.assertEqual(len(plan), 1)
        self.assertEqual(plan[0]["cluster_id"], 30)
        self.assertEqual(plan[0]["candidates"][0]["name"], "Frank")

    def test_propose_writes_nothing_to_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 30, None), ([0.999, 0.045], 30, None), ([1.0, 0.0], None, 3),
            ], people={3: "Frank"})
            face_index.propose_cluster_labels(min_similarity=0.3, min_margin=0.08, conn=conn)
            rows = conn.execute("SELECT person_id, cluster_id FROM faces WHERE cluster_id = 30").fetchall()

        self.assertTrue(all(r == (None, 30) for r in rows))

    def test_apply_only_labels_accepted_clusters(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 30, None), ([0.999, 0.045], 30, None),   # cluster 30 -> Frank
                ([0.0, 1.0], 40, None), ([0.0, 0.999], 40, None),     # cluster 40 -> Ivy
                ([1.0, 0.0], None, 3),
                ([0.0, 1.0], None, 4),
            ], people={3: "Frank", 4: "Ivy"})
            plan = face_index.propose_cluster_labels(min_similarity=0.3, min_margin=0.08, conn=conn)
            count = face_index.apply_cluster_labels(plan, accepted_cluster_ids=[30], conn=conn)
            cluster30_faces = conn.execute(
                "SELECT person_id, cluster_id FROM faces WHERE face_vector_index IN (0, 1)"
            ).fetchall()
            cluster40_faces = conn.execute(
                "SELECT person_id, cluster_id FROM faces WHERE face_vector_index IN (2, 3)"
            ).fetchall()

        self.assertEqual(len(plan), 2)  # both clusters had a clear single-candidate winner
        self.assertEqual(count, 2)  # only cluster 30's 2 faces got labeled
        self.assertTrue(all(r == (3, None) for r in cluster30_faces))  # cluster 30 labeled as Frank
        self.assertTrue(all(r == (None, 40) for r in cluster40_faces))  # cluster 40 left untouched

    def test_apply_clears_cluster_id_after_labeling(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 30, None), ([0.999, 0.045], 30, None), ([1.0, 0.0], None, 3),
            ], people={3: "Frank"})
            plan = face_index.propose_cluster_labels(min_similarity=0.3, min_margin=0.08, conn=conn)
            face_index.apply_cluster_labels(plan, accepted_cluster_ids=[30], conn=conn)
            rows = conn.execute("SELECT person_id, cluster_id FROM faces WHERE face_vector_index IN (0, 1)").fetchall()

        self.assertTrue(all(r == (3, None) for r in rows))


class TestUnlabelAndUndo(unittest.TestCase):
    """unlabel_faces (Remove from Person) / snapshot_face_states + restore_face_states
    (single-level Undo) - both 2026-08-24 requests."""

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
        return conn

    def _seed(self, conn, faces):
        """faces: list of (embedding_first_2_dims, cluster_id, person_id, discarded)."""
        dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
        embeddings = []
        for i, (e2, cluster_id, person_id, discarded) in enumerate(faces):
            embeddings.append(e2 + dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, cluster_id, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, ?, ?, ?)",
                (i, f"photo_{i}.jpg", cluster_id, person_id, int(discarded), f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_unlabel_faces_only_touches_the_explicit_list(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            conn.execute("INSERT INTO people (person_id, name) VALUES (1, 'Alice')")
            self._seed(conn, [
                ([1.0, 0.0], None, 1, False), ([1.0, 0.0], None, 1, False), ([1.0, 0.0], None, 1, False),
            ])
            count = face_index.unlabel_faces([0, 1], conn=conn)
            rows = {r[0]: r[1] for r in conn.execute("SELECT face_vector_index, person_id FROM faces")}

        self.assertEqual(count, 2)
        self.assertIsNone(rows[0])
        self.assertIsNone(rows[1])
        self.assertEqual(rows[2], 1)  # never in the explicit list - must stay labeled

    def test_unlabel_faces_deletes_person_left_with_zero_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            conn.execute("INSERT INTO people (person_id, name) VALUES (1, 'Alice')")
            self._seed(conn, [([1.0, 0.0], None, 1, False)])
            face_index.unlabel_faces([0], conn=conn)
            people = face_index.list_people(conn)

        self.assertEqual(people, [])

    def test_unlabel_faces_keeps_person_with_remaining_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            conn.execute("INSERT INTO people (person_id, name) VALUES (1, 'Alice')")
            self._seed(conn, [([1.0, 0.0], None, 1, False), ([1.0, 0.0], None, 1, False)])
            face_index.unlabel_faces([0], conn=conn)
            people = face_index.list_people(conn)

        self.assertEqual(people, [{"person_id": 1, "name": "Alice", "face_count": 1}])

    def test_snapshot_captures_current_state(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 5, None, False), ([1.0, 0.0], None, 2, True)])
            snapshot = face_index.snapshot_face_states([0, 1], conn=conn)

        self.assertEqual(snapshot[0], {"person_id": None, "cluster_id": 5, "discarded": 0})
        self.assertEqual(snapshot[1], {"person_id": 2, "cluster_id": None, "discarded": 1})

    def test_restore_reverses_a_label_action(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 5, None, False)])
            snapshot = face_index.snapshot_face_states([0], conn=conn)  # before labeling
            face_index.label_faces([0], "Alice", conn=conn)
            row_after_label = conn.execute("SELECT person_id, cluster_id FROM faces WHERE face_vector_index = 0").fetchone()

            face_index.restore_face_states(snapshot, conn=conn)
            row_after_undo = conn.execute("SELECT person_id, cluster_id FROM faces WHERE face_vector_index = 0").fetchone()

        self.assertIsNotNone(row_after_label[0])  # really was labeled
        self.assertEqual(row_after_undo, (None, 5))  # back to exactly the pre-label state

    def test_restore_cleans_up_the_now_empty_person_it_created(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 5, None, False)])
            snapshot = face_index.snapshot_face_states([0], conn=conn)
            face_index.label_faces([0], "Brand New Person", conn=conn)  # creates the person row

            face_index.restore_face_states(snapshot, conn=conn)
            people = face_index.list_people(conn)

        self.assertEqual(people, [])  # the person that only ever had this one face is gone


class TestGetFilePathsForPerson(unittest.TestCase):
    """get_file_paths_for_person() - the join point media_search.search()'s
    person filter uses (2026-08-24)."""

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
        return conn

    def _seed(self, conn, faces, people=None):
        """faces: list of (file_path, person_id, discarded)."""
        for person_id, name in (people or {}).items():
            conn.execute("INSERT INTO people (person_id, name) VALUES (?, ?)", (person_id, name))
        dummy = [0.0] * face_index.FACE_EMBEDDING_DIM
        embeddings = []
        for i, (file_path, person_id, discarded) in enumerate(faces):
            embeddings.append(dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, ?, ?)",
                (i, file_path, person_id, int(discarded), f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_returns_paths_for_the_named_person_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ("DRIVE::photo1.jpg", 1, False),
                ("DRIVE::photo2.jpg", 1, False),
                ("DRIVE::photo3.jpg", 2, False),
            ], people={1: "Alice", 2: "Bob"})
            paths = face_index.get_file_paths_for_person("Alice", conn=conn)

        self.assertEqual(paths, {"DRIVE::photo1.jpg", "DRIVE::photo2.jpg"})

    def test_deduplicates_multiple_faces_in_the_same_photo(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ("DRIVE::group_photo.jpg", 1, False),
                ("DRIVE::group_photo.jpg", 1, False),  # a second face of Alice in the same photo
            ], people={1: "Alice"})
            paths = face_index.get_file_paths_for_person("Alice", conn=conn)

        self.assertEqual(paths, {"DRIVE::group_photo.jpg"})

    def test_excludes_discarded_faces(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [("DRIVE::photo1.jpg", 1, True)], people={1: "Alice"})
            paths = face_index.get_file_paths_for_person("Alice", conn=conn)

        self.assertEqual(paths, set())

    def test_unknown_name_returns_empty_set(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [("DRIVE::photo1.jpg", 1, False)], people={1: "Alice"})
            paths = face_index.get_file_paths_for_person("Nobody", conn=conn)

        self.assertEqual(paths, set())


class TestExtractMentionedPeople(unittest.TestCase):
    """extract_mentioned_people() - the natural-language layer media_search.
    smart_search() uses so "Huy and cats" means "photos of Huy, ranked by
    'cats'" (2026-08-24). Case-sensitive by design, refined the same day -
    replaced an earlier hand-curated exclusion list for names that collide
    with common English words: matching only the exact stored capitalization
    disambiguates the whole category at once ("Red" the person vs. "red" the
    color, "An" the person vs. "an" the article), not just the specific
    collisions that happened to be spotted by hand."""

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
        return conn

    def _seed_people(self, conn, names):
        for name in names:
            conn.execute("INSERT INTO people (name) VALUES (?)", (name,))
        conn.commit()

    def test_single_name_extracted_and_remainder_kept(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed_people(conn, ["Alice"])
            remainder, mentioned = face_index.extract_mentioned_people("Alice and cats", conn=conn)

        self.assertEqual(remainder, "cats")
        self.assertEqual(mentioned, ["Alice"])

    def test_multi_word_name_matches_as_one_phrase(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed_people(conn, ["Alice Smith"])
            remainder, mentioned = face_index.extract_mentioned_people("Alice Smith at the beach", conn=conn)

        self.assertEqual(remainder, "at the beach")
        self.assertEqual(mentioned, ["Alice Smith"])

    def test_compound_name_takes_priority_over_its_own_standalone_words(self):
        # Real scenario, 2026-08-24: "Fraser" and "Mum" are each their own
        # labeled person too, separate from the compound "Fraser's Mum" -
        # longest-name-first matching means the compound consumes that
        # stretch of text before the shorter standalone names ever get a
        # chance to match inside it, so this must resolve to ONE person, not
        # two ("Fraser" + "Mum" fragmenting apart).
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed_people(conn, ["Fraser", "Mum", "Fraser's Mum"])
            remainder, mentioned = face_index.extract_mentioned_people("Fraser's Mum at the park", conn=conn)

        self.assertEqual(mentioned, ["Fraser's Mum"])
        self.assertEqual(remainder, "at the park")

    def test_standalone_names_still_match_separately_without_the_possessive(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed_people(conn, ["Fraser", "Mum", "Fraser's Mum"])
            remainder, mentioned = face_index.extract_mentioned_people("Fraser and Mum at the park", conn=conn)

        self.assertEqual(set(mentioned), {"Fraser", "Mum"})
        self.assertEqual(remainder, "at the park")

    def test_multiple_names_all_extracted(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed_people(conn, ["Alice", "Bob"])
            remainder, mentioned = face_index.extract_mentioned_people("Alice and Bob with cats", conn=conn)

        self.assertEqual(remainder, "cats")
        self.assertEqual(mentioned, ["Alice", "Bob"])

    def test_lowercase_name_does_not_match_even_though_unambiguous(self):
        # The real tradeoff of the capitalization rule: a perfectly ordinary,
        # non-colliding name typed lowercase ("alice") no longer auto-detects
        # either, not just the genuinely ambiguous ones - simplicity over a
        # more permissive but harder-to-predict partial rule.
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed_people(conn, ["Alice"])
            remainder, mentioned = face_index.extract_mentioned_people("alice and cats", conn=conn)

        self.assertEqual(mentioned, [])
        self.assertIn("alice", remainder.lower())  # left as ordinary query text

    def test_no_names_mentioned_only_strips_connector_words(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed_people(conn, ["Alice"])
            remainder, mentioned = face_index.extract_mentioned_people("cats and dogs", conn=conn)

        self.assertEqual(remainder, "cats dogs")
        self.assertEqual(mentioned, [])

    def test_ambiguous_common_word_name_only_matches_when_capitalized(self):
        # Real finding, 2026-08-24: "Red" is a real label in the live library
        # AND a common photo-search color word. Lowercase "red" must not
        # silently become "photos of Red" - but properly-capitalized "Red"
        # now correctly can, unlike the old blanket-exclusion approach which
        # disabled auto-detection for this name entirely either way.
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed_people(conn, ["Red"])
            lower_remainder, lower_mentioned = face_index.extract_mentioned_people("a red car", conn=conn)
            cap_remainder, cap_mentioned = face_index.extract_mentioned_people("a Red car", conn=conn)

        self.assertEqual(lower_mentioned, [])
        self.assertIn("red", lower_remainder.lower())
        self.assertEqual(cap_mentioned, ["Red"])

    def test_an_only_matches_the_person_when_capitalized(self):
        # Real conflict found and resolved 2026-08-24: "An" is a real label
        # that also collides with the English article. Case-insensitive
        # matching couldn't satisfy both "an old photo" (should NOT match)
        # and "huy and an with cats" (SHOULD match) at once - capitalization
        # resolves both correctly: lowercase "an" is the article, "An" is
        # the person.
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed_people(conn, ["An"])
            lower_remainder, lower_mentioned = face_index.extract_mentioned_people("an old photo", conn=conn)
            cap_remainder, cap_mentioned = face_index.extract_mentioned_people("An old photo", conn=conn)

        self.assertEqual(lower_mentioned, [])
        self.assertEqual(cap_mentioned, ["An"])
        self.assertEqual(cap_remainder, "old photo")

    def test_family_relation_words_match_when_capitalized(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed_people(conn, ["Dad", "Mum"])
            remainder, mentioned = face_index.extract_mentioned_people("Dad and Mum at the beach", conn=conn)

        self.assertEqual(set(mentioned), {"Dad", "Mum"})
        self.assertEqual(remainder, "at the beach")

    def test_empty_remainder_when_query_is_only_a_name(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed_people(conn, ["Alice"])
            remainder, mentioned = face_index.extract_mentioned_people("Alice", conn=conn)

        self.assertEqual(remainder, "")
        self.assertEqual(mentioned, ["Alice"])


class TestSuggestDuplicatePeople(unittest.TestCase):
    """suggest_duplicate_people() / write_duplicate_people_report() - flags
    labeled people whose FACES look alike (possible split identity), not
    people who are just often photographed together (2026-08-24 real
    finding: "Fraser's Mum"/"Mum" shared 100% of their photos but are
    genuinely different people). Pure suggestion - these tests also confirm
    nothing ever gets written."""

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
        return conn

    def _seed(self, conn, faces, people):
        """faces: list of (embedding_first_2_dims, person_id). people: {person_id: name}."""
        for person_id, name in people.items():
            conn.execute("INSERT INTO people (person_id, name) VALUES (?, ?)", (person_id, name))
        dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
        embeddings = []
        for i, (e2, person_id) in enumerate(faces):
            embeddings.append(e2 + dummy)
            conn.execute(
                "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                "passes_filter, person_id, discarded, crop_filename) "
                "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, 0, ?)",
                (i, f"photo_{i}.jpg", person_id, f"crop_{i}.jpg"),
            )
        conn.commit()
        np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

    def test_finds_similar_looking_people_above_threshold(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 1), ([0.99, 0.01], 1),   # "Alice" - a real cluster of similar faces
                ([0.98, 0.02], 2), ([1.0, 0.0], 2),   # "Alicia" - near-identical face to Alice
            ], people={1: "Alice", 2: "Alicia"})
            suggestions = face_index.suggest_duplicate_people(similarity_threshold=0.5, conn=conn)

        self.assertEqual(len(suggestions), 1)
        self.assertEqual({suggestions[0]["person_a_name"], suggestions[0]["person_b_name"]}, {"Alice", "Alicia"})
        self.assertGreater(suggestions[0]["similarity"], 0.9)

    def test_genuinely_different_looking_people_not_flagged(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 1),
                ([0.0, 1.0], 2),   # orthogonal - a genuinely different-looking face
            ], people={1: "Alice", 2: "Bob"})
            suggestions = face_index.suggest_duplicate_people(similarity_threshold=0.5, conn=conn)

        self.assertEqual(suggestions, [])

    def test_shared_photos_alone_does_not_trigger_a_suggestion(self):
        # The real 2026-08-24 case: two people whose faces look nothing alike,
        # even though every face happens to sit in the exact same photos
        # (photographed together constantly) - must NOT be flagged just from
        # co-occurrence, since suggest_duplicate_people() never even looks at
        # file_path, only the embeddings.
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            conn.execute("INSERT INTO people (person_id, name) VALUES (1, 'Fraser''s Mum')")
            conn.execute("INSERT INTO people (person_id, name) VALUES (2, 'Mum')")
            dummy = [0.0] * (face_index.FACE_EMBEDDING_DIM - 2)
            embeddings = [[1.0, 0.0] + dummy, [0.0, 1.0] + dummy]
            for i, (path, pid) in enumerate([("shared.jpg", 1), ("shared.jpg", 2)]):
                conn.execute(
                    "INSERT INTO faces (face_vector_index, file_path, media_type, det_score, blur, width, height, "
                    "passes_filter, person_id, discarded, crop_filename) "
                    "VALUES (?, ?, 'image', 0.9, 500, 100, 100, 1, ?, 0, ?)",
                    (i, path, pid, f"crop_{i}.jpg"),
                )
            conn.commit()
            np.save(face_index.get_face_embeddings_path(), np.array(embeddings, dtype="float32"))

            suggestions = face_index.suggest_duplicate_people(similarity_threshold=0.5, conn=conn)

        self.assertEqual(suggestions, [])

    def test_sorted_most_confident_first(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [
                ([1.0, 0.0], 1),
                ([0.99, 0.02], 2),   # very close to person 1
                ([0.9, 0.1], 3),     # less close to person 1
            ], people={1: "Alice", 2: "Bob", 3: "Carol"})
            suggestions = face_index.suggest_duplicate_people(similarity_threshold=0.0, conn=conn)

        scores = [s["similarity"] for s in suggestions]
        self.assertEqual(scores, sorted(scores, reverse=True))

    def test_writes_nothing_to_the_database(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 1), ([0.99, 0.01], 2)], people={1: "Alice", 2: "Alicia"})
            face_index.suggest_duplicate_people(similarity_threshold=0.5, conn=conn)
            rows = conn.execute("SELECT person_id FROM faces").fetchall()

        self.assertEqual({r[0] for r in rows}, {1, 2})  # unchanged - nothing merged

    def test_empty_with_fewer_than_two_people(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 1)], people={1: "Alice"})
            suggestions = face_index.suggest_duplicate_people(conn=conn)

        self.assertEqual(suggestions, [])

    def test_write_report_references_real_crops_and_names(self):
        with tempfile.TemporaryDirectory() as tmp:
            conn = self._fake_conn(tmp)
            self._seed(conn, [([1.0, 0.0], 1), ([0.99, 0.01], 2)], people={1: "Alice", 2: "Alicia"})
            suggestions = face_index.suggest_duplicate_people(similarity_threshold=0.5, conn=conn)
            out_path = Path(tmp) / "duplicates.html"
            face_index.write_duplicate_people_report(suggestions, out_path, crops_rel="face_crops")
            html = out_path.read_text()

        self.assertIn("Possible duplicate people", html)
        self.assertIn("Alice", html)
        self.assertIn("Alicia", html)
        self.assertIn('src="face_crops/crop_0.jpg"', html)
        self.assertIn('src="face_crops/crop_1.jpg"', html)


class TestScoreFaceQuality(unittest.TestCase):
    """Continuous ranking used to pick ONE representative frame out of
    several within a consolidated video run - Phase 3, 2026-08-24."""

    def test_higher_det_score_ranks_higher_all_else_equal(self):
        weak = _fake_face(det_score=0.5, blur=100, width=150, height=150)
        strong = _fake_face(det_score=0.9, blur=100, width=150, height=150)
        self.assertGreater(face_index.score_face_quality(strong), face_index.score_face_quality(weak))

    def test_sharper_and_bigger_ranks_higher_at_equal_det_score(self):
        worse = _fake_face(det_score=0.8, blur=10, width=50, height=50)
        better = _fake_face(det_score=0.8, blur=200, width=200, height=200)
        self.assertGreater(face_index.score_face_quality(better), face_index.score_face_quality(worse))


class TestConsolidateFaceRuns(unittest.TestCase):
    """consolidate_face_runs() - Phase 3's one genuinely new algorithm:
    collapses a video's per-frame face detections into one row per
    continuous appearance, instead of one row per sampled second (2026-08-24).
    All three scenarios explicitly called for in the original plan."""

    def test_one_continuous_appearance_becomes_one_run(self):
        same_person = [_fake_face(embedding=(1.0, 0.0)) for _ in range(5)]
        frame_detections = [(float(t), [same_person[t]]) for t in range(5)]

        runs = face_index.consolidate_face_runs(frame_detections)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["start_seconds"], 0.0)
        self.assertEqual(runs[0]["end_seconds"], 4.0)

    def test_two_people_on_screen_at_once_become_two_runs(self):
        frame_detections = [
            (float(t), [_fake_face(embedding=(1.0, 0.0)), _fake_face(embedding=(0.0, 1.0))])
            for t in range(4)
        ]

        runs = face_index.consolidate_face_runs(frame_detections)

        self.assertEqual(len(runs), 2)
        for run in runs:
            self.assertEqual(run["start_seconds"], 0.0)
            self.assertEqual(run["end_seconds"], 3.0)

    def test_leaving_and_returning_is_two_separate_runs_not_bridged(self):
        frame_detections = [
            (0.0, [_fake_face(embedding=(1.0, 0.0))]),
            (1.0, [_fake_face(embedding=(1.0, 0.0))]),
            # gone from 2s to 14s - a real absence, well past the default 3s tolerance
            (15.0, [_fake_face(embedding=(1.0, 0.0))]),
            (16.0, [_fake_face(embedding=(1.0, 0.0))]),
        ]

        runs = face_index.consolidate_face_runs(frame_detections)

        self.assertEqual(len(runs), 2)
        starts = sorted(r["start_seconds"] for r in runs)
        ends = sorted(r["end_seconds"] for r in runs)
        self.assertEqual(starts, [0.0, 15.0])
        self.assertEqual(ends, [1.0, 16.0])

    def test_brief_gap_within_tolerance_does_not_break_the_run(self):
        # A missed frame (head-turn/blink/brief occlusion) at t=2 - within the
        # default 3s gap tolerance of the run's last sighting at t=1.
        frame_detections = [
            (0.0, [_fake_face(embedding=(1.0, 0.0))]),
            (1.0, [_fake_face(embedding=(1.0, 0.0))]),
            (2.0, []),  # missed this frame entirely
            (3.0, [_fake_face(embedding=(1.0, 0.0))]),
        ]

        runs = face_index.consolidate_face_runs(frame_detections)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["start_seconds"], 0.0)
        self.assertEqual(runs[0]["end_seconds"], 3.0)

    def test_representative_frame_is_the_highest_quality_one_in_the_run(self):
        weak = _fake_face(embedding=(1.0, 0.0), det_score=0.5, blur=10, width=50, height=50)
        strong = _fake_face(embedding=(1.0, 0.0), det_score=0.95, blur=200, width=200, height=200)
        frame_detections = [(0.0, [weak]), (1.0, [strong])]

        runs = face_index.consolidate_face_runs(frame_detections)

        self.assertEqual(len(runs), 1)
        self.assertEqual(runs[0]["det_score"], 0.95)
        self.assertEqual(runs[0]["width"], 200)

    def test_empty_frame_detections_returns_no_runs(self):
        self.assertEqual(face_index.consolidate_face_runs([]), [])


if __name__ == "__main__":
    unittest.main(verbosity=2)
