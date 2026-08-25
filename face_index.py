"""
face_index.py - face-recognition indexing engine: detects faces (insightface's
buffalo_l - SCRFD detector + ArcFace recognizer), filters out low-quality
detections, clusters unlabeled faces by identity (HDBSCAN), and persists
results into the shared index (media_index.init_db()'s `faces`/`people`
tables) plus a parallel `face_embeddings.npy` - the same one-vector-one-
positional-index convention media_index.py already uses for CLIP's `items`.

A separate, independent pipeline from CLIP search - different model, 512-d
ArcFace embeddings vs. CLIP's 768-d, its own clustering step with no CLIP
equivalent. No GUI trigger yet - CLI-only, same as media_index.build_index().

Detection/quality-filter logic moved here from face_bakeoff.py 2026-08-19,
once real review of that script's output settled on real thresholds - this
is now the one place that logic lives; face_bakeoff.py imports it back for
any future model bake-offs rather than keeping its own copy.
"""
import hashlib
import os
import platform
import re
from pathlib import Path

import cv2
import hdbscan
import numpy as np
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

from config import EXTERNAL_DRIVE_LABEL, VIDEO_EXTENSIONS, get_os_profile, to_portable_path
from media_index import (
    FRAME_INTERVAL_SECONDS, IMAGE_EXTENSIONS, extract_video_frames,
    get_device, get_index_dir, init_db, list_library_media,
)

FACE_EMBEDDING_DIM = 512

MAX_IMAGE_DIM = 1600  # downscale before detection - iPhone originals (4032x3024) are far
                       # more than any of these detectors need, and CPU detection time
                       # scales with pixel count
BLUR_NORM_SIZE = 160   # fixed size every crop gets resized to before measuring blur, so a
                       # small crop and a large crop of equally sharp faces score the same -
                       # Laplacian variance otherwise scales with how much edge content is in
                       # the image, which scales with crop size, not with actual sharpness
BLUR_CENTER_MARGIN = 0.15  # fraction trimmed off each side before measuring, so hair/glasses/
                            # background at the crop's edges don't dominate over the face itself

# Quality filter thresholds, picked 2026-08-19 from real review of a bake-off's diagnostics
# report - not guessed, checked against actual crops at the boundary in each case.
BLUR_MIN = 5           # below this the crop is unusably soft (checked against real examples)
SCORE_MIN = 0.64       # below this det_score is usually a bad detection - EXCEPT a big enough
                       # crop can still be a good, correctly-identified face despite a low score
                       # (extreme angle/glare cases) - SIZE_RESCUE_MIN exists so a low score
                       # alone doesn't discard those
SIZE_RESCUE_MIN = 100  # px, both dimensions - a crop at least this big is trusted even with a
                       # low det_score, rather than being dropped by the score check alone
MIN_CLUSTER_SIZE = 3   # a face needs to recur at least this many times to form its own cluster

# Phase 3 (video) run-consolidation parameters, added 2026-08-24 - NOT tuned against
# real video data yet, same "deliberately conservative starting point, revisit once
# real data exists" posture as DEFAULT_MATCH_THRESHOLD below.
FACE_RUN_SIMILARITY_THRESHOLD = 0.6  # higher than DEFAULT_MATCH_THRESHOLD's cross-photo bar -
                                      # frame-to-frame within one run is a much easier comparison
                                      # (near-identical lighting/angle/expression a second apart)
FACE_RUN_GAP_TOLERANCE_SECONDS = 3.0  # how long a person can go undetected (head-turn, blink,
                                       # brief occlusion) before a later appearance counts as a
                                       # genuinely new run instead of a continuation


def get_face_embeddings_path():
    return get_index_dir() / "face_embeddings.npy"


def get_face_crops_dir():
    return get_index_dir() / "face_crops"


def get_reports_dir():
    """Where review/report HTML files (write_candidate_suggestions_report(),
    write_cluster_grouping_report(), write_match_threshold_report()) should
    be written going forward - the OneDrive-synced project folder, sibling
    to this file, NOT get_index_dir(). Confirmed 2026-08-22: media/index
    data (sqlite db, embeddings, face crops - 17GB, dominated by 33k+ crop
    images) stays HDD-only, since syncing a live sqlite db through OneDrive
    risks corruption and a 17GB upload isn't worth it just for some reports.
    But the reports themselves are small, static once written, and meant to
    be easy to open - they belong with the project files instead. Pass
    crops_rel=str(get_face_crops_dir()) (an absolute HDD path) to the
    write_*_report() call when using this, since the report no longer sits
    next to face_crops/ once it's here - a relative "face_crops" path would
    silently break every image reference."""
    reports_dir = Path(__file__).resolve().parent / "search_reports"
    reports_dir.mkdir(exist_ok=True)
    return reports_dir


def load_image_rgb(path):
    img = Image.open(path).convert("RGB")
    if max(img.size) > MAX_IMAGE_DIM:
        scale = MAX_IMAGE_DIM / max(img.size)
        img = img.resize((int(img.width * scale), int(img.height * scale)), Image.LANCZOS)
    return img


def blur_variance(pil_crop):
    """Variance of the Laplacian on a size-normalized, center-cropped version of
    the face crop - a standard cheap blur metric (low variance = low detail/edge
    content = more blurred), made comparable across differently-sized detections
    and less thrown off by non-face content (hair, background) near the edges."""
    img = pil_crop.resize((BLUR_NORM_SIZE, BLUR_NORM_SIZE), Image.LANCZOS)
    m = int(BLUR_NORM_SIZE * BLUR_CENTER_MARGIN)
    center = img.crop((m, m, BLUR_NORM_SIZE - m, BLUR_NORM_SIZE - m))
    gray = cv2.cvtColor(np.array(center), cv2.COLOR_RGB2GRAY)
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def passes_quality_filter(face):
    """Whether a detected face is trustworthy enough to cluster/label, per the
    thresholds above. A face only fails on score if it's ALSO small - a low
    score from pose/glare on an otherwise large, clear face shouldn't be
    thrown out just because the detector wasn't confident about the angle."""
    if face["blur"] <= BLUR_MIN:
        return False
    if face["det_score"] > SCORE_MIN:
        return True
    return face["width"] >= SIZE_RESCUE_MIN and face["height"] >= SIZE_RESCUE_MIN


def score_face_quality(face):
    """Continuous ranking for picking the single BEST frame out of several
    candidates (Phase 3's consolidate_face_runs() - one video "run" of the
    same person collapses to one representative frame) - distinct from
    passes_quality_filter()'s boolean pass/fail, this only ever compares
    candidates against each other, never against an absolute cutoff, so a
    face that already failed passes_quality_filter() can still meaningfully
    outrank another failing face. Combines detector confidence with size and
    sharpness (each capped so one huge outlier frame can't dominate purely
    on size/sharpness while the detector itself was unsure). Not yet tuned
    against real video data - a deliberately simple starting formula, same
    "revisit once real data exists" posture as DEFAULT_MATCH_THRESHOLD."""
    size_score = min(min(face["width"], face["height"]), 200) / 200.0
    blur_score = min(face["blur"], 200.0) / 200.0
    return face["det_score"] * (0.5 + 0.25 * size_score + 0.25 * blur_score)


def load_insightface_backend():
    """ctx_id=-1 is CPU, ctx_id=0 is GPU device 0 - mirrors get_device()'s
    platform detection (cuda -> GPU; cpu/mps -> CPU, since onnxruntime has
    no Apple MPS execution provider).

    On Windows, onnxruntime-gpu's CUDA execution provider needs matching
    cuBLAS/cuDNN DLLs on the process's DLL search path - it doesn't bundle
    them itself, and there's no guarantee they're installed system-wide
    (confirmed missing on a real machine: onnxruntime-gpu silently fell back
    to CPU with zero error, only visible via get_available_providers()).
    Rather than requiring a separate multi-GB CUDA Toolkit install, this
    reuses the copies already bundled inside the project's existing torch
    dependency (same driver/toolkit family CLIP indexing already relies on
    via media_index.get_device())."""
    if platform.system() == "Windows" and hasattr(os, "add_dll_directory"):
        import torch
        torch_lib = Path(torch.__file__).parent / "lib"
        if torch_lib.is_dir():
            os.add_dll_directory(str(torch_lib))

    from insightface.app import FaceAnalysis
    ctx_id = 0 if get_device() == "cuda" else -1
    app = FaceAnalysis(name="buffalo_l")
    app.prepare(ctx_id=ctx_id, det_size=(640, 640))
    return app


def detect_insightface(app, pil_img):
    img_bgr = np.array(pil_img)[:, :, ::-1].copy()
    faces = app.get(img_bgr)
    results = []
    for f in faces:
        x1, y1, x2, y2 = [max(int(v), 0) for v in f.bbox]
        crop_bgr = img_bgr[y1:y2, x1:x2]
        if crop_bgr.size == 0:
            continue
        crop = Image.fromarray(crop_bgr[:, :, ::-1])
        results.append({
            "bbox": (x1, y1, x2, y2),
            "embedding": np.asarray(f.normed_embedding, dtype=np.float32),
            "crop": crop,
            "det_score": float(f.det_score),
            "blur": blur_variance(crop),
            "width": crop.size[0],
            "height": crop.size[1],
        })
    return results


def _crop_filename(storage_path, face_index_in_image):
    """Deterministic (not Python's randomized str hash()) so re-processing the
    same file always names its crops the same way."""
    digest = hashlib.sha1(storage_path.encode()).hexdigest()[:16]
    return f"{digest}_{face_index_in_image}.jpg"


def consolidate_face_runs(frame_detections, similarity_threshold=FACE_RUN_SIMILARITY_THRESHOLD,
                           gap_tolerance_seconds=FACE_RUN_GAP_TOLERANCE_SECONDS):
    """Walks a video's per-frame face detections in timestamp order and
    collapses consecutive same-person detections into "runs" - a person
    talking for 30 seconds at 1 sampled frame/second shouldn't produce 30
    nearly-identical faces rows, just one row spanning [start, end] with a
    single best-quality representative frame (Phase 3, the one genuinely
    new algorithm in the video-support plan).

    frame_detections: [(timestamp_seconds, [face_dict, ...]), ...] in
    timestamp order, one entry per SAMPLED frame - face_dict is whatever
    detect_insightface() returns (bbox/embedding/crop/det_score/blur/
    width/height). A frame can have zero, one, or several faces (multiple
    people on screen at once).

    Two faces in nearby frames are matched into the same run if their
    embeddings are close enough (similarity_threshold) AND the gap since
    that run's last detection is within gap_tolerance_seconds - covering a
    brief head-turn/blink/occlusion without bridging a real absence-and-
    return, which correctly starts a new run instead. Each frame's faces
    are matched independently and greedily (closest-similarity run wins,
    each run can only claim one face per frame), so several people on
    screen simultaneously naturally produce several concurrent runs without
    any special-casing.

    Returns one dict per consolidated run: the winning frame's embedding/
    bbox/crop/det_score/blur/width/height (chosen by score_face_quality(),
    NOT passes_quality_filter() - a run can still get a representative frame
    picked even if every frame in it happens to fail the boolean filter;
    the filter is applied separately by the caller, same as the image path)
    plus "start_seconds"/"end_seconds" spanning the whole run."""
    open_runs = []  # each: {"embeddings": [...], "faces": [...], "start": t, "last_seen": t}
    closed_runs = []

    for timestamp, faces in frame_detections:
        claimed_this_frame = set()
        for face in faces:
            best_idx, best_sim = None, -1.0
            for idx, run in enumerate(open_runs):
                if idx in claimed_this_frame:
                    continue  # a run can only extend by one face per frame
                if timestamp - run["last_seen"] > gap_tolerance_seconds:
                    continue  # too long since this run was last seen - not eligible
                sim = float(np.dot(run["embeddings"][-1], face["embedding"]))
                if sim >= similarity_threshold and sim > best_sim:
                    best_idx, best_sim = idx, sim
            if best_idx is not None:
                run = open_runs[best_idx]
                run["embeddings"].append(face["embedding"])
                run["faces"].append(face)
                run["last_seen"] = timestamp
                claimed_this_frame.add(best_idx)
            else:
                open_runs.append({
                    "embeddings": [face["embedding"]], "faces": [face],
                    "start": timestamp, "last_seen": timestamp,
                })
                claimed_this_frame.add(len(open_runs) - 1)

        still_open = []
        for run in open_runs:
            if timestamp - run["last_seen"] > gap_tolerance_seconds:
                closed_runs.append(run)  # stale as of this frame - won't ever extend again
            else:
                still_open.append(run)
        open_runs = still_open

    closed_runs.extend(open_runs)  # anything still open at the last frame also closes

    results = []
    for run in closed_runs:
        best = max(run["faces"], key=score_face_quality)
        results.append({
            "embedding": best["embedding"], "bbox": best["bbox"], "crop": best["crop"],
            "det_score": best["det_score"], "blur": best["blur"],
            "width": best["width"], "height": best["height"],
            "start_seconds": run["start"], "end_seconds": run["last_seen"],
        })
    return results


def build_face_index(profile=None, progress_callback=None, flush_every=50):
    """Incrementally face-indexes the local library, images AND video (Phase
    3, 2026-08-24): skips files already face-indexed (unchanged path + mtime,
    tracked separately from CLIP's own indexed_files - a file can be
    CLIP-indexed without being face-indexed yet, or vice versa), detects +
    quality-filters + embeds new/changed ones, appends to face_embeddings.npy
    + sqlite `faces` rows. Safe to re-run any time.

    Video frames are sampled the same way media_index.build_index() already
    does for CLIP (extract_video_frames(), FRAME_INTERVAL_SECONDS - a second
    real face-detection pass over the same frames, not shared with CLIP's
    own extraction; see the plan's Open Question 3 for why that's accepted),
    detected per frame, then collapsed via consolidate_face_runs() into one
    faces row per continuous appearance rather than one per sampled second -
    a person talking for 30 seconds shouldn't produce 30 nearly-identical
    rows. timestamp_end_seconds is NULL for images (a single instant), set
    to the run's last-seen timestamp for video.

    Flushes both the embeddings array AND the sqlite commit together every
    `flush_every` files, mirroring media_index.build_index()'s own safety
    fix for the same reason - an interrupted run should never leave sqlite
    claiming a file is face-indexed while its vectors were never saved."""
    profile = profile or get_os_profile()
    conn = init_db()
    backend = load_insightface_backend()

    already_indexed = dict(conn.execute("SELECT file_path, mtime FROM face_indexed_files"))
    library = [p for p in list_library_media(profile) if p.suffix.lower() in IMAGE_EXTENSIONS | VIDEO_EXTENSIONS]

    embeddings_path = get_face_embeddings_path()
    embeddings = np.load(embeddings_path) if embeddings_path.exists() else np.zeros((0, FACE_EMBEDDING_DIM), dtype="float32")
    next_vector_index = embeddings.shape[0]
    new_vectors = []
    crops_dir = get_face_crops_dir()
    crops_dir.mkdir(parents=True, exist_ok=True)

    total_new_files = 0
    total_new_faces = 0
    files_since_flush = 0

    def flush():
        nonlocal embeddings, new_vectors
        if new_vectors:
            embeddings = np.vstack([embeddings, np.array(new_vectors, dtype="float32")])
            np.save(embeddings_path, embeddings)
            new_vectors = []
        conn.commit()

    def insert_face(storage_path, media_type, timestamp_seconds, timestamp_end_seconds, index_in_file, face):
        nonlocal next_vector_index, total_new_faces
        crop_name = _crop_filename(storage_path, index_in_file)
        face["crop"].save(crops_dir / crop_name, "JPEG", quality=85)
        passes = passes_quality_filter(face)
        new_vectors.append(face["embedding"])
        conn.execute(
            "INSERT INTO faces (face_vector_index, file_path, media_type, timestamp_seconds, "
            "timestamp_end_seconds, bbox_x1, bbox_y1, bbox_x2, bbox_y2, det_score, blur, width, "
            "height, passes_filter, crop_filename) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (next_vector_index, storage_path, media_type, timestamp_seconds, timestamp_end_seconds,
             *face["bbox"], face["det_score"], face["blur"], face["width"], face["height"],
             int(passes), crop_name),
        )
        next_vector_index += 1
        total_new_faces += 1

    for i, path in enumerate(library):
        storage_path = to_portable_path(str(path), EXTERNAL_DRIVE_LABEL)
        mtime = path.stat().st_mtime
        if already_indexed.get(storage_path) == mtime:
            continue

        if progress_callback:
            progress_callback(i + 1, len(library), path.name)

        try:
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                frames, tmp_dir = extract_video_frames(
                    profile["ffmpeg_binary"], path, FRAME_INTERVAL_SECONDS,
                    hwaccel=profile.get("ffmpeg_hwaccel"),
                )
                try:
                    frame_detections = [
                        (ts, detect_insightface(backend, load_image_rgb(frame_path)))
                        for frame_path, ts in frames
                    ]
                finally:
                    for f in tmp_dir.glob("*"):
                        f.unlink()
                    tmp_dir.rmdir()
                for j, run in enumerate(consolidate_face_runs(frame_detections)):
                    insert_face(storage_path, "video", run["start_seconds"], run["end_seconds"], j, run)
            else:
                img = load_image_rgb(path)
                for j, face in enumerate(detect_insightface(backend, img)):
                    insert_face(storage_path, "image", None, None, j, face)
            conn.execute(
                "INSERT OR REPLACE INTO face_indexed_files (file_path, mtime) VALUES (?, ?)",
                (storage_path, mtime),
            )
            total_new_files += 1
        except Exception as e:
            print(f"Skipped {path.name}: {e}")
            continue

        files_since_flush += 1
        if files_since_flush >= flush_every:
            flush()
            files_since_flush = 0

    flush()
    conn.close()
    return total_new_files, total_new_faces


def recluster_faces(conn=None):
    """Re-clusters every face that isn't yet labeled, discarded, or filtered
    out. Deliberately separate from build_face_index() - clustering is
    global (one new face can reshuffle other faces' cluster assignments,
    unlike embedding which is purely additive) - so this always reclusters
    the FULL unlabeled set, not just newly-added faces. cluster_id is
    ephemeral: HDBSCAN can renumber every cluster on a rerun, so it's only
    meaningful immediately after this function runs. person_id, once set by
    labeling, is the durable identity and is never touched here - a labeled
    face is excluded from clustering entirely, not just from re-labeling."""
    close_after = conn is None
    conn = conn or init_db()

    rows = conn.execute(
        "SELECT face_vector_index FROM faces WHERE person_id IS NULL AND discarded = 0 AND passes_filter = 1"
    ).fetchall()
    face_indices = [r[0] for r in rows]
    if not face_indices:
        if close_after:
            conn.close()
        return 0

    embeddings = np.load(get_face_embeddings_path())
    subset = embeddings[face_indices]

    clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean")
    labels = clusterer.fit_predict(subset)

    conn.executemany(
        "UPDATE faces SET cluster_id = ? WHERE face_vector_index = ?",
        [(int(label), face_index) for face_index, label in zip(face_indices, labels)],
    )
    conn.commit()
    if close_after:
        conn.close()
    return len(set(l for l in labels if l >= 0))


# --- Labeling: browsing clusters/people and recording the user's decisions ---
# Every function here takes an explicit conn (defaulting to init_db()) rather than reading
# self off a Qt widget, matching the load_manual_imports_from_disk/save_manual_imports_to_disk
# precedent in main.py - keeps this testable without a Qt window and keeps main.py itself free
# of any direct sqlite access.

def list_unlabeled_clusters(conn=None):
    """{cluster_id, count} for every pending group awaiting a decision,
    biggest first - real HDBSCAN clusters (id >= 0) plus a single "-1"
    entry for unclustered noise faces, if any exist. Only ever reflects
    faces from the *last* recluster_faces() run - cluster_id is ephemeral,
    so call recluster_faces() first if faces have been added since."""
    close_after = conn is None
    conn = conn or init_db()
    rows = conn.execute(
        "SELECT cluster_id, COUNT(*) FROM faces "
        "WHERE person_id IS NULL AND discarded = 0 AND passes_filter = 1 AND cluster_id IS NOT NULL "
        "GROUP BY cluster_id ORDER BY cluster_id = -1, COUNT(*) DESC"
    ).fetchall()
    if close_after:
        conn.close()
    return [{"cluster_id": r[0], "count": r[1]} for r in rows]


def list_people(conn=None):
    """{person_id, name, face_count} for every already-labeled person,
    alphabetical by name (case-insensitive) - makes a specific person easy
    to find by scrolling once dozens of people are labeled, unlike sorting
    by face count which reshuffles the list as counts change."""
    close_after = conn is None
    conn = conn or init_db()
    rows = conn.execute(
        "SELECT p.person_id, p.name, COUNT(f.face_vector_index) FROM people p "
        "LEFT JOIN faces f ON f.person_id = p.person_id "
        "GROUP BY p.person_id ORDER BY p.name COLLATE NOCASE ASC"
    ).fetchall()
    if close_after:
        conn.close()
    return [{"person_id": r[0], "name": r[1], "face_count": r[2]} for r in rows]


def get_file_paths_for_person(name, conn=None):
    """Every portable file_path with at least one non-discarded face labeled
    as this person - the join point between face recognition and the CLIP
    search index (media_search.py's search(person=...) filter, added
    2026-08-24). Works because faces.file_path and items.file_path share the
    exact same storage convention (config.to_portable_path) - a plain string
    comparison, no path resolution needed until a result is actually
    displayed. Returns a set for fast membership checks; empty for a name
    that doesn't match any labeled person."""
    close_after = conn is None
    conn = conn or init_db()
    rows = conn.execute(
        "SELECT DISTINCT faces.file_path FROM faces JOIN people ON faces.person_id = people.person_id "
        "WHERE people.name = ? AND faces.discarded = 0",
        (name,),
    ).fetchall()
    if close_after:
        conn.close()
    return {r[0] for r in rows}


# Connector words stripped from what's left after a name is removed, so
# "Huy and cats" cleanly reduces to "cats" for the CLIP half, not "and cats".
_QUERY_CONNECTOR_WORDS = {"and", "with", "featuring", "of", "&"}


def extract_mentioned_people(query, conn=None):
    """Splits a free-text search query into (remaining_text, mentioned_person_
    names) - e.g. "Huy and cats" -> ("cats", ["Huy"]). The natural-language
    layer on top of get_file_paths_for_person(), letting a person's name be
    typed directly into the search box instead of needing the separate
    dropdown filter (2026-08-24 request - media_search.smart_search() is the
    actual caller).

    Matches labeled people's names as whole words/phrases, longest names
    first, so a two-word name like "Alex Kane" matches as one person rather
    than "Alex" then failing on a stray "Kane". CASE-SENSITIVE - the query
    text has to match the name's exact stored capitalization (2026-08-24
    refinement, replacing an earlier hand-curated exclusion list for names
    that collide with common English words like "Red"/"Honey"/"An"): typing
    "Huy" matches the person, but "huy" (or "an", "red", "honey" lowercase)
    doesn't match anyone and just falls through as ordinary search text.
    Simpler and self-maintaining - no list to keep updating as new people
    get labeled - and it's the general fix for the whole category of
    collision, not just the specific ones happened to be spotted by hand.
    The tradeoff: a name typed in lowercase, even an unambiguous one like
    "huy", no longer auto-detects either - it takes the real capitalization
    to trigger the person filter now. Leftover connector words
    (_QUERY_CONNECTOR_WORDS) are stripped from the remainder too."""
    close_after = conn is None
    conn = conn or init_db()
    candidates = sorted(list_people(conn), key=lambda p: len(p["name"]), reverse=True)
    if close_after:
        conn.close()

    remaining = query
    mentioned = []
    for p in candidates:
        pattern = r"\b" + re.escape(p["name"]) + r"\b"
        match = re.search(pattern, remaining)
        if match:
            mentioned.append(p["name"])
            remaining = remaining[:match.start()] + " " + remaining[match.end():]

    words = [w for w in remaining.split() if w.strip(".,!?").lower() not in _QUERY_CONNECTOR_WORDS]
    return " ".join(words).strip(), mentioned


def get_faces_for_cluster(cluster_id, conn=None):
    """The still-pending faces belonging to one unlabeled cluster - what the
    labeling UI shows when a cluster is opened for review."""
    close_after = conn is None
    conn = conn or init_db()
    rows = conn.execute(
        "SELECT face_vector_index, file_path, crop_filename, det_score, blur, width, height FROM faces "
        "WHERE cluster_id = ? AND person_id IS NULL AND discarded = 0 AND passes_filter = 1",
        (cluster_id,),
    ).fetchall()
    if close_after:
        conn.close()
    return [dict(zip(("face_vector_index", "file_path", "crop_filename", "det_score", "blur", "width", "height"), r)) for r in rows]


def get_faces_for_person(person_id, conn=None):
    """Every face already labeled as this person - what the labeling UI
    shows when browsing an existing, named person."""
    close_after = conn is None
    conn = conn or init_db()
    rows = conn.execute(
        "SELECT face_vector_index, file_path, crop_filename, det_score, blur, width, height FROM faces "
        "WHERE person_id = ?",
        (person_id,),
    ).fetchall()
    if close_after:
        conn.close()
    return [dict(zip(("face_vector_index", "file_path", "crop_filename", "det_score", "blur", "width", "height"), r)) for r in rows]


def label_faces(face_vector_indices, name, conn=None):
    """Names an explicit list of faces - the caller decides exactly which
    ones, this function never re-derives "everything in a cluster" on its
    own. That distinction matters once a group is bigger than what a caller
    actually inspected (e.g. main.py's People tab caps how many faces it
    ever renders at once, MAX_FACES_TO_DISPLAY - a UI can only have reviewed
    what it actually displayed, never the full underlying group)."""
    close_after = conn is None
    conn = conn or init_db()
    name = name.strip()
    if not name:
        raise ValueError("A person needs a name.")

    conn.execute("INSERT OR IGNORE INTO people (name) VALUES (?)", (name,))
    person_id = conn.execute("SELECT person_id FROM people WHERE name = ?", (name,)).fetchone()[0]

    conn.executemany(
        "UPDATE faces SET person_id = ?, cluster_id = NULL WHERE face_vector_index = ?",
        [(person_id, i) for i in face_vector_indices],
    )
    conn.commit()
    if close_after:
        conn.close()
    return person_id, len(face_vector_indices)


def discard_faces(face_vector_indices, conn=None):
    """Marks an explicit list of faces as reviewed-and-not-a-real-person
    (blurry double-detection, a stranger in the background, etc.) so they
    stop resurfacing in list_unlabeled_clusters() - same "caller decides
    exactly which ones" reasoning as label_faces()."""
    close_after = conn is None
    conn = conn or init_db()
    conn.executemany(
        "UPDATE faces SET discarded = 1, cluster_id = NULL "
        "WHERE face_vector_index = ? AND person_id IS NULL AND discarded = 0",
        [(i,) for i in face_vector_indices],
    )
    conn.commit()
    count = len(face_vector_indices)
    if close_after:
        conn.close()
    return count


def unlabel_faces(face_vector_indices, conn=None):
    """Removes an explicit list of faces from whoever they're currently labeled
    as - the reverse of label_faces(), for undoing a wrong grouping (2026-08-24
    request: "remove selected photos from the cluster" when browsing a labeled
    person). Sent back to the general unlabeled pool, not discarded - cluster_id
    stays NULL until the next recluster_faces() run picks them up again, same as
    a freshly-detected face. Cleans up any person left with zero faces afterward,
    same reasoning as rename_person()'s merge cleanup - an empty person row is
    just clutter in list_people(), not a meaningful state."""
    close_after = conn is None
    conn = conn or init_db()
    conn.executemany(
        "UPDATE faces SET person_id = NULL WHERE face_vector_index = ?",
        [(i,) for i in face_vector_indices],
    )
    conn.execute(
        "DELETE FROM people WHERE person_id NOT IN "
        "(SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL)"
    )
    conn.commit()
    count = len(face_vector_indices)
    if close_after:
        conn.close()
    return count


def snapshot_face_states(face_vector_indices, conn=None):
    """{face_vector_index: {"person_id", "cluster_id", "discarded"}} for the
    given faces' CURRENT state, right before a write action - the undo
    mechanism (2026-08-24 request) for label_faces()/discard_faces()/
    unlabel_faces()/apply_matches(). Call this immediately before writing,
    then pass the result to restore_face_states() to reverse exactly that
    one action. Single-level by design (main.py only ever keeps the most
    recent snapshot, not a stack) - this function itself doesn't care how
    many snapshots the caller keeps."""
    close_after = conn is None
    conn = conn or init_db()
    if not face_vector_indices:
        if close_after:
            conn.close()
        return {}
    placeholders = ",".join("?" * len(face_vector_indices))
    rows = conn.execute(
        f"SELECT face_vector_index, person_id, cluster_id, discarded FROM faces "
        f"WHERE face_vector_index IN ({placeholders})",
        list(face_vector_indices),
    ).fetchall()
    if close_after:
        conn.close()
    return {r[0]: {"person_id": r[1], "cluster_id": r[2], "discarded": r[3]} for r in rows}


def restore_face_states(face_states, conn=None):
    """Writes back the exact person_id/cluster_id/discarded values captured by
    snapshot_face_states() - undoes whatever write happened in between. Not a
    general-purpose function; it trusts the caller to have snapshotted the
    right faces at the right time. Cleans up any person left with zero faces
    afterward (e.g. undoing the one Confirm that created them), same as
    unlabel_faces()."""
    close_after = conn is None
    conn = conn or init_db()
    conn.executemany(
        "UPDATE faces SET person_id = ?, cluster_id = ?, discarded = ? WHERE face_vector_index = ?",
        [(s["person_id"], s["cluster_id"], s["discarded"], fvi) for fvi, s in face_states.items()],
    )
    conn.execute(
        "DELETE FROM people WHERE person_id NOT IN "
        "(SELECT DISTINCT person_id FROM faces WHERE person_id IS NOT NULL)"
    )
    conn.commit()
    if close_after:
        conn.close()


def rename_person(person_id, new_name, conn=None):
    """Renames an already-labeled person. people.name is UNIQUE, so if
    new_name already belongs to a DIFFERENT person, this merges into that
    existing person instead of erroring - every face currently on person_id
    moves over and the now-empty person_id row is removed. Same "reuse an
    existing name" precedent label_faces() already set for labeling a second
    cluster with a name that's already in use. Returns the resulting
    person_id (unchanged unless a merge happened)."""
    close_after = conn is None
    conn = conn or init_db()
    new_name = new_name.strip()
    if not new_name:
        raise ValueError("A person needs a name.")

    row = conn.execute("SELECT person_id FROM people WHERE name = ?", (new_name,)).fetchone()
    if row and row[0] != person_id:
        target_id = row[0]
        conn.execute("UPDATE faces SET person_id = ? WHERE person_id = ?", (target_id, person_id))
        conn.execute("DELETE FROM people WHERE person_id = ?", (person_id,))
        conn.commit()
        if close_after:
            conn.close()
        return target_id

    conn.execute("UPDATE people SET name = ? WHERE person_id = ?", (new_name, person_id))
    conn.commit()
    if close_after:
        conn.close()
    return person_id


# --- Phase 2: recovery matching for faces that never joined a labeled cluster ---
# Same propose-then-apply split as reorganize_camera_roll.py's build_plan()/execute_plan() -
# propose_matches() computes and returns a plan, writing nothing; apply_matches() is the only
# function that writes, and only for whatever the caller explicitly accepted.

DEFAULT_MATCH_THRESHOLD = 0.5  # cosine similarity - NOT tuned against real data yet (unlike
                                # BLUR_MIN/SCORE_MIN, which were picked from real diagnostics
                                # review). A deliberately conservative starting point - lower it
                                # if real testing shows it's missing obvious matches, raise it if
                                # it's proposing wrong ones. Revisit once real people are labeled.


def compute_person_centroids(conn=None):
    """{person_id: mean L2-normalized embedding} for every labeled person -
    the reference point each unlabeled face gets compared against. Renormalized
    after averaging since the mean of several unit vectors isn't itself unit
    length, and every comparison below assumes unit vectors (cosine similarity
    via a plain dot product)."""
    close_after = conn is None
    conn = conn or init_db()
    rows = conn.execute("SELECT person_id, face_vector_index FROM faces WHERE person_id IS NOT NULL").fetchall()
    if close_after:
        conn.close()
    if not rows:
        return {}

    embeddings = np.load(get_face_embeddings_path())
    by_person = {}
    for person_id, face_vector_index in rows:
        by_person.setdefault(person_id, []).append(embeddings[face_vector_index])

    centroids = {}
    for person_id, vectors in by_person.items():
        mean = np.mean(vectors, axis=0)
        centroids[person_id] = mean / np.linalg.norm(mean)
    return centroids


def propose_matches(similarity_threshold=DEFAULT_MATCH_THRESHOLD, candidate_faces=None, conn=None):
    """For every candidate face with no person yet, finds the closest labeled
    person's centroid and proposes it IF above threshold. Writes nothing -
    a pure plan the caller reviews before anything is applied, same as
    reorganize_camera_roll.build_plan(). candidate_faces defaults to every
    face eligible for labeling at all (unlabeled, non-discarded, passes the
    quality filter) - the same pool recluster_faces() draws from."""
    close_after = conn is None
    conn = conn or init_db()
    centroids = compute_person_centroids(conn)
    if not centroids:
        if close_after:
            conn.close()
        return []

    if candidate_faces is None:
        candidate_faces = conn.execute(
            "SELECT face_vector_index, file_path, crop_filename, det_score, blur, width, height FROM faces "
            "WHERE person_id IS NULL AND discarded = 0 AND passes_filter = 1"
        ).fetchall()
        candidate_faces = [dict(zip(
            ("face_vector_index", "file_path", "crop_filename", "det_score", "blur", "width", "height"), r
        )) for r in candidate_faces]

    people_by_id = {p["person_id"]: p["name"] for p in list_people(conn)}
    if close_after:
        conn.close()

    embeddings = np.load(get_face_embeddings_path())
    person_ids = list(centroids.keys())
    centroid_matrix = np.stack([centroids[pid] for pid in person_ids])

    plan = []
    for face in candidate_faces:
        similarities = centroid_matrix @ embeddings[face["face_vector_index"]]
        best_idx = int(np.argmax(similarities))
        best_score = float(similarities[best_idx])
        if best_score >= similarity_threshold:
            best_person_id = person_ids[best_idx]
            plan.append({
                **face,
                "proposed_person_id": best_person_id,
                "proposed_person_name": people_by_id.get(best_person_id, "?"),
                "similarity": best_score,
            })
    # Most-confident first, same convention as suggest_cluster_groupings() - with
    # MAX_FACES_TO_DISPLAY capping the UI's view, whatever gets cut off should be
    # the least-confident tail, not an arbitrary insertion-order tail.
    plan.sort(key=lambda m: -m["similarity"])
    return plan


def apply_matches(plan, accepted_face_vector_indices, conn=None):
    """Writes person_id only for the plan entries in accepted_face_vector_indices -
    the caller (main.py) decides what counts as accepted; this function itself
    doesn't care whether the UI's checkboxes default to checked or unchecked,
    it just never writes anything outside that explicit set."""
    close_after = conn is None
    conn = conn or init_db()
    accepted = set(accepted_face_vector_indices)
    to_apply = [m for m in plan if m["face_vector_index"] in accepted]
    conn.executemany(
        "UPDATE faces SET person_id = ?, cluster_id = NULL WHERE face_vector_index = ?",
        [(m["proposed_person_id"], m["face_vector_index"]) for m in to_apply],
    )
    conn.commit()
    if close_after:
        conn.close()
    return len(to_apply)


def suggest_person_for_cluster(cluster_id, similarity_threshold=DEFAULT_MATCH_THRESHOLD, conn=None):
    """For one pending cluster, finds the closest ALREADY-LABELED person by
    centroid similarity and suggests their name if above threshold - the
    cluster-level counterpart to propose_matches() (which works per-face).
    Meant to pre-fill the labeling UI's name box so confirming a cluster
    that's clearly an existing person is a glance-and-click rather than
    retyping their name - still just a suggestion, writes nothing, the
    actual label only happens if/when the user hits Confirm.

    Not meaningful for cluster_id -1 (the unclustered/noise bucket - a
    large mixed bag of many different people, not one identity to average
    into a single centroid) - callers should not call this for -1."""
    close_after = conn is None
    conn = conn or init_db()
    centroids = compute_person_centroids(conn)
    if not centroids:
        if close_after:
            conn.close()
        return None

    faces = get_faces_for_cluster(cluster_id, conn)
    people_by_id = {p["person_id"]: p["name"] for p in list_people(conn)}
    if close_after:
        conn.close()
    if not faces:
        return None

    embeddings = np.load(get_face_embeddings_path())
    indices = [f["face_vector_index"] for f in faces]
    mean = embeddings[indices].mean(axis=0)
    cluster_centroid = mean / np.linalg.norm(mean)

    person_ids = list(centroids.keys())
    sims = np.stack([centroids[pid] for pid in person_ids]) @ cluster_centroid
    best_idx = int(np.argmax(sims))
    best_score = float(sims[best_idx])
    if best_score < similarity_threshold:
        return None
    best_person_id = person_ids[best_idx]
    return {"person_id": best_person_id, "name": people_by_id.get(best_person_id, "?"), "similarity": best_score}


def suggest_people_for_all_clusters(similarity_threshold=DEFAULT_MATCH_THRESHOLD, conn=None):
    """Batch version of suggest_person_for_cluster() - computes centroids
    and loads embeddings ONCE and reuses them across every pending cluster,
    instead of each cluster redoing that setup independently. Needed for
    grouping the People tab's list by suggestion (main.py) - calling the
    single-cluster version in a loop over ~650 real clusters would redo the
    same expensive setup ~650 times, the same "fine alone, breaks at real
    scale" trap as the reclustering and display-cap bugs. Returns
    {cluster_id: {person_id, name, similarity}} - only clusters with a
    confident suggestion are included, real clusters only (never -1)."""
    close_after = conn is None
    conn = conn or init_db()
    centroids = compute_person_centroids(conn)
    if not centroids:
        if close_after:
            conn.close()
        return {}

    rows = conn.execute(
        "SELECT cluster_id, face_vector_index FROM faces "
        "WHERE cluster_id IS NOT NULL AND cluster_id != -1 AND person_id IS NULL "
        "AND discarded = 0 AND passes_filter = 1"
    ).fetchall()
    people_by_id = {p["person_id"]: p["name"] for p in list_people(conn)}
    if close_after:
        conn.close()
    if not rows:
        return {}

    cluster_face_indices = {}
    for cluster_id, face_vector_index in rows:
        cluster_face_indices.setdefault(cluster_id, []).append(face_vector_index)

    embeddings = np.load(get_face_embeddings_path())
    person_ids = list(centroids.keys())
    centroid_matrix = np.stack([centroids[pid] for pid in person_ids])

    suggestions = {}
    for cluster_id, indices in cluster_face_indices.items():
        mean = embeddings[indices].mean(axis=0)
        cluster_centroid = mean / np.linalg.norm(mean)
        sims = centroid_matrix @ cluster_centroid
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= similarity_threshold:
            best_person_id = person_ids[best_idx]
            suggestions[cluster_id] = {
                "person_id": best_person_id,
                "name": people_by_id.get(best_person_id, "?"),
                "similarity": best_score,
            }
    return suggestions


def analyze_match_score_separation(min_cluster_size=10, sample_cluster_count=30, conn=None):
    """Real-data estimate of whether an auto-apply tier for propose_matches()
    is safe, and where the line would sit - without needing anyone actually
    labeled yet. Nobody's been named, so there's no real ground truth - this
    uses existing HDBSCAN clusters as a stand-in: a large, confident cluster
    is HDBSCAN's own best guess at "one real person," which is good enough to
    measure against even before a human confirms it.

    For each sampled cluster (the biggest ones - a more reliable centroid):
    - "positive" scores: each member face's similarity to its own cluster's
      centroid, computed leave-one-out (excluding that face itself from the
      centroid) so a face isn't partly measured against its own contribution -
      this is what a genuine same-person match score looks like in practice.
    - "negative" scores: each member face's similarity to every OTHER sampled
      cluster's centroid - what a genuine different-person score looks like.

    Returns the raw arrays, writes nothing and decides nothing - the same
    plan-not-action split as propose_matches() itself. Where these two
    distributions overlap is exactly the risk zone for an auto-apply
    threshold: pick a threshold below that overlap and it's evidence-backed,
    not guessed."""
    close_after = conn is None
    conn = conn or init_db()

    rows = conn.execute(
        "SELECT cluster_id, COUNT(*) as n FROM faces "
        "WHERE cluster_id IS NOT NULL AND cluster_id != -1 AND person_id IS NULL "
        "AND discarded = 0 AND passes_filter = 1 "
        "GROUP BY cluster_id HAVING n >= ? ORDER BY n DESC LIMIT ?",
        (min_cluster_size, sample_cluster_count),
    ).fetchall()
    cluster_ids = [r[0] for r in rows]

    cluster_face_indices = {}
    for cluster_id in cluster_ids:
        face_rows = conn.execute(
            "SELECT face_vector_index FROM faces WHERE cluster_id = ? "
            "AND person_id IS NULL AND discarded = 0 AND passes_filter = 1",
            (cluster_id,),
        ).fetchall()
        cluster_face_indices[cluster_id] = [r[0] for r in face_rows]
    if close_after:
        conn.close()

    if len(cluster_ids) < 2:
        return {"positive": np.array([]), "negative": np.array([]), "cluster_ids": cluster_ids}

    embeddings = np.load(get_face_embeddings_path())
    full_centroids = {}
    for cluster_id, indices in cluster_face_indices.items():
        mean = embeddings[indices].mean(axis=0)
        full_centroids[cluster_id] = mean / np.linalg.norm(mean)

    positive, negative = [], []
    for cluster_id, indices in cluster_face_indices.items():
        cluster_embeddings = embeddings[indices]
        other_centroids = np.stack([c for cid, c in full_centroids.items() if cid != cluster_id])
        for i, face_idx in enumerate(indices):
            if len(indices) > 1:
                # Leave-one-out: this face's own contribution is excluded from the
                # centroid it's compared against, so the score isn't inflated by
                # measuring a face partly against itself.
                loo = np.delete(cluster_embeddings, i, axis=0).mean(axis=0)
                loo /= np.linalg.norm(loo)
                positive.append(float(embeddings[face_idx] @ loo))
            negative.extend((other_centroids @ embeddings[face_idx]).tolist())

    return {"positive": np.array(positive), "negative": np.array(negative), "cluster_ids": cluster_ids}


def build_match_threshold_report(min_cluster_size=10, sample_cluster_count=30, conn=None):
    """Per-face detail behind analyze_match_score_separation() - one record
    per face with its own-cluster (leave-one-out) score AND its single most-
    confusable OTHER cluster's score, plus crop filenames for both, so real
    boundary cases can be looked at directly rather than trusting aggregate
    percentiles alone (the same "open the real crops" habit that caught
    facenet-pytorch's cross-person merge in the original bake-off)."""
    close_after = conn is None
    conn = conn or init_db()

    rows = conn.execute(
        "SELECT cluster_id, COUNT(*) as n FROM faces "
        "WHERE cluster_id IS NOT NULL AND cluster_id != -1 AND person_id IS NULL "
        "AND discarded = 0 AND passes_filter = 1 "
        "GROUP BY cluster_id HAVING n >= ? ORDER BY n DESC LIMIT ?",
        (min_cluster_size, sample_cluster_count),
    ).fetchall()
    cluster_ids = [r[0] for r in rows]

    cluster_faces = {}
    for cluster_id in cluster_ids:
        face_rows = conn.execute(
            "SELECT face_vector_index, crop_filename FROM faces WHERE cluster_id = ? "
            "AND person_id IS NULL AND discarded = 0 AND passes_filter = 1",
            (cluster_id,),
        ).fetchall()
        cluster_faces[cluster_id] = [{"face_vector_index": r[0], "crop_filename": r[1]} for r in face_rows]
    if close_after:
        conn.close()

    if len(cluster_ids) < 2:
        return []

    embeddings = np.load(get_face_embeddings_path())
    full_centroids = {}
    for cluster_id, faces in cluster_faces.items():
        indices = [f["face_vector_index"] for f in faces]
        mean = embeddings[indices].mean(axis=0)
        full_centroids[cluster_id] = mean / np.linalg.norm(mean)
    representative_crop = {cid: faces[0]["crop_filename"] for cid, faces in cluster_faces.items()}

    records = []
    for cluster_id, faces in cluster_faces.items():
        indices = [f["face_vector_index"] for f in faces]
        cluster_embeddings = embeddings[indices]
        other_ids = [cid for cid in cluster_ids if cid != cluster_id]
        other_matrix = np.stack([full_centroids[cid] for cid in other_ids])
        for i, face in enumerate(faces):
            vec = embeddings[face["face_vector_index"]]
            positive_score = None
            if len(indices) > 1:
                loo = np.delete(cluster_embeddings, i, axis=0).mean(axis=0)
                loo /= np.linalg.norm(loo)
                positive_score = float(vec @ loo)
            sims = other_matrix @ vec
            worst_idx = int(np.argmax(sims))
            records.append({
                "face_vector_index": face["face_vector_index"],
                "crop_filename": face["crop_filename"],
                "cluster_id": cluster_id,
                "positive_score": positive_score,
                "negative_score": float(sims[worst_idx]),
                "negative_cluster_id": other_ids[worst_idx],
                "negative_crop_filename": representative_crop[other_ids[worst_idx]],
            })
    return records


def write_match_threshold_report(records, output_path, crops_rel="face_crops"):
    """Two sorted views, worst-first, mirroring face_bakeoff.py's
    diagnostics.html pattern: the lowest positive (own-person) scores - real
    risk of a true match getting missed - and the highest negative
    (different-person) scores - real risk of a false match slipping through
    an auto-apply tier. Each row pairs the query face's crop with a
    representative crop from whichever cluster it was compared against, so
    the two faces can actually be looked at side by side."""
    style = """
<style>
body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 2rem; }
h1 { font-size: 1.2rem; } h2 { font-size: 1rem; color: #9cf; margin-top: 2rem; }
.row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; background: #1c1c1c; padding: 6px; border-radius: 6px; }
.row img { width: 80px; height: 80px; object-fit: cover; border-radius: 4px; }
.arrow { color: #666; font-size: 1.2rem; }
.score { color: #9cf; font-size: 0.85rem; width: 70px; }
</style>
"""
    parts = [f'<!doctype html><html><head><meta charset="utf-8"><title>Match threshold report</title>{style}</head><body>']
    parts.append("<h1>Match threshold report</h1>")

    positives = sorted((r for r in records if r["positive_score"] is not None), key=lambda r: r["positive_score"])
    parts.append("<h2>Lowest own-person scores (risk of a real match being missed)</h2>")
    for r in positives[:40]:
        parts.append(
            f'<div class="row"><img src="{crops_rel}/{r["crop_filename"]}">'
            f'<span class="score">{r["positive_score"]:.3f}</span>'
            f'<span>cluster {r["cluster_id"]} vs. its own other members</span></div>'
        )

    negatives = sorted(records, key=lambda r: -r["negative_score"])
    parts.append("<h2>Highest different-person scores (risk of a false match slipping through)</h2>")
    for r in negatives[:40]:
        parts.append(
            f'<div class="row"><img src="{crops_rel}/{r["crop_filename"]}"><span class="arrow">vs</span>'
            f'<img src="{crops_rel}/{r["negative_crop_filename"]}">'
            f'<span class="score">{r["negative_score"]:.3f}</span>'
            f'<span>cluster {r["cluster_id"]} vs. cluster {r["negative_cluster_id"]}</span></div>'
        )

    parts.append("</body></html>")
    Path(output_path).write_text("".join(parts), encoding="utf-8")


def suggest_cluster_groupings(min_cluster_size=MIN_CLUSTER_SIZE, similarity_threshold=DEFAULT_MATCH_THRESHOLD, conn=None):
    """For every still-pending cluster, finds its single most-similar OTHER
    pending cluster by centroid similarity - purely informational, exactly
    the "many small clusters are actually the same person" pattern from the
    2026-08-20 diary note, surfaced as a hint rather than guessed at by eye.

    Deliberately writes and labels nothing - same plan-not-action posture as
    propose_matches(). No min_cluster_size floor beyond HDBSCAN's own
    (unlike analyze_match_score_separation's min_cluster_size=10, which
    needed bigger samples for reliable aggregate stats) - the whole point
    here is surfacing suggestions FOR the small clusters, so excluding them
    would defeat the purpose."""
    close_after = conn is None
    conn = conn or init_db()

    rows = conn.execute(
        "SELECT cluster_id, COUNT(*) as n FROM faces "
        "WHERE cluster_id IS NOT NULL AND cluster_id != -1 AND person_id IS NULL "
        "AND discarded = 0 AND passes_filter = 1 "
        "GROUP BY cluster_id HAVING n >= ?",
        (min_cluster_size,),
    ).fetchall()
    cluster_ids = [r[0] for r in rows]
    cluster_sizes = {r[0]: r[1] for r in rows}

    cluster_faces = {}
    for cluster_id in cluster_ids:
        face_rows = conn.execute(
            "SELECT face_vector_index, crop_filename FROM faces WHERE cluster_id = ? "
            "AND person_id IS NULL AND discarded = 0 AND passes_filter = 1",
            (cluster_id,),
        ).fetchall()
        cluster_faces[cluster_id] = [{"face_vector_index": r[0], "crop_filename": r[1]} for r in face_rows]
    if close_after:
        conn.close()

    if len(cluster_ids) < 2:
        return []

    embeddings = np.load(get_face_embeddings_path())
    centroids = {}
    representative_crop = {}
    for cluster_id, faces in cluster_faces.items():
        indices = [f["face_vector_index"] for f in faces]
        mean = embeddings[indices].mean(axis=0)
        centroids[cluster_id] = mean / np.linalg.norm(mean)
        representative_crop[cluster_id] = faces[0]["crop_filename"]

    suggestions = []
    for cluster_id in cluster_ids:
        other_ids = [cid for cid in cluster_ids if cid != cluster_id]
        other_matrix = np.stack([centroids[cid] for cid in other_ids])
        sims = other_matrix @ centroids[cluster_id]
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        if best_score >= similarity_threshold:
            suggestions.append({
                "cluster_id": cluster_id,
                "cluster_size": cluster_sizes[cluster_id],
                "cluster_crop": representative_crop[cluster_id],
                "suggested_cluster_id": other_ids[best_idx],
                "suggested_cluster_size": cluster_sizes[other_ids[best_idx]],
                "suggested_crop": representative_crop[other_ids[best_idx]],
                "similarity": best_score,
            })

    suggestions.sort(key=lambda s: -s["similarity"])
    return suggestions


def write_cluster_grouping_report(suggestions, output_path, crops_rel="face_crops"):
    """Most-confident suggestions first, each pairing a representative crop
    from the smaller cluster against its suggested match - a pure review
    aid, nothing here labels or merges anything."""
    style = """
<style>
body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 2rem; }
h1 { font-size: 1.2rem; } .stats { color: #aaa; margin-bottom: 1rem; }
.row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; background: #1c1c1c; padding: 6px; border-radius: 6px; }
.row img { width: 90px; height: 90px; object-fit: cover; border-radius: 4px; }
.arrow { color: #666; font-size: 1.2rem; }
.score { color: #9cf; font-size: 0.85rem; width: 70px; }
</style>
"""
    parts = [f'<!doctype html><html><head><meta charset="utf-8"><title>Cluster grouping suggestions</title>{style}</head><body>']
    parts.append("<h1>Cluster grouping suggestions</h1>")
    parts.append(f'<div class="stats">{len(suggestions)} suggested pairing(s), most confident first - '
                  f'nothing here has been applied, purely for review</div>')
    for s in suggestions:
        parts.append(
            f'<div class="row"><img src="{crops_rel}/{s["cluster_crop"]}">'
            f'<span class="arrow">â‰ˆ</span>'
            f'<img src="{crops_rel}/{s["suggested_crop"]}">'
            f'<span class="score">{s["similarity"]:.3f}</span>'
            f'<span>cluster {s["cluster_id"]} ({s["cluster_size"]} faces) '
            f'vs. cluster {s["suggested_cluster_id"]} ({s["suggested_cluster_size"]} faces)</span></div>'
        )
    parts.append("</body></html>")
    Path(output_path).write_text("".join(parts), encoding="utf-8")


def suggest_duplicate_people(similarity_threshold=DEFAULT_MATCH_THRESHOLD, conn=None):
    """Compares every labeled person's centroid against every OTHER labeled
    person's centroid, flagging pairs above the threshold as possibly the
    same real person split under two different labels - the person-level
    counterpart to suggest_cluster_groupings() (which does this for pending
    clusters, before anyone's named). Pure suggestion, changes nothing -
    same posture as every other suggest_* function in this module.

    Deliberately NOT based on how many photos two people share (2026-08-24):
    "Fraser's Mum" and "Mum" shared 100% of their photos (every one of
    Fraser's Mum's 5 files also had the user's own Mum in it) and were
    checked and confirmed to be genuinely different people who are just
    often photographed together - real evidence that file-overlap measures
    "photographed together," not "same face." Centroid similarity measures
    whether the two people's FACES actually look alike instead, which is
    what an accidental split-identity would actually produce."""
    close_after = conn is None
    conn = conn or init_db()
    centroids = compute_person_centroids(conn)
    people_by_id = {p["person_id"]: p for p in list_people(conn)}
    person_crop = dict(conn.execute(
        "SELECT person_id, crop_filename FROM faces WHERE person_id IS NOT NULL GROUP BY person_id"
    ).fetchall())
    if close_after:
        conn.close()

    person_ids = list(centroids.keys())
    if len(person_ids) < 2:
        return []

    matrix = np.stack([centroids[pid] for pid in person_ids])
    sims = matrix @ matrix.T

    suggestions = []
    for i, pid_a in enumerate(person_ids):
        for j in range(i + 1, len(person_ids)):
            pid_b = person_ids[j]
            score = float(sims[i, j])
            if score >= similarity_threshold:
                suggestions.append({
                    "person_a_id": pid_a,
                    "person_a_name": people_by_id[pid_a]["name"],
                    "person_a_crop": person_crop.get(pid_a),
                    "person_b_id": pid_b,
                    "person_b_name": people_by_id[pid_b]["name"],
                    "person_b_crop": person_crop.get(pid_b),
                    "similarity": score,
                })

    suggestions.sort(key=lambda s: -s["similarity"])
    return suggestions


def write_duplicate_people_report(suggestions, output_path, crops_rel="face_crops"):
    """Most-confident pairs first, each pairing a representative crop from
    both people so a real visual check can confirm or rule out a split
    identity before anyone merges anything - pure review aid, nothing here
    labels, merges, or renames. See suggest_duplicate_people()'s docstring
    for why file-overlap alone isn't trustworthy evidence here - always
    look at the actual crops, the same "open the real crops" habit that's
    caught real mistakes elsewhere in this project."""
    style = """
<style>
body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 2rem; }
h1 { font-size: 1.2rem; } .stats { color: #aaa; margin-bottom: 1rem; }
.row { display: flex; align-items: center; gap: 10px; margin-bottom: 8px; background: #1c1c1c; padding: 6px; border-radius: 6px; }
.row img { width: 90px; height: 90px; object-fit: cover; border-radius: 4px; }
.arrow { color: #666; font-size: 1.2rem; }
.score { color: #9cf; font-size: 0.85rem; width: 70px; }
.name { min-width: 120px; }
</style>
"""
    parts = [f'<!doctype html><html><head><meta charset="utf-8"><title>Possible duplicate people</title>{style}</head><body>']
    parts.append("<h1>Possible duplicate people</h1>")
    parts.append(
        f'<div class="stats">{len(suggestions)} suggested pairing(s), most confident first - based on how similar '
        f'their faces look, NOT how many photos they share; nothing here has been merged, purely for review</div>'
    )
    for s in suggestions:
        parts.append(
            f'<div class="row"><img src="{crops_rel}/{s["person_a_crop"]}">'
            f'<span class="name">{s["person_a_name"]}</span>'
            f'<span class="arrow">â‰ˆ</span>'
            f'<img src="{crops_rel}/{s["person_b_crop"]}">'
            f'<span class="name">{s["person_b_name"]}</span>'
            f'<span class="score">{s["similarity"]:.3f}</span></div>'
        )
    parts.append("</body></html>")
    Path(output_path).write_text("".join(parts), encoding="utf-8")


def build_candidate_suggestions_report(top_k=5, min_similarity=0.3, conn=None):
    """For every still-pending cluster, the FULL ranked shortlist of candidate
    labeled people - not just the single best. suggest_person_for_cluster()
    (and the UI it feeds) only ever surfaces its #1 pick, so there's no way
    to see how close a runner-up came, or whether the top pick's margin over
    #2 is actually comfortable versus a coin-flip. Review-only, same plan-
    not-action posture as everything else here - writes and labels nothing.

    min_similarity is a low floor (well below DEFAULT_MATCH_THRESHOLD) just to
    cut near-zero noise candidates, not a suggestion cutoff - the point is
    seeing candidates that DIDN'T clear the live suggestion bar too, so a
    real close-second isn't hidden just because it's under 0.5. Excludes
    cluster -1 (unclustered noise), same reasoning as suggest_person_for_
    cluster() - no single identity to average into one centroid."""
    close_after = conn is None
    conn = conn or init_db()
    centroids = compute_person_centroids(conn)
    if not centroids:
        if close_after:
            conn.close()
        return []

    rows = conn.execute(
        "SELECT cluster_id, COUNT(*) as n FROM faces "
        "WHERE cluster_id IS NOT NULL AND cluster_id != -1 AND person_id IS NULL "
        "AND discarded = 0 AND passes_filter = 1 "
        "GROUP BY cluster_id"
    ).fetchall()
    cluster_ids = [r[0] for r in rows]
    cluster_sizes = {r[0]: r[1] for r in rows}

    cluster_faces = {}
    for cluster_id in cluster_ids:
        face_rows = conn.execute(
            "SELECT face_vector_index, crop_filename FROM faces WHERE cluster_id = ? "
            "AND person_id IS NULL AND discarded = 0 AND passes_filter = 1",
            (cluster_id,),
        ).fetchall()
        cluster_faces[cluster_id] = [{"face_vector_index": r[0], "crop_filename": r[1]} for r in face_rows]

    people_by_id = {p["person_id"]: p["name"] for p in list_people(conn)}
    # One representative crop per labeled person - GROUP BY with an unaggregated
    # column picks an arbitrary row per group, same informality as the
    # representative_crop pattern already used for clusters elsewhere in this file.
    person_crop = dict(conn.execute(
        "SELECT person_id, crop_filename FROM faces WHERE person_id IS NOT NULL GROUP BY person_id"
    ).fetchall())
    if close_after:
        conn.close()
    if not cluster_ids:
        return []

    embeddings = np.load(get_face_embeddings_path())
    person_ids = list(centroids.keys())
    centroid_matrix = np.stack([centroids[pid] for pid in person_ids])

    records = []
    for cluster_id, faces in cluster_faces.items():
        indices = [f["face_vector_index"] for f in faces]
        mean = embeddings[indices].mean(axis=0)
        cluster_centroid = mean / np.linalg.norm(mean)
        sims = centroid_matrix @ cluster_centroid
        ranked = np.argsort(-sims)[:top_k]
        candidates = [
            {
                "person_id": person_ids[i],
                "name": people_by_id.get(person_ids[i], "?"),
                "similarity": float(sims[i]),
                "crop_filename": person_crop.get(person_ids[i]),
            }
            for i in ranked if sims[i] >= min_similarity
        ]
        if candidates:
            records.append({
                "cluster_id": cluster_id,
                "cluster_size": cluster_sizes[cluster_id],
                "cluster_crop": faces[0]["crop_filename"],
                "candidates": candidates,
            })

    records.sort(key=lambda r: -r["candidates"][0]["similarity"])
    return records


def write_candidate_suggestions_report(records, output_path, crops_rel="face_crops"):
    """One block per pending cluster, most-confident top-candidate first -
    shows the FULL shortlist (build_candidate_suggestions_report), not just
    the single suggestion the live UI pre-fills, so a close runner-up or a
    thin margin between #1 and #2 is visible at a glance instead of hidden."""
    style = """
<style>
body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 2rem; }
h1 { font-size: 1.2rem; } .stats { color: #aaa; margin-bottom: 1rem; }
.cluster { display: flex; align-items: center; gap: 14px; margin-bottom: 10px; background: #1c1c1c; padding: 8px; border-radius: 6px; }
.cluster img.main { width: 90px; height: 90px; object-fit: cover; border-radius: 4px; }
.cluster-label { color: #aaa; font-size: 0.8rem; width: 90px; }
.candidates { display: flex; gap: 10px; flex-wrap: wrap; }
.candidate { display: flex; flex-direction: column; align-items: center; font-size: 0.8rem; }
.candidate img { width: 60px; height: 60px; object-fit: cover; border-radius: 4px; }
.candidate .score { color: #9cf; }
.candidate.top img { outline: 2px solid #4caf50; }
</style>
"""
    parts = [f'<!doctype html><html><head><meta charset="utf-8"><title>Candidate suggestions report</title>{style}</head><body>']
    parts.append("<h1>Candidate suggestions report</h1>")
    parts.append(
        f'<div class="stats">{len(records)} pending cluster(s) with at least one candidate, most confident '
        f'top pick first - green outline is what the live People tab pre-fills; nothing here labels or writes anything</div>'
    )
    for r in records:
        candidate_html = "".join(
            f'<div class="candidate{" top" if i == 0 else ""}">'
            f'<img src="{crops_rel}/{c["crop_filename"]}">'
            f'<span>{c["name"]}</span><span class="score">{c["similarity"]:.3f}</span></div>'
            for i, c in enumerate(r["candidates"])
        )
        parts.append(
            f'<div class="cluster"><img class="main" src="{crops_rel}/{r["cluster_crop"]}">'
            f'<span class="cluster-label">cluster {r["cluster_id"]}<br>{r["cluster_size"]} faces</span>'
            f'<div class="candidates">{candidate_html}</div></div>'
        )
    parts.append("</body></html>")
    Path(output_path).write_text("".join(parts), encoding="utf-8")


def propose_cluster_labels(min_similarity=0.3, min_margin=0.08, top_k=5, conn=None):
    """Cluster-level counterpart to propose_matches() - proposes labeling
    each pending cluster as its best-candidate labeled person, built on
    build_candidate_suggestions_report()'s same ranked-shortlist logic.
    Excludes any cluster whose top two candidates are within min_margin of
    each other - a genuine close call that score alone shouldn't resolve
    (2026-08-21 request: batch-accept everything confident, but hold out the
    close calls for a human). Writes nothing - a pure plan, same posture as
    propose_matches(); each entry keeps its full candidate shortlist (not
    just the winner) so the same records can feed
    write_candidate_suggestions_report() for a visual review before
    anything is applied."""
    records = build_candidate_suggestions_report(top_k=top_k, min_similarity=min_similarity, conn=conn)
    plan = []
    for r in records:
        top = r["candidates"][0]
        if len(r["candidates"]) > 1 and (top["similarity"] - r["candidates"][1]["similarity"]) < min_margin:
            continue
        plan.append(r)
    return plan


def apply_cluster_labels(plan, accepted_cluster_ids, conn=None):
    """Writes person_id (the top candidate) for every pending face in each
    accepted cluster - the only function here that writes, and only for
    cluster_ids the caller explicitly accepted from propose_cluster_labels()'s
    plan. Re-fetches each cluster's current pending faces at apply time
    (get_faces_for_cluster(), the whole cluster) rather than trusting a
    stale list from when the plan was built - safe here since this is a
    batch/script operation building from real cluster_ids, not a UI display
    that could have been capped (see label_faces()'s docstring for why that
    distinction matters elsewhere)."""
    close_after = conn is None
    conn = conn or init_db()
    accepted = set(accepted_cluster_ids)
    total = 0
    for entry in plan:
        if entry["cluster_id"] not in accepted:
            continue
        person_id = entry["candidates"][0]["person_id"]
        faces = get_faces_for_cluster(entry["cluster_id"], conn)
        conn.executemany(
            "UPDATE faces SET person_id = ?, cluster_id = NULL WHERE face_vector_index = ?",
            [(person_id, f["face_vector_index"]) for f in faces],
        )
        total += len(faces)
    conn.commit()
    if close_after:
        conn.close()
    return total


if __name__ == "__main__":
    def report(i, total, name):
        print(f"[{i}/{total}] {name}")

    new_files, new_faces = build_face_index(progress_callback=report)
    print(f"Face-indexed {new_files} new files, {new_faces} new faces.")
    cluster_count = recluster_faces()
    print(f"Reclustered - {cluster_count} clusters among unlabeled faces.")
