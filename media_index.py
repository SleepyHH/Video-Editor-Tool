"""
Mode A engine, indexing half: builds a local, incremental CLIP embedding
index over the media library so media_search.py can match text queries
against it. No GUI - designed to be called from a future main.py tab later
without needing a rewrite.

Videos aren't understood as video directly - CLIP only reads still images,
so each video gets sampled into frames (one every FRAME_INTERVAL_SECONDS)
and each frame is embedded separately. A search "hits" a video if any of
its frames scores well against the query.

Only already-downloaded/local files are indexed. Cloud-only items are
skipped, not specially handled - re-running this later after downloading
more of the library picks them up automatically via the same incremental
skip-if-unchanged logic (keyed on file path + mtime).
"""
import gc
import platform
import re
import sqlite3
import subprocess
import tempfile
from pathlib import Path

import numpy as np
import open_clip
import pillow_heif
import torch
from PIL import ExifTags, Image

from config import (
    EXTERNAL_DRIVE_LABEL, IGNORED_DIR_NAMES, VIDEO_EXTENSIONS,
    find_volume_by_label, get_os_profile, resolve_portable_path, to_portable_path, walk_media_files,
)

pillow_heif.register_heif_opener()  # lets PIL.Image.open() read iPhone .heic/.heif photos

# --- Shared index location, added 05-08-2026 ---
# Lives ON the external drive (see EXTERNAL_DRIVE_LABEL below) when it's connected, so the
# Mac and the PC read/write the SAME index instead of each building its own copy from
# scratch - the whole point being to only pay the embedding-compute cost once, not once per
# machine. Falls back to the original ~/Documents location if the drive isn't connected,
# so the tool still works (against whatever's already indexed) without it plugged in.
# Computed fresh via functions, not fixed module-level constants, since drive availability
# can change between runs.
def get_index_dir():
    drive_root = find_volume_by_label(EXTERNAL_DRIVE_LABEL)
    if drive_root:
        return drive_root / "HuysVideoEditor_SearchIndex"
    return Path.home() / "Documents" / "HuysVideoEditor_SearchIndex"


def get_embeddings_path():
    return get_index_dir() / "embeddings.npy"


def get_db_path():
    return get_index_dir() / "index.sqlite3"

# Tunable - to be calibrated against real accuracy testing, not guessed upfront.
FRAME_INTERVAL_SECONDS = 1

# Committed 04-08-2026 after real research (not benchmark-shopping): DataComp-1B's curated
# training recipe beats plain LAION-2B on the same architecture (79.2% vs 75.3% zero-shot
# ImageNet), and CLIP's contrastive objective suits retrieval/compositional queries better
# than SigLIP's classification-optimized sigmoid loss - directly relevant to this project's
# "green bag"-style compound queries. See diary 03/04-08-2026 for the full reasoning.
CLIP_MODEL_NAME = "ViT-L-14"
CLIP_PRETRAINED = "datacomp_xl_s13b_b90k"
EMBEDDING_DIM = 768  # matches ViT-L-14's output width (confirmed at runtime before reindexing)

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".heic"}

# TEMP_TEST_FOLDERS (~/Documents/Photos back up) removed 05-08-2026: confirmed via a full
# recursive filename comparison that all 4,502 of its files already exist inside the "Camera
# Roll" folder on the external drive below (which also has ~13k MORE files on top of that) -
# indexing both would have meant double-embedding the same photos under two different paths.

# EXTERNAL_DRIVE_LABEL, IGNORED_DIR_NAMES, VIDEO_EXTENSIONS moved to config.py 08-08-2026 -
# main.py's media list now scans the same drive under the same rules, so both need one shared
# definition rather than two that could quietly drift apart.


def _is_windows_cloud_placeholder(path):
    """True if Windows has this file marked offline/cloud-only (an
    iCloud-for-Windows placeholder that hasn't actually been downloaded
    yet, despite showing up in a normal directory listing with a real-
    looking size). Confirmed live 06-08-2026: Image.open() on a file like
    this doesn't raise - it blocks indefinitely waiting on an on-demand
    cloud recall that can stall forever, silently hanging the whole
    indexing run with no exception to catch. Filtered out here before ever
    being opened, consistent with this project's existing scope (only
    already-downloaded files are indexed - see module docstring) - a
    placeholder is exactly a cloud-only file that hasn't downloaded yet,
    the config/cloud-mode assumption that it already had turned out not to
    always hold. No-op (returns False) off Windows."""
    if platform.system() != "Windows":
        return False
    import ctypes
    FILE_ATTRIBUTE_OFFLINE = 0x1000
    FILE_ATTRIBUTE_RECALL_ON_OPEN = 0x40000
    FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS = 0x400000
    attrs = ctypes.windll.kernel32.GetFileAttributesW(str(path))
    if attrs == 0xFFFFFFFF:  # INVALID_FILE_ATTRIBUTES - let normal open()/stat() surface the real error
        return False
    return bool(attrs & (FILE_ATTRIBUTE_OFFLINE | FILE_ATTRIBUTE_RECALL_ON_OPEN | FILE_ATTRIBUTE_RECALL_ON_DATA_ACCESS))


def _walk_media_files(folder):
    """Recursively finds files under folder, skipping hidden/housekeeping
    directories and AppleDouble sidecar files (config.walk_media_files) plus
    Windows cloud-only placeholders. Unlike the top-level folders above, a
    real external media drive is expected to have real subfolder structure
    (by date/project/etc.), so this needs to actually recurse rather than a
    flat iterdir()."""
    for path in walk_media_files(folder):
        if not _is_windows_cloud_placeholder(path):
            yield path

_model_cache = {}


def get_device():
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def load_clip_model():
    """Loads once per process - downloads weights on first use (~350MB)."""
    if "model" not in _model_cache:
        device = get_device()
        model, _, preprocess = open_clip.create_model_and_transforms(CLIP_MODEL_NAME, pretrained=CLIP_PRETRAINED)
        tokenizer = open_clip.get_tokenizer(CLIP_MODEL_NAME)
        model.eval().to(device)
        _model_cache.update(model=model, preprocess=preprocess, tokenizer=tokenizer, device=device)
    c = _model_cache
    return c["model"], c["preprocess"], c["tokenizer"], c["device"]


def unload_clip_model():
    """Frees the CLIP model from memory/accelerator. Safe no-op if nothing's
    loaded. A caller that touches load_clip_model() again afterward just pays
    the load cost again.

    Intentionally NOT wired into load_clip_model() itself or any idle-timer
    here - build_index() calls load_clip_model() exactly once and holds that
    reference across a run that can take hours, so an idle timer keyed off
    load_clip_model() calls would fire mid-run and force a wasteful (and on
    an 8GB card, risky - see media-search-windows-first-full-index) second
    load alongside the still-live first one. Idle-unload policy belongs to
    whichever caller is actually long-lived and idle-prone (the search
    server), not to this shared loader."""
    if "model" not in _model_cache:
        return
    device = _model_cache.get("device")
    _model_cache.clear()
    gc.collect()
    if device == "mps":
        torch.mps.empty_cache()
    elif device == "cuda":
        torch.cuda.empty_cache()


def init_db(db_path=None):
    db_path = db_path or get_db_path()
    db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    # Needed on exFAT specifically (confirmed live, 05-08-2026): this Mac's exFAT driver
    # (Apple's newer FSKit-based one, "noowners") doesn't fully support the POSIX file
    # locking SQLite's default mode relies on, causing spurious "readonly database" errors.
    # EXCLUSIVE mode sidesteps the repeated per-operation locking that was failing - safe
    # here since this is a single-writer tool and the drive is only ever plugged into one
    # machine at a time anyway, never genuinely concurrent.
    conn.execute("PRAGMA locking_mode=EXCLUSIVE")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS indexed_files (
            file_path TEXT PRIMARY KEY,
            mtime REAL NOT NULL,
            date_taken TEXT,
            lat REAL,
            lon REAL
        )
    """)
    # ALTER TABLE for anyone who indexed before EXIF support existed - keeps their
    # existing embeddings/rows intact rather than forcing a from-scratch rebuild.
    existing_cols = {row[1] for row in conn.execute("PRAGMA table_info(indexed_files)")}
    for col, col_type in [("date_taken", "TEXT"), ("lat", "REAL"), ("lon", "REAL"), ("frame_interval_seconds", "INTEGER")]:
        if col not in existing_cols:
            conn.execute(f"ALTER TABLE indexed_files ADD COLUMN {col} {col_type}")
    conn.execute("""
        CREATE TABLE IF NOT EXISTS items (
            vector_index INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            media_type TEXT NOT NULL,
            timestamp_seconds REAL
        )
    """)

    # Face recognition schema, added 2026-08-19 - a separate, independent pipeline from the
    # CLIP `items` table above (different model, different embedding dimension, its own
    # face_embeddings.npy), not layered onto it.
    conn.execute("""
        CREATE TABLE IF NOT EXISTS people (
            person_id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL UNIQUE
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS faces (
            face_vector_index INTEGER PRIMARY KEY,
            file_path TEXT NOT NULL,
            media_type TEXT NOT NULL,
            timestamp_seconds REAL,
            timestamp_end_seconds REAL,
            bbox_x1 INTEGER, bbox_y1 INTEGER, bbox_x2 INTEGER, bbox_y2 INTEGER,
            det_score REAL NOT NULL,
            blur REAL NOT NULL,
            width INTEGER NOT NULL,
            height INTEGER NOT NULL,
            passes_filter INTEGER NOT NULL,
            cluster_id INTEGER,
            person_id INTEGER,
            discarded INTEGER NOT NULL DEFAULT 0,
            crop_filename TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS face_indexed_files (
            file_path TEXT PRIMARY KEY,
            mtime REAL NOT NULL
        )
    """)
    conn.commit()
    return conn


def _dms_to_decimal(dms, ref):
    degrees, minutes, seconds = dms
    decimal = float(degrees) + float(minutes) / 60 + float(seconds) / 3600
    return -decimal if ref in ("S", "W") else decimal


def extract_image_metadata(path):
    """(date_taken_iso, lat, lon) from EXIF - any can be None (most photos
    have a date, far fewer have GPS, since that needs location services on
    at capture time)."""
    try:
        exif = Image.open(path).getexif()
        if not exif:
            return None, None, None

        date_taken = None
        raw_date = exif.get_ifd(ExifTags.IFD.Exif).get(36867) or exif.get(306)  # DateTimeOriginal, else DateTime
        if raw_date:
            date_taken = raw_date.replace(":", "-", 2)  # "2026:07:15 14:32:10" -> "2026-07-15 14:32:10"

        lat = lon = None
        gps_ifd = exif.get_ifd(ExifTags.IFD.GPSInfo)
        if gps_ifd:
            if gps_ifd.get(2) and gps_ifd.get(1):
                lat = _dms_to_decimal(gps_ifd[2], gps_ifd[1])
            if gps_ifd.get(4) and gps_ifd.get(3):
                lon = _dms_to_decimal(gps_ifd[4], gps_ifd[3])

        return date_taken, lat, lon
    except Exception:
        return None, None, None


def extract_video_metadata(ffmpeg_binary, path):
    """Best-effort creation-time via ffmpeg's own metadata dump.

    Prefers `com.apple.quicktime.creationdate` (includes an explicit UTC
    offset, e.g. "2024-06-22T01:34:19-0700") over the plain `creation_time`
    tag - confirmed live this was a real bug, not a style choice: QuickTime/
    MP4 containers store the plain tag in UTC, and reading it as if it were
    already local time was putting every iPhone video's date off by exactly
    the local UTC offset (7 hours, matching Pacific Daylight Time), verified
    against a paired Live Photo's correctly-local EXIF time matching
    `com.apple.quicktime.creationdate` to the second. Non-Apple-recorded
    videos without that field fall back to the plain (UTC) tag, which can
    still carry the same offset error - a known remaining gap, not silently
    treated as fully solved.

    GPS is skipped for video (v1) - far less consistently embedded than in
    photos, and the location-tag format varies enough not to be worth the
    parsing complexity yet."""
    try:
        result = subprocess.run([ffmpeg_binary, "-i", str(path)], capture_output=True, text=True, timeout=120)

        apple_match = re.search(r"com\.apple\.quicktime\.creationdate\s*:\s*(\d{4}-\d{2}-\d{2})T(\d{2}:\d{2}:\d{2})", result.stderr)
        if apple_match:
            return f"{apple_match.group(1)} {apple_match.group(2)}", None, None

        utc_match = re.search(r"creation_time\s*:\s*(\S+)", result.stderr)
        if utc_match:
            return utc_match.group(1).split(".")[0].replace("T", " "), None, None
    except Exception:
        pass
    return None, None, None


def list_library_media(profile):
    """Headless enumeration of the local/downloaded photo+video library -
    same OS branching as main.py's load_iphone_photos, but returns plain
    Path objects instead of populating a Qt list, and only ever includes
    files that already exist on disk (see module docstring re: cloud-only)."""
    results = []
    mode = profile["pipeline_mode"]

    if mode == "native_db":
        import osxphotos
        photosdb = osxphotos.PhotosDB()
        for item in photosdb.photos():
            path = item.path or item.path_edited
            if path and Path(path).exists():
                results.append(Path(path))
    elif mode == "cloud":
        folder = profile["photos_path"]
        if folder and folder.exists():
            # iterdir() + suffix filtering below, NOT folder.glob(f"*{ext}") - glob is
            # case-sensitive on a case-sensitive filesystem, and iPhone-exported files are
            # inconsistently cased (IMG_1234.MOV vs img_1234.mov) - confirmed live: a fixed-
            # case glob pattern silently missed over half the videos in a real test folder.
            results.extend(
                p for p in folder.iterdir()
                if p.is_file() and not _is_windows_cloud_placeholder(p)
            )

    # External media drive (permanent source, not test data - see EXTERNAL_DRIVE_LABEL above)
    external_drive = find_volume_by_label(EXTERNAL_DRIVE_LABEL)
    if external_drive:
        results.extend(_walk_media_files(external_drive))

    return [p for p in results if p.suffix.lower() in VIDEO_EXTENSIONS | IMAGE_EXTENSIONS]


def extract_video_frames(ffmpeg_binary, video_path, interval_seconds, hwaccel=None):
    """One frame every `interval_seconds` of video, into a temp dir.
    Returns (list of (frame_path, timestamp_seconds), temp_dir_to_clean_up).

    hwaccel (e.g. "cuda" - see config.get_os_profile) offloads video decode
    to the GPU. Falls back to a plain software-decode retry if the
    accelerated path fails on a given file - hw decode isn't guaranteed for
    every codec/pixel-format edge case, and this library already has some
    source files ffmpeg struggles with regardless of decode path."""
    tmp_dir = Path(tempfile.mkdtemp(prefix="huys_search_frames_"))
    pattern = tmp_dir / "frame_%05d.jpg"

    def run(use_hwaccel):
        cmd = [ffmpeg_binary, "-y"]
        if use_hwaccel:
            cmd += ["-hwaccel", hwaccel]
        cmd += [
            "-i", str(video_path),
            "-vf", f"fps=1/{interval_seconds}",
            "-qscale:v", "4",
            str(pattern),
        ]
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True, timeout=300)

    try:
        run(use_hwaccel=bool(hwaccel))
    except subprocess.CalledProcessError:
        if not hwaccel:
            raise
        run(use_hwaccel=False)

    frames = [
        (frame_path, i * interval_seconds)
        for i, frame_path in enumerate(sorted(tmp_dir.glob("frame_*.jpg")))
    ]
    return frames, tmp_dir


EMBED_BATCH_SIZE = 32


def embed_images(image_paths, model, preprocess, device, batch_size=EMBED_BATCH_SIZE):
    """Batch-embeds image file paths into an (N, EMBEDDING_DIM) L2-normalized
    float32 array. Processes in fixed-size chunks rather than one giant
    batch - confirmed live 06-08-2026: this used to batch an entire
    video's frames into a single forward pass with no cap, and at the
    current 1s sampling rate a long/high-res video produced a batch too
    large for the 8GB card, triggering a genuine CUDA out-of-memory error.
    That then corrupted the CUDA context for the rest of the process -
    every subsequent file failed identically until the process was
    restarted, silently wasting real time without ever crashing loudly."""
    all_features = []
    for i in range(0, len(image_paths), batch_size):
        chunk = image_paths[i:i + batch_size]
        tensors = [preprocess(Image.open(p).convert("RGB")) for p in chunk]
        batch = torch.stack(tensors).to(device)
        with torch.no_grad():
            features = model.encode_image(batch)
            features = features / features.norm(dim=-1, keepdim=True)
        all_features.append(features.cpu().numpy().astype("float32"))
    return np.concatenate(all_features, axis=0)


def build_index(profile=None, progress_callback=None, flush_every=50):
    """Incrementally indexes the local library: skips files already indexed
    (unchanged path + mtime), embeds new/changed ones, appends to the
    on-disk embeddings array + sqlite metadata. Safe to re-run any time,
    e.g. after downloading more of the library from iCloud.

    Flushes both the embeddings array AND the sqlite commit together every
    `flush_every` files, rather than committing sqlite per-file but only
    saving embeddings once at the very end (the original design - a real
    bug found 05-08-2026: an interrupted run would leave sqlite claiming
    files were indexed while their vectors were never actually written to
    disk, and would lose the *entire* run's progress on top of that).
    Bounding it to flush_every keeps any interruption's damage small - at
    most that many files get silently redone on the next run, since they
    were never committed as "done" in the first place."""
    profile = profile or get_os_profile()
    conn = init_db()
    model, preprocess, _, device = load_clip_model()
    embeddings_path = get_embeddings_path()

    already_indexed = dict(conn.execute("SELECT file_path, mtime FROM indexed_files"))
    library = list_library_media(profile)

    embeddings = np.load(embeddings_path) if embeddings_path.exists() else np.zeros((0, EMBEDDING_DIM), dtype="float32")
    next_vector_index = embeddings.shape[0]
    new_vectors = []
    total_new = 0
    files_since_flush = 0

    def flush():
        nonlocal embeddings, new_vectors
        if new_vectors:
            embeddings = np.vstack([embeddings, np.array(new_vectors, dtype="float32")])
            np.save(embeddings_path, embeddings)
            new_vectors = []
        conn.commit()

    for i, path in enumerate(library):
        # Stored/looked-up under its portable form (see to_portable_path) if it's on the
        # shared drive, so the index stays valid regardless of which machine reads it -
        # the real `path` below is still used for every actual file operation, only the
        # database representation changes.
        storage_path = to_portable_path(str(path), EXTERNAL_DRIVE_LABEL)
        mtime = path.stat().st_mtime
        if already_indexed.get(storage_path) == mtime:
            continue

        if progress_callback:
            progress_callback(i + 1, len(library), path.name)

        try:
            frame_interval = None
            if path.suffix.lower() in VIDEO_EXTENSIONS:
                frame_interval = FRAME_INTERVAL_SECONDS
                date_taken, lat, lon = extract_video_metadata(profile["ffmpeg_binary"], path)
                frames, tmp_dir = extract_video_frames(
                    profile["ffmpeg_binary"], path, FRAME_INTERVAL_SECONDS,
                    hwaccel=profile.get("ffmpeg_hwaccel"),
                )
                if frames:
                    vectors = embed_images([f for f, _ in frames], model, preprocess, device)
                    for (_, ts), vec in zip(frames, vectors):
                        new_vectors.append(vec)
                        conn.execute(
                            "INSERT INTO items (vector_index, file_path, media_type, timestamp_seconds) "
                            "VALUES (?, ?, 'video', ?)",
                            (next_vector_index, storage_path, ts),
                        )
                        next_vector_index += 1
                        total_new += 1
                for f in tmp_dir.glob("*"):
                    f.unlink()
                tmp_dir.rmdir()
            else:
                date_taken, lat, lon = extract_image_metadata(path)
                vec = embed_images([path], model, preprocess, device)[0]
                new_vectors.append(vec)
                conn.execute(
                    "INSERT INTO items (vector_index, file_path, media_type, timestamp_seconds) "
                    "VALUES (?, ?, 'image', NULL)",
                    (next_vector_index, storage_path),
                )
                next_vector_index += 1
                total_new += 1

            conn.execute(
                "INSERT OR REPLACE INTO indexed_files (file_path, mtime, date_taken, lat, lon, frame_interval_seconds) "
                "VALUES (?, ?, ?, ?, ?, ?)",
                (storage_path, mtime, date_taken, lat, lon, frame_interval),
            )
        except Exception as e:
            print(f"Skipped {path.name}: {e}")
            continue

        files_since_flush += 1
        if files_since_flush >= flush_every:
            flush()
            files_since_flush = 0

    flush()  # final partial batch
    conn.close()
    return total_new


def _delete_and_renumber(conn, stale_paths):
    """Shared by force_reembed_stale_videos and prune_missing_files: deletes
    the given file_paths' rows from items/indexed_files, then compacts
    embeddings.npy and renumbers the surviving items' vector_index to match.

    This isn't just tidiness: vector_index is a direct row-position pointer
    into the embeddings array (see media_search.search's `scores[vector_index]`),
    so deleting rows without renumbering the rest would leave every later
    lookup pointing at the wrong vector. Caller owns the connection (commit/close)."""
    placeholders = ",".join("?" * len(stale_paths))
    stale_vector_indices = {row[0] for row in conn.execute(
        f"SELECT vector_index FROM items WHERE file_path IN ({placeholders})", stale_paths
    )}
    conn.execute(f"DELETE FROM items WHERE file_path IN ({placeholders})", stale_paths)
    conn.execute(f"DELETE FROM indexed_files WHERE file_path IN ({placeholders})", stale_paths)

    embeddings_path = get_embeddings_path()
    embeddings = np.load(embeddings_path)
    keep_mask = np.ones(len(embeddings), dtype=bool)
    keep_mask[list(stale_vector_indices)] = False

    # Ascending order guarantees each new_vi is freshly allocated (never
    # collides with an already-updated row) and always <= its old_vi (never
    # collides with a not-yet-updated row still holding its original,
    # larger value) - safe against the PRIMARY KEY uniqueness constraint
    # without needing a temporary offset or a table rebuild.
    old_to_new = {}
    new_vi = 0
    for old_vi in range(len(embeddings)):
        if keep_mask[old_vi]:
            old_to_new[old_vi] = new_vi
            new_vi += 1

    remaining = conn.execute("SELECT vector_index FROM items ORDER BY vector_index ASC").fetchall()
    for (old_vi,) in remaining:
        if old_to_new[old_vi] != old_vi:
            conn.execute("UPDATE items SET vector_index = ? WHERE vector_index = ?", (old_to_new[old_vi], old_vi))

    np.save(embeddings_path, embeddings[keep_mask])


def force_reembed_stale_videos():
    """Un-indexes every video whose stored frame_interval_seconds doesn't
    match the current FRAME_INTERVAL_SECONDS (including legacy rows from
    before that column existed, which read as NULL), so the next
    build_index() call re-embeds them fresh at the current density instead
    of skipping them as unchanged. Images are untouched - frame interval
    doesn't apply to them.

    Returns the number of videos queued for re-embedding - call
    build_index() again afterward to actually do it."""
    conn = init_db()
    video_clause = " OR ".join(f"file_path LIKE '%{ext}'" for ext in VIDEO_EXTENSIONS) + \
        " OR " + " OR ".join(f"file_path LIKE '%{ext.upper()}'" for ext in VIDEO_EXTENSIONS)
    stale_paths = [row[0] for row in conn.execute(
        f"SELECT file_path FROM indexed_files WHERE ({video_clause}) "
        "AND (frame_interval_seconds IS NULL OR frame_interval_seconds != ?)",
        (FRAME_INTERVAL_SECONDS,),
    )]
    if not stale_paths:
        conn.close()
        return 0

    _delete_and_renumber(conn, stale_paths)
    conn.commit()
    conn.close()
    return len(stale_paths)


def prune_missing_files(dry_run=True):
    """Removes index entries for files that no longer exist on the shared
    drive - deleted, or moved without the index being told (this project
    deliberately doesn't do move-detection; a relocated file both loses its
    old entry here and gets fully re-embedded as new on the next
    build_index() run - see huys-video-editor-search-tab-built memory for
    that scoping discussion).

    Only ever considers DRIVE::-portable paths - local-only content (e.g. a
    Mac's native Photos library, or a PC's local iCloud folder) is EXPECTED
    to be unresolvable from whichever machine isn't its native one, and
    pruning those would be wrong, not a cleanup.

    dry_run=True (default): reports the count that WOULD be pruned without
    touching the database at all. Call again with dry_run=False, once
    reviewed, to actually delete.

    Refuses to run at all if the drive isn't currently connected - every
    file would otherwise incorrectly look missing, risking wiping the whole
    index rather than just the genuinely-stale part of it."""
    if not find_volume_by_label(EXTERNAL_DRIVE_LABEL):
        raise RuntimeError(
            f"'{EXTERNAL_DRIVE_LABEL}' isn't connected - refusing to prune, since every "
            "file would incorrectly look missing and the whole index could be wiped."
        )

    conn = init_db()
    rows = conn.execute("SELECT file_path FROM indexed_files WHERE file_path LIKE 'DRIVE::%'").fetchall()
    stale_paths = []
    for (stored_path,) in rows:
        real_path = resolve_portable_path(stored_path, EXTERNAL_DRIVE_LABEL)
        if not real_path or not real_path.exists():
            stale_paths.append(stored_path)

    if dry_run or not stale_paths:
        conn.close()
        return len(stale_paths)

    _delete_and_renumber(conn, stale_paths)
    conn.commit()
    conn.close()
    return len(stale_paths)


def backfill_metadata(profile=None, progress_callback=None, force=False, video_only=False):
    """Catch-up pass for metadata (fast, no re-embedding needed).

    force=False (default): only fills in rows still missing date_taken -
    the original one-time-catch-up behavior for files indexed before EXIF
    support existed.
    force=True: reprocesses every matching row regardless of its current
    value - needed (not just a nice-to-have) for correcting the video UTC/
    timezone bug fixed 05-08-2026, where existing rows already had a value,
    just a wrong one, so filtering on "still NULL" would never have touched
    them. video_only=True narrows a forced run to video files, since the
    bug just fixed was video-specific - no need to re-touch already-correct
    photo dates.

    Safe to re-run either way."""
    profile = profile or get_os_profile()
    conn = init_db()
    where = "1=1" if force else "date_taken IS NULL"
    if video_only:
        video_clause = " OR ".join(f"file_path LIKE '%{ext}'" for ext in VIDEO_EXTENSIONS) + \
            " OR " + " OR ".join(f"file_path LIKE '%{ext.upper()}'" for ext in VIDEO_EXTENSIONS)
        where = f"({where}) AND ({video_clause})"
    rows = conn.execute(f"SELECT file_path FROM indexed_files WHERE {where}").fetchall()

    for i, (file_path,) in enumerate(rows):
        path = resolve_portable_path(file_path, EXTERNAL_DRIVE_LABEL)
        if progress_callback:
            progress_callback(i + 1, len(rows), Path(file_path).name)
        if not path or not path.exists():
            continue
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            date_taken, lat, lon = extract_video_metadata(profile["ffmpeg_binary"], path)
        else:
            date_taken, lat, lon = extract_image_metadata(path)
        conn.execute(
            "UPDATE indexed_files SET date_taken = ?, lat = ?, lon = ? WHERE file_path = ?",
            (date_taken, lat, lon, file_path),
        )
        if i % 200 == 0:
            conn.commit()

    conn.commit()
    conn.close()
    return len(rows)


if __name__ == "__main__":
    def report(i, total, name):
        print(f"[{i}/{total}] {name}")

    new_count = build_index(progress_callback=report)
    print(f"Indexed {new_count} new frames/images.")
