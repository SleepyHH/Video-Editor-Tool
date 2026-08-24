"""
One-time tool: reorganizes the shared drive's Camera Roll folder into
YYYYMM__ folders by each file's true capture date, amalgamating existing
batch-suffixed folders (202309_a, 202309_b, etc.) into one canonical folder
per month first. Scope: Camera Roll only - the loose files at its root, plus
Vn/, italy/, "Photos back up"/, "New folder"/ inside it. Not part of the
ongoing indexing pipeline - run manually, once.

Dry-run by default - writes a report, touches nothing:

    python3 reorganize_camera_roll.py

Only actually moves files with an explicit flag, after reviewing that report:

    python3 reorganize_camera_roll.py --execute
"""
import argparse
import hashlib
import re
import shutil
import subprocess
from pathlib import Path

from PIL import Image

from config import find_volume_by_label, get_os_profile, walk_media_files
from media_index import (
    EXTERNAL_DRIVE_LABEL, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
    extract_image_metadata, extract_video_metadata,
)

UNKNOWN_DATE_FOLDER = "Unknown date"
DUPLICATES_FOLDER = "Duplicates"
AAE_EXTENSION = ".aae"
LOOSE_SOURCE_FOLDERS = ["Vn", "italy", "Photos back up", "New folder"]
HASH_CHUNK_SIZE = 1024 * 1024  # 1MB - reads large videos in chunks, not all at once

CANONICAL_SUFFIX = "__"
CANONICAL_NAME_RE = re.compile(r"^(\d{6})__$")   # e.g. 202204__
VARIANT_NAME_RE = re.compile(r"^(\d{6})_[a-zA-Z]?$")  # e.g. 202309_a, 202501_

REPORT_PATH = Path(__file__).resolve().parent / "reorganize_camera_roll_report.txt"
LOG_PATH = Path(__file__).resolve().parent / "reorganize_camera_roll.log"


def find_camera_roll_root():
    drive_root = find_volume_by_label(EXTERNAL_DRIVE_LABEL)
    if not drive_root:
        raise RuntimeError(f"'{EXTERNAL_DRIVE_LABEL}' isn't connected.")
    root = drive_root / "Camera Roll"
    if not root.is_dir():
        raise RuntimeError(f"No 'Camera Roll' folder found at {root}")
    return root


def real_files_in(folder_path):
    """Non-recursive listing of a folder's real files - excludes macOS
    AppleDouble sidecar files ("._IMG_1234.mov"), the same exFAT artifact
    already found and filtered out of media_index.py's own drive walk.
    Left alone entirely (never moved, never counted), matching how the rest
    of this project already treats them."""
    return [p for p in folder_path.iterdir() if p.is_file() and not p.name.startswith("._")]


def canonical_folder_name(date_taken):
    """'2023-06-23 18:41:11' -> '202306__'. Pure/no I/O, directly testable."""
    year, month = date_taken[:4], date_taken[5:7]
    return f"{year}{month}{CANONICAL_SUFFIX}"


def classify_existing_folders(camera_roll_root):
    """Splits the folders already sitting in Camera Roll into two dicts:
    {month: canonical_path} and {month: [variant_paths]}. Non-date folders
    (Vn, italy, ...) match neither pattern and are ignored here - handled
    separately as loose sources instead."""
    canonical = {}
    variants = {}
    for entry in sorted(camera_roll_root.iterdir()):
        if not entry.is_dir():
            continue
        canon_match = CANONICAL_NAME_RE.match(entry.name)
        if canon_match:
            canonical[canon_match.group(1)] = entry
            continue
        variant_match = VARIANT_NAME_RE.match(entry.name)
        if variant_match:
            variants.setdefault(variant_match.group(1), []).append(entry)
    return canonical, variants


def _video_is_openable(ffmpeg_binary, path):
    """A corrupted/truncated video fails to report a duration. Same
    ffprobe-based check as main.py's probe_video_duration, kept local here
    rather than importing all of main.py (a PyQt6 GUI module) for one
    subprocess call."""
    try:
        result = subprocess.run([ffmpeg_binary, "-i", str(path)], capture_output=True, text=True, timeout=120)
        return bool(re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr))
    except Exception:
        return False


def is_openable(path, ffmpeg_binary):
    """Whether a file can actually be decoded, not just that it exists -
    used to break collisions (a corrupted/truncated file always loses,
    regardless of size). Unclassified file types (zip, ini, ...) have no
    meaningful "open" check and are treated as fine."""
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        return _video_is_openable(ffmpeg_binary, path)
    if suffix in IMAGE_EXTENSIONS:
        try:
            with Image.open(path) as img:
                img.verify()
            return True
        except Exception:
            return False
    return True


def pick_collision_winner(candidates, ffmpeg_binary):
    """Given every file that wants the same destination filename, returns
    (winner, losers) - the winner keeps the plain name, losers each get a
    distinguishing suffix. Nothing is ever discarded.

    Priority: a file that fails to open (corrupted/truncated/0 bytes) always
    loses to one that opens fine, regardless of size. Among files that open
    (or all failing to), the larger file wins - a full-resolution original
    is almost always bigger than a thumbnail, re-compressed copy, or partial
    duplicate."""
    def sort_key(path):
        return (is_openable(path, ffmpeg_binary), path.stat().st_size)

    ranked = sorted(candidates, key=sort_key, reverse=True)
    return ranked[0], ranked[1:]


def content_hash(path):
    """Full-file SHA-256, read in chunks so multi-GB videos don't need to
    fit in memory at once. Deliberately the whole file, not a cheap sample -
    unlike the move-detection idea considered and dropped elsewhere in this
    project, this result can cause a file to be routed away from the main
    library into Duplicates/, so it needs to be certain, not just likely."""
    digest = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(HASH_CHUNK_SIZE), b""):
            digest.update(chunk)
    return digest.hexdigest()


def find_duplicate_groups(paths):
    """Groups of 2+ files with byte-identical content, keyed by hash. Only
    ever computes the (expensive) full hash for files that already share an
    exact size with at least one other candidate - two files of different
    sizes can never be duplicates, so this skips hashing the (typical
    majority) of files with a unique size entirely, for free."""
    by_size = {}
    for path in paths:
        by_size.setdefault(path.stat().st_size, []).append(path)

    groups = {}
    for size, candidates in by_size.items():
        if len(candidates) < 2:
            continue
        for path in candidates:
            groups.setdefault(content_hash(path), []).append(path)

    return {h: members for h, members in groups.items() if len(members) > 1}


def pick_duplicate_keeper(members, canonical_folder_paths):
    """Which copy of an identical-content group stays in the main library -
    the rest move to Duplicates/. All members are byte-identical, so
    is_openable/size (used for filename collisions) can't distinguish them;
    instead: prefer whichever copy is already correctly sitting in an
    existing canonical YYYYMM folder (needs no action at all), then the
    shortest filename (a plain "IMG_1234.jpg" over an exported UUID name),
    then alphabetical - fully deterministic either way."""
    already_filed = [p for p in members if p.parent in canonical_folder_paths]
    pool = already_filed or members
    return sorted(pool, key=lambda p: (len(p.name), p.name))[0]


def suffixed_name(filename, n):
    path = Path(filename)
    return f"{path.stem} (dup{n}){path.suffix}"


def find_aae_pairs(paths):
    """Splits a list of paths into (media_paths, aae_by_stem) - .AAE files
    don't get their own date extracted (they're not media), they follow
    whatever destination their paired media file resolves to instead."""
    media_paths = []
    aae_by_stem = {}
    for path in paths:
        if path.suffix.lower() == AAE_EXTENSION:
            aae_by_stem[path.stem] = path
        else:
            media_paths.append(path)
    return media_paths, aae_by_stem


def extract_true_date(path, ffmpeg_binary):
    suffix = path.suffix.lower()
    if suffix in VIDEO_EXTENSIONS:
        date_taken, _, _ = extract_video_metadata(ffmpeg_binary, path)
        return date_taken
    if suffix in IMAGE_EXTENSIONS:
        date_taken, _, _ = extract_image_metadata(path)
        return date_taken
    return None  # unclassified type - no meaningful "true date" to extract


def collect_loose_sources(camera_roll_root):
    """Every file that needs sorting: the loose root files, plus everything
    inside Vn/, italy/, "Photos back up"/, "New folder"/ - all in scope per
    the user's explicit confirmation. real_files_in() only returns files,
    not directories, so this naturally never descends into the
    canonical/variant YYYYMM folders, Unknown date/, or Duplicates/ -
    those are handled by the amalgamation/dedup passes, not re-swept here."""
    files = real_files_in(camera_roll_root)
    for folder_name in LOOSE_SOURCE_FOLDERS:
        folder = camera_roll_root / folder_name
        if folder.is_dir():
            files.extend(walk_media_files(folder))
    return files


def build_plan(camera_roll_root, ffmpeg_binary, log=print):
    """Computes every planned action without touching the filesystem (aside
    from read-only stats/opens needed for date extraction, duplicate
    detection, and collision resolution). Returns a list of
    (source_path, destination_path, reason) - the exact same list dry-run
    reporting and real execution both consume, so what you review is what
    actually happens."""
    canonical, variants = classify_existing_folders(camera_roll_root)
    canonical_folder_paths = set(canonical.values())

    # --- Phase 0: content-based duplicate detection, across every file this
    # run considers (already-canonical, variant, and loose) - a duplicate
    # anywhere in that combined set is still a duplicate, not just within
    # one folder. Runs before anything else, so losers never enter the
    # normal amalgamation/date-sort logic at all - they go straight to
    # Duplicates/ instead. ---
    all_candidates = []
    for canon_path in canonical.values():
        all_candidates.extend(real_files_in(canon_path))
    for group in variants.values():
        for variant_folder in group:
            all_candidates.extend(real_files_in(variant_folder))
    all_candidates.extend(collect_loose_sources(camera_roll_root))

    duplicate_losers = {}  # source_path -> reason string
    for members in find_duplicate_groups(all_candidates).values():
        keeper = pick_duplicate_keeper(members, canonical_folder_paths)
        for member in members:
            if member != keeper:
                duplicate_losers[member] = f"duplicate of {keeper}"

    # destination_claims: (folder_path, filename) -> list of source paths
    # that want it - seeded from what's already really on disk, then grown
    # as loose/variant files get classified, so a brand-new arrival competes
    # fairly against both real existing files and other new arrivals.
    destination_claims = {}

    def claim(folder_path, filename, source_path):
        destination_claims.setdefault((folder_path, filename), []).append(source_path)

    def claim_or_duplicate(folder_path, filename, source_path):
        if source_path in duplicate_losers:
            claim(camera_roll_root / DUPLICATES_FOLDER, source_path.name, source_path)
        else:
            claim(folder_path, filename, source_path)

    def seed_existing(folder_path):
        if folder_path.exists():
            for entry in real_files_in(folder_path):
                claim_or_duplicate(folder_path, entry.name, entry)

    # --- Phase 1: amalgamate variant folders into their canonical folder ---
    amalgamation_targets = {}  # month -> resolved canonical folder path (may not exist yet)
    for month, canon_path in canonical.items():
        amalgamation_targets[month] = canon_path
        seed_existing(canon_path)
    for month in variants:
        if month not in amalgamation_targets:
            amalgamation_targets[month] = camera_roll_root / f"{month}{CANONICAL_SUFFIX}"

    variant_files_by_month = {}
    for month, variant_folders in variants.items():
        target = amalgamation_targets[month]
        for variant_folder in variant_folders:
            media_files, aae_by_stem = find_aae_pairs(real_files_in(variant_folder))
            for media_path in media_files:
                claim_or_duplicate(target, media_path.name, media_path)
                paired_aae = aae_by_stem.pop(media_path.stem, None)
                if paired_aae:
                    claim_or_duplicate(target, paired_aae.name, paired_aae)
            for orphan_aae in aae_by_stem.values():  # no matching media in this folder
                claim_or_duplicate(target, orphan_aae.name, orphan_aae)
        variant_files_by_month[month] = target

    # --- Phase 2: sort loose sources by true date ---
    loose_files = collect_loose_sources(camera_roll_root)
    all_loose_media, aae_by_stem = find_aae_pairs(loose_files)
    # Duplicates never get their (wasted) date extracted - route straight to
    # Duplicates/ instead of the normal date-sort below. Kept in
    # all_loose_media (not dropped) so a duplicate's .AAE can still find its
    # pair and follow it to Duplicates/, rather than being orphaned to
    # Unknown date/ just because its media file was filtered out first.
    for loser in all_loose_media:
        if loser in duplicate_losers:
            claim(camera_roll_root / DUPLICATES_FOLDER, loser.name, loser)
    media_files = [p for p in all_loose_media if p not in duplicate_losers]

    media_destination = {}  # source_path -> (folder_path, reason)
    for media_path in media_files:
        date_taken = extract_true_date(media_path, ffmpeg_binary)
        if date_taken:
            month = canonical_folder_name(date_taken)[:6]
            target = amalgamation_targets.get(month) or (camera_roll_root / f"{month}{CANONICAL_SUFFIX}")
            amalgamation_targets.setdefault(month, target)
            reason = f"dated {date_taken}"
        else:
            target = camera_roll_root / UNKNOWN_DATE_FOLDER
            reason = "no extractable date"
        media_destination[media_path] = (target, reason)
        claim(target, media_path.name, media_path)

    for stem, aae_path in aae_by_stem.items():
        paired = next((m for m in all_loose_media if m.stem == stem), None)
        if paired and paired in media_destination:
            target, _ = media_destination[paired]
            claim_or_duplicate(target, aae_path.name, aae_path)
        elif paired in duplicate_losers:
            claim_or_duplicate(camera_roll_root / DUPLICATES_FOLDER, aae_path.name, aae_path)
        else:
            target = camera_roll_root / UNKNOWN_DATE_FOLDER
            claim_or_duplicate(target, aae_path.name, aae_path)

    # --- Resolve every destination slot, including collisions ---
    reasons = {}
    for month, target in amalgamation_targets.items():
        for variant_folder in variants.get(month, []):
            for p in real_files_in(variant_folder):
                reasons[p] = f"amalgamated from {variant_folder.name}/"
    for source, (target, reason) in media_destination.items():
        reasons[source] = reason
    reasons.update(duplicate_losers)  # duplicate status always wins the displayed reason

    plan = []
    for (folder_path, filename), candidates in destination_claims.items():
        if len(candidates) == 1:
            source = candidates[0]
            if source.parent == folder_path and source.name == filename:
                continue  # already exactly where it belongs - no action needed
            plan.append((source, folder_path / filename, reasons.get(source, "amalgamated")))
            continue

        winner, losers = pick_collision_winner(candidates, ffmpeg_binary)
        if not (winner.parent == folder_path and winner.name == filename):
            plan.append((winner, folder_path / filename, reasons.get(winner, "amalgamated") + " (collision winner)"))
        for i, loser in enumerate(losers, start=1):
            new_name = suffixed_name(filename, i)
            plan.append((loser, folder_path / new_name, reasons.get(loser, "amalgamated") + " (collision, renamed)"))

    return plan


def write_report(plan, path=REPORT_PATH):
    lines = [f"{len(plan)} planned actions\n"]
    for source, destination, reason in plan:
        lines.append(f"{source} -> {destination}  [{reason}]")
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def execute_plan(plan, log_path=LOG_PATH):
    with open(log_path, "a", encoding="utf-8") as log_file:
        for source, destination, reason in plan:
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(destination))
            log_file.write(f"{source} -> {destination}  [{reason}]\n")
            log_file.flush()

    # Remove now-empty variant folders left behind by the amalgamation pass.
    canonical, variants = classify_existing_folders(find_camera_roll_root())
    for variant_folders in variants.values():
        for folder in variant_folders:
            if folder.exists() and not any(folder.iterdir()):
                folder.rmdir()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--execute", action="store_true",
                         help="Actually perform the moves. Without this flag, only a dry-run report is written.")
    args = parser.parse_args()

    camera_roll_root = find_camera_roll_root()
    ffmpeg_binary = get_os_profile()["ffmpeg_binary"]

    print(f"Planning against {camera_roll_root} ...")
    plan = build_plan(camera_roll_root, ffmpeg_binary)

    if not args.execute:
        report_path = write_report(plan)
        print(f"Dry run only - {len(plan)} planned actions, nothing moved.")
        print(f"Report written to {report_path}")
        return

    print(f"Executing {len(plan)} moves...")
    execute_plan(plan)
    print(f"Done. Log written to {LOG_PATH}")


if __name__ == "__main__":
    main()
