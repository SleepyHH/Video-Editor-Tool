import os
import platform
from pathlib import Path

APP_DIR = Path(__file__).resolve().parent

def get_os_profile():
    home = Path.home()

    profile = {
        "os": platform.system(),
        "photos_path": home / "Pictures" / "iCloud Photos" / "Photos",
        "ffmpeg_binary": "ffmpeg",
        "pipeline_mode": "cloud",  # Default configuration for Windows setups
        "app_data_dir": Path(os.environ.get("APPDATA", home / "AppData" / "Roaming")) / "HuysVideoEditor",
        # Confirmed live 06-08-2026: this PC's ffmpeg build (gyan.dev full_build) has NVDEC
        # compiled in and an RTX 4060 to use it on - video frame extraction was previously
        # 100% CPU software-decode even though embedding already ran on GPU, and that decode
        # step was the real bottleneck (near-0% GPU utilization during indexing despite the
        # model being loaded and active on it). None on Mac below - can't verify the bundled
        # ffmpeg binary there has videotoolbox support, so left unchanged rather than risking
        # every video silently failing there.
        "ffmpeg_hwaccel": "cuda",
    }

    if platform.system() == "Darwin":  # macOS Profile configuration properties
        profile["ffmpeg_binary"] = str(APP_DIR / "ffmpeg")
        profile["pipeline_mode"] = "native_db"  # Uses the fast local database bridge on Mac
        profile["photos_path"] = home / "Pictures" / "Photos Library.photoslibrary"
        profile["app_data_dir"] = home / "Library" / "Application Support" / "HuysVideoEditor"
        profile["ffmpeg_hwaccel"] = None

    return profile


def find_volume_by_label(label):
    """Finds a removable/external drive by its volume label rather than a
    hardcoded path - the same physical drive mounts at a different path
    depending on OS (Mac: /Volumes/<label>) and even across plug-ins on
    Windows (drive letters aren't stable), but the label given to it at
    format time stays constant. Returns a Path, or None if not currently
    connected."""
    if platform.system() == "Darwin":
        candidate = Path("/Volumes") / label
        return candidate if candidate.exists() else None

    # Windows: no plug-and-play drive-letter guarantee, so check every letter's
    # actual volume label via the same Win32 API Explorer itself uses - unverified
    # on real Windows hardware (no machine to test on), same caveat as the rest
    # of this project's Windows branch.
    import ctypes
    kernel32 = ctypes.windll.kernel32
    for letter in "ABCDEFGHIJKLMNOPQRSTUVWXYZ":
        root = f"{letter}:\\"
        if not Path(root).exists():
            continue
        name_buf = ctypes.create_unicode_buffer(261)
        ok = kernel32.GetVolumeInformationW(root, name_buf, 260, None, None, None, None, 0)
        if ok and name_buf.value == label:
            return Path(root)
    return None


# Real, permanent source - added 05-08-2026: an external drive holding the actual media
# library to be edited, shared between the user's Mac and PC. Found by volume label (see
# find_volume_by_label) rather than a hardcoded path, since the same drive mounts
# differently per-OS and even per plug-in. Formatted exFAT specifically for this reason -
# natively readable/writable on both Mac and Windows, unlike NTFS (read-only on Mac) or
# APFS/HFS+ (unreadable on Windows). Shared here (not just in media_index.py) since both the
# search indexer and main.py's media list need to agree on which drive/files count.
EXTERNAL_DRIVE_LABEL = "Huy's HDD"

# Housekeeping folders that exist on every exFAT/NTFS drive but aren't real content - skipped
# during any recursive scan of the drive, along with anything starting with ".".
# "HuysVideoEditor_SearchIndex" is this tool's own index/thumbnail-cache folder - confirmed
# live 07-08-2026: without this, a drive walk recurses into its own generated thumbnails and
# indexes them as if they were separate library photos, quietly self-polluting a little more
# every time someone browses search results.
IGNORED_DIR_NAMES = {"$RECYCLE.BIN", "System Volume Information", ".Trashes", ".Spotlight-V100", ".fseventsd", "HuysVideoEditor_SearchIndex"}

VIDEO_EXTENSIONS = {".mp4", ".mov", ".m4v"}


def walk_media_files(folder):
    """Recursively yields real file paths under folder, skipping hidden/
    housekeeping directories (IGNORED_DIR_NAMES, or starting with ".") and
    macOS AppleDouble sidecar files ("._IMG_1234.mov"). Confirmed live
    08-08-2026: on exFAT (this drive's format, chosen for Mac/Windows
    compatibility - see EXTERNAL_DRIVE_LABEL above), macOS silently writes
    one of these tiny resource-fork stubs next to every real file it touches.
    They carry the same extension as the real file, so an extension-only
    filter would let them through as if they were actual media - confirmed by
    finding several sorted to the very top of a newest-first video listing.
    Shared here (not duplicated per-caller) so both media_index.py's indexer
    and main.py's media list are protected by one definition, not two that
    could quietly drift apart."""
    for root, dirs, files in os.walk(folder):
        dirs[:] = [d for d in dirs if d not in IGNORED_DIR_NAMES and not d.startswith(".")]
        for name in files:
            if not name.startswith("._"):
                yield Path(root) / name


DRIVE_PATH_PREFIX = "DRIVE::"


def to_portable_path(real_path, drive_label):
    """If real_path lives on the given drive, returns a portable
    "DRIVE::relative/path" form that's independent of OS or mount point -
    otherwise returns the path unchanged (for genuinely local-only content,
    e.g. the Mac Photos library, which has no cross-machine equivalent and
    doesn't need this). Needed because a stored absolute path like
    "/Volumes/Huy's HDD/italy/video.mov" is Mac-specific - the same file
    is "E:\\italy\\video.mov" or similar on Windows, so storing the raw
    absolute path would silently break the moment the index is read from
    the other machine."""
    drive_root = find_volume_by_label(drive_label)
    if drive_root:
        try:
            rel = Path(real_path).resolve().relative_to(drive_root.resolve())
            return f"{DRIVE_PATH_PREFIX}{rel.as_posix()}"
        except ValueError:
            pass
    return str(real_path)


def resolve_portable_path(stored_path, drive_label):
    """Reverses to_portable_path: turns a stored "DRIVE::..." reference
    into a real, currently-valid absolute path using wherever the drive is
    mounted on THIS machine right now. Returns None if the drive isn't
    currently connected. A path that was never portable (local-only
    content) passes through unchanged as a Path."""
    if not str(stored_path).startswith(DRIVE_PATH_PREFIX):
        return Path(stored_path)
    drive_root = find_volume_by_label(drive_label)
    if not drive_root:
        return None
    return drive_root / str(stored_path)[len(DRIVE_PATH_PREFIX):]

