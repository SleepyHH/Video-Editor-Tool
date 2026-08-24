# --- LEGACY REPAIR SHIELD FOR PYTHON 3.12+ ---
import sys
import types
if 'imp' not in sys.modules:
    # We dynamically fake the old 'imp' library in memory so older packages don't crash!
    fake_imp = types.ModuleType('imp')
    fake_imp.find_module = lambda name, path=None: (None, None, None)
    fake_imp.load_module = lambda name, file, filename, details: None
    sys.modules['imp'] = fake_imp
# ---------------------------------------------

import subprocess
import time
import json
import re
from pathlib import Path

from PyQt6.QtWidgets import (QApplication, QMainWindow, QWidget, QHBoxLayout,
                             QVBoxLayout, QGridLayout, QPushButton, QListWidget, QListWidgetItem, QLabel,
                             QFileDialog, QMessageBox, QDoubleSpinBox, QSpinBox, QComboBox, QTextEdit, QCheckBox,
                             QTabWidget, QScrollArea, QDateEdit, QLineEdit, QCompleter)
from PyQt6.QtCore import QDate, QThread, QTimer, QUrl, Qt, pyqtSignal, QStringListModel
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QDesktopServices, QPixmap, QImage
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from config import EXTERNAL_DRIVE_LABEL, VIDEO_EXTENSIONS, find_volume_by_label, get_os_profile, resolve_portable_path, walk_media_files

# One consistent folder for videos exported out of Photos for editing - easy to find, easy to clean up
IMPORT_CACHE_DIR = Path.home() / "Documents" / "HuysVideoEditor_Imports"

# Default destination for rendered videos and transcripts when a manual save location isn't chosen
EXPORT_CACHE_DIR = Path.home() / "Documents" / "HuysVideoEditor_Exports"

# Auto-detect is unreliable on quiet/accented/short or code-switched audio (confirmed: it
# misidentified real Vietnamese speech as Khmer and produced gibberish) - forcing the actual
# language is both far more accurate and several times faster since it skips language-guessing
# entirely. For content that genuinely switches between languages mid-clip, "Mixed languages"
# (multilingual=True) re-detects per segment instead of once for the whole file - confirmed via
# live testing to handle Vietnamese/English code-switching cleanly, unlike whole-file auto-detect.
MULTILINGUAL_SENTINEL = "__multilingual__"

WHISPER_IDLE_UNLOAD_MS = 2 * 60 * 1000

# Matches media_search_server.py's existing CLIP_IDLE_UNLOAD_SECONDS (10 min) -
# one consistent idle window across every place CLIP gets loaded.
CLIP_IDLE_UNLOAD_MS = 10 * 60 * 1000

# Search result card sizing - kept small deliberately (08-08-2026) so the
# shared preview pane can take the larger ~1/3-of-window share instead.
RESULT_THUMB_WIDTH = 160
RESULT_THUMB_HEIGHT = 110
RESULT_CARD_WIDTH = 180

# People tab - a real freeze otherwise, not just a slow-feeling one: loading every crop
# image for a large group (measured live, 2026-08-20: the real library's "Unclustered"
# bucket alone, 6,440 faces) synchronously on the UI thread took 88.67s for the image
# loads alone, before any widget construction - the same "fine at test scale, breaks at
# real scale" category of bug as the reclustering fix. Search results never hit this
# since the user picks a result count (search_top_box, capped at 200); nothing capped
# how many faces a single cluster/person/match-review could try to render at once.
MAX_FACES_TO_DISPLAY = 200

# All 100 languages faster-whisper/Whisper supports, code -> display name
WHISPER_LANGUAGE_NAMES = {
    "en": "English", "zh": "Chinese", "de": "German", "es": "Spanish", "ru": "Russian",
    "ko": "Korean", "fr": "French", "ja": "Japanese", "pt": "Portuguese", "tr": "Turkish",
    "pl": "Polish", "ca": "Catalan", "nl": "Dutch", "ar": "Arabic", "sv": "Swedish",
    "it": "Italian", "id": "Indonesian", "hi": "Hindi", "fi": "Finnish", "vi": "Vietnamese",
    "he": "Hebrew", "uk": "Ukrainian", "el": "Greek", "ms": "Malay", "cs": "Czech",
    "ro": "Romanian", "da": "Danish", "hu": "Hungarian", "ta": "Tamil", "no": "Norwegian",
    "th": "Thai", "ur": "Urdu", "hr": "Croatian", "bg": "Bulgarian", "lt": "Lithuanian",
    "la": "Latin", "mi": "Maori", "ml": "Malayalam", "cy": "Welsh", "sk": "Slovak",
    "te": "Telugu", "fa": "Persian", "lv": "Latvian", "bn": "Bengali", "sr": "Serbian",
    "az": "Azerbaijani", "sl": "Slovenian", "kn": "Kannada", "et": "Estonian", "mk": "Macedonian",
    "br": "Breton", "eu": "Basque", "is": "Icelandic", "hy": "Armenian", "ne": "Nepali",
    "mn": "Mongolian", "bs": "Bosnian", "kk": "Kazakh", "sq": "Albanian", "sw": "Swahili",
    "gl": "Galician", "mr": "Marathi", "pa": "Punjabi", "si": "Sinhala", "km": "Khmer",
    "sn": "Shona", "yo": "Yoruba", "so": "Somali", "af": "Afrikaans", "oc": "Occitan",
    "ka": "Georgian", "be": "Belarusian", "tg": "Tajik", "sd": "Sindhi", "gu": "Gujarati",
    "am": "Amharic", "yi": "Yiddish", "lo": "Lao", "uz": "Uzbek", "fo": "Faroese",
    "ht": "Haitian Creole", "ps": "Pashto", "tk": "Turkmen", "nn": "Nynorsk", "mt": "Maltese",
    "sa": "Sanskrit", "lb": "Luxembourgish", "my": "Myanmar", "bo": "Tibetan", "tl": "Tagalog",
    "mg": "Malagasy", "as": "Assamese", "tt": "Tatar", "haw": "Hawaiian", "ln": "Lingala",
    "ha": "Hausa", "ba": "Bashkir", "jw": "Javanese", "su": "Sundanese", "yue": "Cantonese",
}

# Dropdown order: English default first, then Mixed languages, then everything else A-Z.
# No "Auto-detect" - confirmed unreliable (misidentified real Vietnamese speech as Khmer).
TRANSCRIPTION_LANGUAGES = {"English": "en", "Mixed languages": MULTILINGUAL_SENTINEL}
for _code, _name in sorted(WHISPER_LANGUAGE_NAMES.items(), key=lambda kv: kv[1]):
    if _name != "English":
        TRANSCRIPTION_LANGUAGES[_name] = _code


def format_srt_timestamp(seconds):
    ms = int(round(seconds * 1000))
    h, ms = divmod(ms, 3600000)
    m, ms = divmod(ms, 60000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def get_unique_path(path):
    """Avoid clobbering an existing file - appends -1, -2, -3... until the name is free."""
    if not path.exists():
        return path
    counter = 1
    while True:
        candidate = path.with_name(f"{path.stem}-{counter}{path.suffix}")
        if not candidate.exists():
            return candidate
        counter += 1


def load_manual_imports_from_disk(app_data_dir):
    try:
        manual_imports_file = app_data_dir / "manual_imports.json"
        entries = json.loads(manual_imports_file.read_text())
        return [(e["name"], e["path"]) for e in entries if Path(e["path"]).exists()]
    except Exception:
        return []


def save_manual_imports_to_disk(app_data_dir, items):
    manual_imports_file = app_data_dir / "manual_imports.json"
    manual_imports_file.parent.mkdir(parents=True, exist_ok=True)
    manual_imports_file.write_text(json.dumps([{"name": n, "path": p} for n, p in items]))


def list_hdd_media(drive_root):
    """Finds video files on the shared external drive via config's shared
    walker (skips housekeeping folders and macOS AppleDouble sidecar files -
    see config.walk_media_files) - keeps what's browsable here in sync with
    what's actually searchable, rather than two definitions that could
    drift. Returns (display_name, path) tuples, newest-first by mtime."""
    found = []
    for path in walk_media_files(drive_root):
        if path.suffix.lower() in VIDEO_EXTENSIONS:
            found.append((path.name, path, path.stat().st_mtime))
    found.sort(key=lambda entry: entry[2], reverse=True)
    return [(name, path) for name, path, _ in found]


def columns_for_width(available_width, card_width=RESULT_CARD_WIDTH, spacing=8):
    """How many result cards fit per row of the search tab's responsive
    grid at the given width - pure/no-Qt so it's directly testable."""
    return max(1, (available_width + spacing) // (card_width + spacing))


def probe_video_duration(ffmpeg_binary, path):
    """Video duration in seconds via one fast ffmpeg probe, or None if it
    can't be determined. Same regex approach as ProfessionalAIEditor's own
    probe_video, but standalone/no self - called from SearchWorker's
    background thread, which has no window instance to read a profile off."""
    result = subprocess.run([ffmpeg_binary, "-i", str(path)], capture_output=True, text=True)
    match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
    if not match:
        return None
    h, m, s = match.groups()
    return int(h) * 3600 + int(m) * 60 + float(s)

# --- UPGRADED INTERACTIVE TIMELINE COMPONENT ---
class VisualTimeline(QWidget):
    def __init__(self, player_reference):
        super().__init__()
        self.player = player_reference # Hook directly to the live video engine
        self.setMinimumHeight(45)      # Taller space to fit numbers
        self.chunks = []
        self.current_progress_pct = 0.0
        self.video_duration_ms = 0
        
        # Enable tracking mouse movements over the bar
        self.setMouseTracking(True)
        self.hover_pct = -1.0

    def set_timeline_data(self, chunks, duration_ms):
        self.chunks = chunks
        self.video_duration_ms = duration_ms
        self.update()

    def set_progress(self, position_ms):
        if self.video_duration_ms > 0:
            self.current_progress_pct = position_ms / self.video_duration_ms
            self.update()

    def format_time(self, ms):
        # Helper to convert raw milliseconds into 00:00 readable format strings
        total_seconds = int(ms / 1000)
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"

    def paintEvent(self, event):
        painter = QPainter(self)
        width = self.width()
        height = self.height()
        bar_height = 25 # Reserve top section for timestamps
        
        # Draw background base track
        painter.fillRect(0, 15, width, bar_height, QColor("#1a1a1a"))

        # No video loaded at all yet (vs. a video loaded but with no
        # speech/silence chunk data - the search tab's preview scrubber uses
        # this widget the second way, just a plain seek bar with no cut blocks).
        if self.video_duration_ms <= 0:
            painter.setPen(QColor("#777777"))
            painter.drawText(10, 32, "Select a video to map timestamps...")
            return

        # 1. Draw the Green/Red Cut Blocks, if this instance has any
        for start, end, is_speech in self.chunks:
            x_start = int(start * width)
            x_end = int(end * width)
            w = max(1, x_end - x_start)
            color = QColor("#28a745") if is_speech else QColor("#dc3545")
            painter.fillRect(x_start, 15, w, bar_height, color)

        # 2. Draw Basic Macro Timestamp Markers across the header track
        painter.setFont(QFont("Arial", 8))
        painter.setPen(QColor("#888888"))
        for pct in [0.0, 0.25, 0.5, 0.75, 1.0]:
            x_pos = int(pct * (width - 30))
            time_at_pct = self.format_time(pct * self.video_duration_ms)
            painter.drawText(x_pos, 10, time_at_pct)

        # 3. Draw Hover Timestamp Tooltip Line
        if self.hover_pct >= 0:
            x_hover = int(self.hover_pct * width)
            painter.setPen(QPen(QColor("#ffffff"), 1, Qt.PenStyle.DashLine))
            painter.drawLine(x_hover, 0, x_hover, height)
            # Draw tiny floating time bubble text box overlay
            hover_time = self.format_time(self.hover_pct * self.video_duration_ms)
            painter.setPen(QColor("#ffffff"))
            painter.drawText(max(5, x_hover - 15), 12, hover_time)

        # 4. Draw Moving White Active Playhead Line tracker marker
        x_playhead = int(self.current_progress_pct * width)
        painter.setPen(QPen(QColor("#ffffff"), 2))
        painter.drawLine(x_playhead, 15, x_playhead, 15 + bar_height)

    # 5. Interactive Click-to-Scrub Logic
    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.video_duration_ms > 0:
            click_x = event.position().x()
            click_pct = click_x / self.width()
            target_ms = int(click_pct * self.video_duration_ms)
            self.player.setPosition(target_ms) # Forces live video stream to jump destinations instantly!

    def mouseMoveEvent(self, event):
        if self.video_duration_ms > 0:
            self.hover_pct = event.position().x() / self.width()
            self.update()

    def leaveEvent(self, event):
        self.hover_pct = -1.0
        self.update()


# --- SEARCH TAB BACKGROUND WORKER ---
class SearchWorker(QThread):
    """Runs the actual CLIP search + thumbnail generation off the UI thread -
    without this, a search would freeze the whole app for its duration, and
    it could no longer run alongside a transcription the way the old
    separate-process (main.py + media_search_server.py) setup could.

    media_search/media_index are imported locally here, not at module level,
    for the same reason faster_whisper is imported locally in
    transcribe_file() - keeps this app's startup free of torch/open_clip
    unless the search tab is actually used."""
    results_ready = pyqtSignal(list)
    search_failed = pyqtSignal(str)

    def __init__(self, query, top_k, after, before, file_types, person=None):
        super().__init__()
        self.query = query
        self.top_k = top_k
        self.after = after
        self.before = before
        self.file_types = file_types
        self.person = person

    def run(self):
        try:
            import media_search
        except ImportError:
            self.search_failed.emit(
                "The search engine's dependencies (torch, open_clip, etc.) aren't installed.\n\n"
                "Run: pip install torch open_clip_torch pillow-heif"
            )
            return
        try:
            results = media_search.smart_search(
                self.query, top_k=self.top_k, after=self.after,
                before=self.before, file_types=self.file_types, explicit_person=self.person,
            )
            # Thumbnail + duration are both generated here, not on the UI
            # thread - by the time results_ready fires, every thumbnail is
            # already a cached JPEG on disk and every video's length is
            # already known, so the UI thread only ever reads ready data.
            ffmpeg_binary = get_os_profile()["ffmpeg_binary"]
            for r in results:
                thumb_path = media_search.get_thumbnail(r["file_path"], r["media_type"], r["timestamp_seconds"])
                r["thumb_path"] = str(thumb_path) if thumb_path else None
                r["duration_seconds"] = (
                    probe_video_duration(ffmpeg_binary, r["file_path"]) if r["media_type"] == "video" else None
                )
            self.results_ready.emit(results)
        except Exception as e:
            self.search_failed.emit(str(e))


class ReclusterWorker(QThread):
    """Runs face_index.recluster_faces() off the UI thread. Measured live
    against the real library (2026-08-20): ~2 minutes over ~14,500 faces -
    trivial against the handful of faces this was first tested with, but a
    real freeze at real scale if left on the UI thread, the same reasoning
    as SearchWorker above."""
    finished_ok = pyqtSignal(int)
    recluster_failed = pyqtSignal(str)

    def run(self):
        try:
            import face_index
            count = face_index.recluster_faces()
            self.finished_ok.emit(count)
        except Exception as e:
            self.recluster_failed.emit(str(e))


class DefaultingDateEdit(QDateEdit):
    """Starts showing a placeholder (via QDateEdit's own special-value-text
    mechanism, at minimumDate()) rather than any real date, so the filter is
    genuinely inactive until touched - but the first click/focus jumps
    straight to a real, useful default date instead of leaving the user to
    land on Qt's own minimum (1752-09-14, the Gregorian calendar cutover
    date it uses internally) or having to pick one from scratch."""

    def __init__(self, placeholder, default_date_fn, parent=None):
        super().__init__(parent)
        self._default_date_fn = default_date_fn
        self._activated = False
        self.setCalendarPopup(True)
        self.setSpecialValueText(placeholder)
        self.setDate(self.minimumDate())

    def _activate(self):
        if not self._activated:
            self._activated = True
            self.setDate(self._default_date_fn())

    def mousePressEvent(self, event):
        self._activate()
        super().mousePressEvent(event)

    def focusInEvent(self, event):
        self._activate()
        super().focusInEvent(event)

    def is_active(self):
        return self._activated


class ResultsScrollArea(QScrollArea):
    """Emits on resize so the search tab's result grid can debounce a
    relayout to the new width - the responsive-grid design decision."""
    resized = pyqtSignal()

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.resized.emit()


class ResultThumbnail(QWidget):
    """A search result card's thumbnail, split into two click zones
    (08-08-2026 design, hover-only overlays added same day): the top ~60%
    previews the result in the shared preview pane, the bottom ~40% toggles
    selection - a "Select" strip (green "✓ Selected" once checked) rather
    than a small checkbox hit-target. Both overlays are hidden until the
    cursor is actually over the thumbnail, and stay fairly transparent so
    the image itself stays visible underneath - except the "✓ Selected"
    state, which stays visible even without hovering, so it's still obvious
    which results are picked once the cursor moves away.

    Deliberately self-painted (one paintEvent, no overlapping child QLabel
    widgets) rather than composited from separate labels layered via
    setGeometry() - simpler and more robust for an image-plus-overlay-text
    widget in general, not because the QLabel version's rendering was
    actually broken. (For the record: the real "thumbnails are black" bug,
    08-08-2026, turned out to be much simpler than a rendering quirk -
    SearchWorker.run() called media_search.get_thumbnail() but discarded its
    return value instead of storing it on the result dict, so every card
    built with thumb_path=None. A rewritten QLabel-based version would have
    "fixed" it too, for the same reason this one does: both were downstream
    of the same missing assignment. Confirmed via the real async
    SearchWorker path this time, not a synchronous shortcut that skipped it.)"""
    preview_clicked = pyqtSignal()
    select_toggled = pyqtSignal(bool)

    PREVIEW_HOVER_TINT = QColor(0, 0, 0, 50)
    SELECT_HOVER_TINT = QColor(120, 120, 120, 90)
    SELECT_ACTIVE_TINT = QColor(40, 167, 69, 140)

    def __init__(self, pixmap, width, height):
        super().__init__()
        self.setFixedSize(width, height)
        self._pixmap = pixmap if (pixmap and not pixmap.isNull()) else None
        self._selected = False
        self._hovering = False

    def _preview_zone_height(self):
        return int(self.height() * 0.6)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(0, 0, 0))
        if self._pixmap:
            scaled = self._pixmap.scaled(
                self.width(), self.height(), Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter.drawPixmap((self.width() - scaled.width()) // 2, (self.height() - scaled.height()) // 2, scaled)

        preview_h = self._preview_zone_height()
        select_h = self.height() - preview_h

        if self._hovering:
            painter.fillRect(0, 0, self.width(), preview_h, self.PREVIEW_HOVER_TINT)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(0, 0, self.width(), preview_h, Qt.AlignmentFlag.AlignCenter, "Preview")

        if self._selected:
            painter.fillRect(0, preview_h, self.width(), select_h, self.SELECT_ACTIVE_TINT)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(0, preview_h, self.width(), select_h, Qt.AlignmentFlag.AlignCenter, "✓ Selected")
        elif self._hovering:
            painter.fillRect(0, preview_h, self.width(), select_h, self.SELECT_HOVER_TINT)
            painter.setPen(QColor(255, 255, 255))
            painter.drawText(0, preview_h, self.width(), select_h, Qt.AlignmentFlag.AlignCenter, "Select")

    def enterEvent(self, event):
        self._hovering = True
        self.update()
        super().enterEvent(event)

    def leaveEvent(self, event):
        self._hovering = False
        self.update()
        super().leaveEvent(event)

    def mousePressEvent(self, event):
        if event.position().y() < self._preview_zone_height():
            self.preview_clicked.emit()
        else:
            self.set_selected(not self._selected)
        super().mousePressEvent(event)

    def is_selected(self):
        return self._selected

    def set_selected(self, value):
        if value == self._selected:
            return
        self._selected = value
        self.update()
        self.select_toggled.emit(self._selected)


class ProfessionalAIEditor(QMainWindow):
    def __init__(self):
        super().__init__()
        self.profile = get_os_profile()
        self.setWindowTitle("Huy's Video Processor")
        self.resize(1200, 650)

        self.tabs = QTabWidget()
        self.tabs.addTab(self._build_editor_tab(), "🎬 Editor")
        self.tabs.addTab(self._build_search_tab(), "🔍 Prompt-Style Search")
        self.tabs.addTab(self._build_people_tab(), "👤 People")
        self.setCentralWidget(self.tabs)
        self.tabs.currentChanged.connect(self._on_tab_changed)

        self.load_iphone_photos()

    def _on_tab_changed(self, index):
        # Pausing both unconditionally on every switch is simpler than tracking
        # which tab was just left, and harmless - pausing a player that isn't
        # currently playing is a no-op.
        self.player.pause()
        self.search_preview_player.pause()
        if self.tabs.tabText(index) == "🔍 Prompt-Style Search":
            self._refresh_search_person_filter()

    def _refresh_search_person_filter(self):
        """Keeps the Search tab's Person dropdown in sync with face_index's
        labeled people - called lazily on switching to this tab, not at app
        startup, same lazy-import discipline as everywhere else face_index
        is touched (it pulls in cv2/hdbscan, real deps most searches don't
        need). Preserves the current selection across a refresh if that name
        is still labeled, rather than always resetting to "All people"."""
        import face_index
        current = self.search_person_box.currentText()
        self.search_person_box.clear()
        self.search_person_box.addItem("All people")
        self.search_person_box.addItems([p["name"] for p in face_index.list_people()])
        idx = self.search_person_box.findText(current)
        if idx >= 0:
            self.search_person_box.setCurrentIndex(idx)

    def _build_editor_tab(self):
        central_widget = QWidget()
        main_layout = QVBoxLayout(central_widget)

        workspace_layout = QHBoxLayout()
        
        # ------------------ COLUMN 1: MEDIA BIN ------------------
        left_panel = QVBoxLayout()
        left_panel.addWidget(QLabel("📂 Media Directory Storage"))

        self.search_box = QLineEdit()
        self.search_box.setPlaceholderText("🔍 Search videos...")
        self.search_box.textChanged.connect(self.filter_media_list)
        left_panel.addWidget(self.search_box)

        self.media_list = QListWidget()
        self.media_list.itemClicked.connect(self.play_selected_video)
        left_panel.addWidget(self.media_list)
        
        # Manual Fallback button we built earlier
        self.btn_add_manual = self.add_button(left_panel, "➕ Add Video Files Manually", self.manually_import_video)
        self.btn_remove_selected = self.add_button(
            left_panel, "🗑️ Remove Selected from List", self.remove_selected_from_list)
        self.btn_refresh_list = self.add_button(left_panel, "🔄 Refresh Media List", self.load_iphone_photos)
        self.btn_open_import_folder = self.add_button(left_panel, "📂 Open Import Folder", self.open_import_folder)
        self.btn_open_export_folder = self.add_button(left_panel, "📂 Open Export Folder", self.open_export_folder)


        # COLUMN 2: PLAYER
        middle_panel = QVBoxLayout()
        middle_panel.addWidget(QLabel("📺 Video Preview Window"))
        self.video_widget = QVideoWidget()
        self.video_widget.setStyleSheet("background-color: black; border: 1px solid #444;")
        
        self.player = QMediaPlayer()
        self.audio_output = QAudioOutput()
        self.player.setAudioOutput(self.audio_output)
        self.player.setVideoOutput(self.video_widget)
        
        # Track changing positions to feed the timeline scrubbing layers
        self.player.positionChanged.connect(self.update_timeline_playhead)
        self.player.durationChanged.connect(self.update_video_duration)
        self.video_duration_cached = 0

        player_controls = QHBoxLayout()
        self.btn_play = QPushButton("▶ Play")
        self.btn_pause = QPushButton("⏸ Pause")
        self.btn_play.clicked.connect(self.player.play)
        self.btn_pause.clicked.connect(self.player.pause)
        player_controls.addWidget(self.btn_play)
        player_controls.addWidget(self.btn_pause)
        
        middle_panel.addWidget(self.video_widget, 4)
        middle_panel.addLayout(player_controls, 1)
        
        # COLUMN 3: PARAMETERS
        right_panel = QVBoxLayout()
        right_panel.addWidget(QLabel("⚙️ Parameter Settings"))
        
        right_panel.addWidget(QLabel("Padding Cushion (Seconds):"))
        self.margin_box = QDoubleSpinBox()
        self.margin_box.setRange(0.0, 2.0)
        self.margin_box.setValue(0.2)
        self.margin_box.setSingleStep(0.05)
        self.margin_box.valueChanged.connect(lambda: self.generate_timeline_preview())
        right_panel.addWidget(self.margin_box)

        right_panel.addWidget(QLabel("Silence Sensitivity (% volume, higher = cuts more):"))
        self.sensitivity_box = QDoubleSpinBox()
        self.sensitivity_box.setRange(0.0, 50.0)  # 0% = only cut genuine, complete digital silence
        self.sensitivity_box.setValue(4.0)
        self.sensitivity_box.setSingleStep(0.1)
        self.sensitivity_box.setDecimals(2)  # lets you type e.g. 0.05 for even finer control than the step size
        self.sensitivity_box.setSuffix("%")
        self.sensitivity_box.valueChanged.connect(lambda: self.generate_timeline_preview())
        right_panel.addWidget(self.sensitivity_box)

        right_panel.addWidget(QLabel("Transcription Language:"))
        self.transcribe_lang_box = QComboBox()
        self.transcribe_lang_box.addItems(list(TRANSCRIPTION_LANGUAGES.keys()))
        self.transcribe_lang_box.setCurrentText("English")
        right_panel.addWidget(self.transcribe_lang_box)

        self.translate_checkbox = QCheckBox("🌐 Translate transcription to English")
        right_panel.addWidget(self.translate_checkbox)

        self.transcribe_location_checkbox = QCheckBox("📍 Choose save location")
        right_panel.addWidget(self.transcribe_location_checkbox)

        self.btn_transcribe = self.add_button(right_panel, "🎙️ Transcribe Audio", self.transcribe_selected_video)

        self.transcript_view = QTextEdit()
        self.transcript_view.setReadOnly(True)
        self.transcript_view.setPlaceholderText("Transcript will appear here after transcribing...")
        self.transcript_view.setMaximumHeight(150)
        right_panel.addWidget(self.transcript_view)

        self.whisper_model = None  # Lazy-loaded on first transcription

        # Auto-unload after this long with no completed transcription, so the
        # ~1.4GB model doesn't sit resident for the rest of the app's session.
        # Restarted only once a transcription actually finishes, not when one
        # begins - a long transcribe can't get its own model unloaded mid-run.
        self.whisper_unload_timer = QTimer(self)
        self.whisper_unload_timer.setSingleShot(True)
        self.whisper_unload_timer.timeout.connect(self.unload_whisper_model)

        self.status_label = QLabel("Ready.")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("color: #aaa; font-style: italic;")
        right_panel.addWidget(self.status_label)
        right_panel.addStretch()
        
        self.transcribe_after_render_checkbox = QCheckBox("🎙️ Also transcribe rendered output")
        right_panel.addWidget(self.transcribe_after_render_checkbox)

        self.render_location_checkbox = QCheckBox("📍 Choose save location")
        right_panel.addWidget(self.render_location_checkbox)

        self.skip_hevc_check_checkbox = QCheckBox("⏭️ Skip conversion")
        self.skip_hevc_check_checkbox.setChecked(True)
        right_panel.addWidget(self.skip_hevc_check_checkbox)

        self.magic_button = self.add_button(right_panel, "🚀 Run Auto-Cut", self.start_magic_cut, primary=True)
        
        workspace_layout.addLayout(left_panel, 1)
        workspace_layout.addLayout(middle_panel, 2)
        workspace_layout.addLayout(right_panel, 1)
        main_layout.addLayout(workspace_layout, 5)
        
        # BOTTOM INTERACTIVE TIMELINE CONTAINER
        main_layout.addWidget(QLabel("📊 Click timeline track to scrub video playback:"))
        self.timeline_tracker = VisualTimeline(self.player) # Pass player inside
        main_layout.addWidget(self.timeline_tracker, 1)

        return central_widget

    def _build_search_tab(self):
        tab = QWidget()
        layout = QVBoxLayout(tab)

        # --- Filter row: query, type, result count, date range, search button ---
        filter_row = QHBoxLayout()

        self.search_query_box = QLineEdit()
        self.search_query_box.setPlaceholderText('Search your library... e.g. "dog jumping in the lake"')
        self.search_query_box.returnPressed.connect(self._run_search)
        filter_row.addWidget(self.search_query_box, 3)

        self.search_type_box = QComboBox()
        self.search_type_box.addItems(["All types", "Video only", "Image only"])
        filter_row.addWidget(self.search_type_box)

        # Starts with just "All people" - populated for real by
        # _refresh_search_person_filter(), called lazily whenever this tab
        # becomes active (see the tabs.currentChanged wiring), not at startup -
        # same lazy-import discipline as everywhere else face_index is touched.
        self.search_person_box = QComboBox()
        self.search_person_box.addItem("All people")
        filter_row.addWidget(self.search_person_box)

        self.search_top_box = QSpinBox()
        self.search_top_box.setRange(1, 200)
        self.search_top_box.setValue(20)
        self.search_top_box.setPrefix("Results: ")
        filter_row.addWidget(self.search_top_box)

        # Starts empty/unfiltered - the first click jumps to a real default
        # (a year ago / today) rather than leaving the field on Qt's own
        # 1752-09-14 minimum, but doesn't apply any date filter until touched.
        filter_row.addWidget(QLabel("From:"))
        self.search_after_box = DefaultingDateEdit("From...", lambda: QDate.currentDate().addYears(-1))
        filter_row.addWidget(self.search_after_box)

        filter_row.addWidget(QLabel("To:"))
        self.search_before_box = DefaultingDateEdit("To...", QDate.currentDate)
        filter_row.addWidget(self.search_before_box)

        self.btn_search = self.add_button(filter_row, "🔍 Search", self._run_search, primary=True)

        layout.addLayout(filter_row)

        self.search_status_label = QLabel("Type a query above to search your library.")
        self.search_status_label.setStyleSheet("color: #aaa; font-style: italic;")
        layout.addWidget(self.search_status_label)

        # --- Body: results grid (left) + shared preview pane (right) ---
        body_layout = QHBoxLayout()

        self.search_results_container = QWidget()
        self.search_results_grid = QGridLayout(self.search_results_container)
        self.search_results_grid.setSpacing(8)

        self.search_scroll_area = ResultsScrollArea()
        self.search_scroll_area.setWidgetResizable(True)
        self.search_scroll_area.setWidget(self.search_results_container)
        self._grid_relayout_timer = QTimer(self)
        self._grid_relayout_timer.setSingleShot(True)
        self._grid_relayout_timer.timeout.connect(self._relayout_result_grid)
        # Debounced - a window drag-resize would otherwise relayout on every pixel.
        self.search_scroll_area.resized.connect(lambda: self._grid_relayout_timer.start(100))
        body_layout.addWidget(self.search_scroll_area, 1)

        preview_panel = QVBoxLayout()
        preview_panel.addWidget(QLabel("Preview"))

        self.search_preview_video = QVideoWidget()
        self.search_preview_video.setStyleSheet("background-color: black; border: 1px solid #444;")
        self.search_preview_player = QMediaPlayer()
        self.search_preview_audio = QAudioOutput()
        self.search_preview_player.setAudioOutput(self.search_preview_audio)
        self.search_preview_player.setVideoOutput(self.search_preview_video)
        preview_panel.addWidget(self.search_preview_video, 4)

        self.search_preview_image = QLabel()
        self.search_preview_image.setStyleSheet("background-color: black;")
        self.search_preview_image.hide()
        preview_panel.addWidget(self.search_preview_image, 4)

        # Same scrubbable timeline widget as the Editor tab, just without any
        # speech/silence chunk data - reused as a plain click-to-seek bar with
        # a live playhead, driven off this pane's own player instead.
        self.search_timeline = VisualTimeline(self.search_preview_player)
        self.search_preview_player.positionChanged.connect(self.search_timeline.set_progress)
        self.search_preview_player.durationChanged.connect(
            lambda duration_ms: self.search_timeline.set_timeline_data([], duration_ms))
        preview_panel.addWidget(self.search_timeline)

        preview_controls = QHBoxLayout()
        btn_preview_play = QPushButton("▶ Play")
        btn_preview_pause = QPushButton("⏸ Pause")
        btn_preview_play.clicked.connect(self.search_preview_player.play)
        btn_preview_pause.clicked.connect(self.search_preview_player.pause)
        preview_controls.addWidget(btn_preview_play)
        preview_controls.addWidget(btn_preview_pause)
        preview_panel.addLayout(preview_controls)

        # Disabled/unhighlighted until at least one result is checked -
        # add_button's primary=True styling dims automatically when disabled.
        self.btn_confirm_selection = self.add_button(
            preview_panel, "✅ Confirm Selection", self._confirm_selection, primary=True)
        self.btn_confirm_selection.setEnabled(False)

        body_layout.addLayout(preview_panel, 1)
        layout.addLayout(body_layout, 1)

        self.selected_search_results = {}  # file_path -> result dict, current selection only
        self.confirmed_search_paths = set()  # accumulates across every Confirm Selection click this session
        self.search_result_cards = []      # card widgets currently on screen, for relayout
        self.search_worker = None

        # Auto-unload after this long with no completed search - mirrors
        # unload_whisper_model's timer exactly (see main.py's WHISPER_IDLE_UNLOAD_MS
        # wiring). Restarted only once a search actually finishes, never at
        # search-start, so a slow search can't get its own model unloaded mid-run.
        self.clip_unload_timer = QTimer(self)
        self.clip_unload_timer.setSingleShot(True)
        self.clip_unload_timer.timeout.connect(self.unload_clip_model_slot)

        return tab

    def _run_search(self):
        query = self.search_query_box.text().strip()
        if not query:
            return
        if self.search_worker is not None and self.search_worker.isRunning():
            return  # a search is already in flight - let it finish first

        file_types = {"All types": None, "Video only": "video", "Image only": "image"}[
            self.search_type_box.currentText()]
        after = self.search_after_box.date().toString("yyyy-MM-dd") if self.search_after_box.is_active() else None
        before = self.search_before_box.date().toString("yyyy-MM-dd") if self.search_before_box.is_active() else None
        person = self.search_person_box.currentText()
        person = None if person == "All people" else person

        self.btn_search.setEnabled(False)
        self.search_status_label.setText(f'🔍 Searching for "{query}"...')

        self.search_worker = SearchWorker(query, self.search_top_box.value(), after, before, file_types, person)
        self.search_worker.results_ready.connect(self._on_search_results)
        self.search_worker.search_failed.connect(self._on_search_failed)
        self.search_worker.start()

    def _on_search_results(self, results):
        self.btn_search.setEnabled(True)
        self.clip_unload_timer.start(CLIP_IDLE_UNLOAD_MS)

        for card in self.search_result_cards:
            card.setParent(None)
        self.search_result_cards = []
        self.selected_search_results.clear()
        self.btn_confirm_selection.setEnabled(False)

        if not results:
            self.search_status_label.setText("No results - try a different query, or check the index has been built.")
            return

        # Surfaces smart_search()'s natural-language people-detection (2026-08-24) -
        # this parsing happens silently otherwise, so confirm what was actually
        # understood from the query text rather than leaving it a black box.
        matched_people = results[0].get("matched_people")
        people_note = f" - matched: {', '.join(matched_people)}" if matched_people else ""
        self.search_status_label.setText(f"{len(results)} results{people_note}")
        for result in results:
            self.search_result_cards.append(self._build_result_card(result))
        self._relayout_result_grid()

    def _on_search_failed(self, message):
        self.btn_search.setEnabled(True)
        self.clip_unload_timer.start(CLIP_IDLE_UNLOAD_MS)
        self.search_status_label.setText(f"❌ Search failed: {message}")

    def unload_clip_model_slot(self):
        import media_index
        media_index.unload_clip_model()

    def _relayout_result_grid(self):
        columns = columns_for_width(self.search_scroll_area.viewport().width())
        for i, card in enumerate(self.search_result_cards):
            self.search_results_grid.addWidget(card, i // columns, i % columns)

    def _build_result_card(self, result):
        card = QWidget()
        card.setFixedWidth(RESULT_CARD_WIDTH)
        card.setStyleSheet("background-color: #1c1c1c; border-radius: 8px;")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)

        thumb_path = result.get("thumb_path")
        pixmap = QPixmap(thumb_path) if thumb_path else None
        thumbnail = ResultThumbnail(pixmap, RESULT_THUMB_WIDTH, RESULT_THUMB_HEIGHT)
        thumbnail.preview_clicked.connect(lambda r=result: self._preview_result(r))
        thumbnail.select_toggled.connect(lambda checked, r=result: self._toggle_result_selection(r, checked))
        card_layout.addWidget(thumbnail)

        date_text = result.get("date_taken") or "no date on file"
        gps_text = f" · GPS {result['lat']:.4f},{result['lon']:.4f}" if result.get("lat") else ""
        duration_text = ""
        if result["media_type"] == "video" and result.get("duration_seconds") is not None:
            duration_text = f" · {self.format_duration(result['duration_seconds'])}"
        info_label = QLabel(f"{date_text}  ·  {result['media_type']}{duration_text}{gps_text}")
        info_label.setStyleSheet("color: #aaa; font-size: 11px;")
        info_label.setWordWrap(True)
        card_layout.addWidget(info_label)

        # Only present on smart_search() results (2026-08-24) that detected at
        # least one person in the query text/dropdown - makes the "prioritize
        # by mentioned people" ranking visible instead of a silent black box.
        if result.get("matched_people"):
            people_label = QLabel(f"👤 {', '.join(result['matched_people'])} ({result['match_count']} matched)")
            people_label.setStyleSheet("color: #9cf; font-size: 10px;")
            people_label.setWordWrap(True)
            card_layout.addWidget(people_label)

        path_label = QLabel(Path(result["file_path"]).name)
        path_label.setStyleSheet("color: #777; font-size: 10px;")
        path_label.setWordWrap(True)
        card_layout.addWidget(path_label)

        return card

    def _toggle_result_selection(self, result, checked):
        if checked:
            self.selected_search_results[result["file_path"]] = result
        else:
            self.selected_search_results.pop(result["file_path"], None)
        self.btn_confirm_selection.setEnabled(bool(self.selected_search_results))

    def _preview_result(self, result):
        self.search_preview_image.hide()
        if result["media_type"] == "video":
            self.search_preview_video.show()
            self.search_timeline.show()
            ts_ms = int((result.get("timestamp_seconds") or 0) * 1000)

            def seek_once_loaded(status):
                if status == QMediaPlayer.MediaStatus.LoadedMedia:
                    self.search_preview_player.setPosition(ts_ms)
                    self.search_preview_player.mediaStatusChanged.disconnect(seek_once_loaded)

            self.search_preview_player.mediaStatusChanged.connect(seek_once_loaded)
            self.search_preview_player.setSource(QUrl.fromLocalFile(result["file_path"]))
            self.search_preview_player.play()
        else:
            self.search_preview_player.stop()
            self.search_preview_video.hide()
            self.search_timeline.hide()
            self.search_preview_image.show()
            # Thumbnail, not the raw source - HEIC has no QPixmap codec, same
            # reason media_search.get_thumbnail() exists in the first place.
            pixmap = QPixmap(result.get("thumb_path") or result["file_path"])
            if not pixmap.isNull():
                self.search_preview_image.setPixmap(pixmap.scaled(
                    self.search_preview_image.size(), Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation))

    def _find_media_list_row_by_path(self, path):
        for i in range(self.media_list.count()):
            data = self.media_list.item(i).data(Qt.ItemDataRole.UserRole)
            if data and data.get("path") == path:
                return i
        return None

    def _confirm_selection(self):
        if not self.selected_search_results:
            return
        # score can be None - smart_search() (2026-08-24) skips CLIP scoring
        # entirely when a query was just a person's name with nothing left to
        # rank by. Scored results still sort first; None-scored ones group
        # together after them rather than crashing the comparison.
        ranked = sorted(
            self.selected_search_results.values(),
            key=lambda r: (r["score"] is not None, r["score"] or 0), reverse=True,
        )

        for i, result in enumerate(ranked):
            path = result["file_path"]
            self.confirmed_search_paths.add(path)
            existing_row = self._find_media_list_row_by_path(path)
            if existing_row is not None:
                # Already in the list (HDD scan or an earlier import) - move it
                # to the top instead of adding a duplicate entry.
                item = self.media_list.takeItem(existing_row)
                data = item.data(Qt.ItemDataRole.UserRole)
                data["manual"] = True  # pins it at the top across a refresh/restart, like any manual import
                item.setData(Qt.ItemDataRole.UserRole, data)
                self.media_list.insertItem(i, item)
            else:
                self.add_media_item(Path(path).name, path, manual=True, index=i)

        save_manual_imports_to_disk(self.profile["app_data_dir"], self.get_manual_items())

        # Hide everything except every result confirmed so far THIS SESSION -
        # not just this click - so a second Confirm Selection doesn't hide
        # picks a previous one already surfaced. "🔄 Refresh Media List"
        # already rebuilds from scratch with manual items (now including
        # these) pinned at the top first, so it doubles as the "reveal the
        # rest, keep every selection on top" action without needing a new button.
        for i in range(self.media_list.count()):
            row_item = self.media_list.item(i)
            data = row_item.data(Qt.ItemDataRole.UserRole)
            row_item.setHidden((data.get("path") if data else None) not in self.confirmed_search_paths)

        self.selected_search_results.clear()
        self.btn_confirm_selection.setEnabled(False)
        for card in self.search_result_cards:
            thumbnail = card.findChild(ResultThumbnail)
            if thumbnail:
                thumbnail.set_selected(False)

        self.tabs.setCurrentIndex(0)
        self.status_label.setText(
            f"Showing {len(ranked)} confirmed item(s) - hit '🔄 Refresh Media List' to reveal the rest again.")

    def _build_people_tab(self):
        """Browse face-recognition clusters produced by face_index.py and
        name them. Deliberately no QThread here (unlike the Search tab's
        SearchWorker) - browsing/labeling is small sqlite + JPEG reads, no
        model inference, so it stays synchronous on the UI thread the same
        way the Search tab already builds its result cards. face_index is
        imported locally in each handler (not at module level) so opening
        this tab, or even launching the app, never pulls in numpy/hdbscan
        unless the tab is actually used - and even then, insightface itself
        (the heavy part) is never touched here at all, only by the separate
        build_face_index() indexing pass."""
        tab = QWidget()
        layout = QVBoxLayout(tab)

        top_row = QHBoxLayout()
        self.people_status_label = QLabel("Loading...")
        self.people_status_label.setStyleSheet("color: #aaa; font-style: italic;")
        top_row.addWidget(self.people_status_label, 1)
        self.btn_refresh_recluster = self.add_button(top_row, "🔄 Refresh && Recluster", self._refresh_people_tab)
        self.btn_find_matches = self.add_button(top_row, "🔍 Find Matches", self._find_people_matches)
        layout.addLayout(top_row)

        body_layout = QHBoxLayout()

        left_panel = QVBoxLayout()
        self.people_groups_header_label = QLabel("Pending clusters + named people")
        left_panel.addWidget(self.people_groups_header_label)
        self.people_groups_list = QListWidget()
        self.people_groups_list.itemClicked.connect(self._on_people_group_selected)
        left_panel.addWidget(self.people_groups_list)
        # No stretch - width is capped to its longest current line (see
        # _fit_people_groups_list_width(), called after every reload) rather than
        # sharing leftover horizontal space, so it stops crowding the name box and
        # preview pane out (2026-08-21 request).
        body_layout.addLayout(left_panel, 0)

        center_panel = QVBoxLayout()
        self.people_faces_container = QWidget()
        self.people_faces_grid = QGridLayout(self.people_faces_container)
        self.people_faces_grid.setSpacing(8)
        self.people_scroll_area = ResultsScrollArea()
        self.people_scroll_area.setWidgetResizable(True)
        self.people_scroll_area.setWidget(self.people_faces_container)
        self._people_grid_relayout_timer = QTimer(self)
        self._people_grid_relayout_timer.setSingleShot(True)
        self._people_grid_relayout_timer.timeout.connect(self._relayout_people_grid)
        self.people_scroll_area.resized.connect(lambda: self._people_grid_relayout_timer.start(100))
        center_panel.addWidget(self.people_scroll_area, 1)

        body_layout.addLayout(center_panel, 2)

        preview_panel = QVBoxLayout()
        preview_panel.addWidget(QLabel("Preview (source photo)"))
        self.people_preview_image = QLabel()
        self.people_preview_image.setStyleSheet("background-color: black;")
        self.people_preview_image.setAlignment(Qt.AlignmentFlag.AlignCenter)  # the scaled
        # pixmap rarely exactly fills the label (aspect ratio varies per photo) -
        # without this it sits top-left instead of centered in the empty space
        preview_panel.addWidget(self.people_preview_image, 1)
        # Bumped from 1 to 2 (now equal to center_panel) now that the pending-list
        # column no longer competes for stretch space - the width it used to take
        # goes to this pane instead (2026-08-21 request).
        body_layout.addLayout(preview_panel, 2)

        layout.addLayout(body_layout, 1)

        # Full tab width, not nested inside center_panel - it used to share center_panel's
        # (roughly half the window) width with 4 buttons, squashing the QLineEdit down to
        # almost nothing since buttons don't shrink below their text but the name box does
        # (2026-08-21 request). Living below body_layout instead doesn't change what it acts
        # on - that's still whatever's selected in the left panel / rendered in the grid above.
        label_row = QHBoxLayout()
        self.people_name_box = QLineEdit()
        self.people_name_box.setPlaceholderText("Name this person...")
        self.people_name_box.returnPressed.connect(self._on_people_name_box_enter)
        # Populated/refreshed in _load_people_lists() with current labeled-person
        # names, so it narrows as you type instead of requiring the full name.
        self.people_name_completer = QCompleter([], self)
        self.people_name_completer.setCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)
        self.people_name_box.setCompleter(self.people_name_completer)
        label_row.addWidget(self.people_name_box, 1)
        self.btn_confirm_label = self.add_button(label_row, "✅ Confirm", self._confirm_people_label, primary=True)
        discard_rename_col = QVBoxLayout()  # stacked, not another column in the row - user's explicit layout ask
        self.btn_discard_cluster = self.add_button(discard_rename_col, "🗑️ Discard cluster", self._discard_people_cluster)
        self.btn_rename_person = self.add_button(discard_rename_col, "✏️ Rename", self._rename_people_person)
        self.btn_remove_from_person = self.add_button(discard_rename_col, "🚫 Remove from Person", self._remove_from_person)
        label_row.addLayout(discard_rename_col)
        self.btn_apply_matches = self.add_button(label_row, "✅ Apply Selected Matches", self._apply_people_matches, primary=True)
        self.btn_undo = self.add_button(label_row, "↩️ Undo", self._undo_last_people_action)
        # Shown/hidden per selection type (2026-08-24 request), not just greyed out -
        # Confirm/Discard only make sense for a pending cluster, Rename/Remove only for
        # an already-labeled person, Apply Selected Matches only in Find-Matches mode.
        # Nothing is selected yet at startup, so all four start hidden; Undo is the one
        # exception - it's not tied to a selection type, just enabled/disabled normally.
        for btn in (self.btn_confirm_label, self.btn_discard_cluster, self.btn_rename_person,
                    self.btn_remove_from_person, self.btn_apply_matches):
            btn.setVisible(False)
        self.btn_undo.setEnabled(False)
        layout.addLayout(label_row)

        self.people_current_group = None  # {"type": "cluster"/"person"/"matches", ...}
        self.people_excluded_faces = set()  # face_vector_index values un-checked in the currently-open cluster
        self.people_current_matches_plan = []  # last propose_matches() result - a plan, nothing written yet
        self.people_accepted_matches = set()  # face_vector_index values checked "yes, this is who it says" in match review
        self.people_face_cards = []
        self.people_displayed_faces = []  # face dicts actually rendered right now - what Confirm/Discard may act on
        self.people_recluster_worker = None
        self.people_last_action = None  # {"description": ..., "faces": {fvi: {"person_id", "cluster_id", "discarded"}}}
                                         # - single-level undo only; a new action overwrites this, it doesn't stack

        self._load_people_lists()
        return tab

    def _refresh_people_tab(self):
        """Triggered by the "Refresh & Recluster" button - runs the real
        HDBSCAN reclustering pass in the background (ReclusterWorker), NOT
        on every tab open/label action. Measured live: ~2 minutes over the
        real ~14,500-face library - freezing the whole app for that on every
        launch, or after every single label, was a real bug, not a style
        choice (see the 2026-08-20 diary/memory note). Labeling/discarding/
        applying matches only need _load_people_lists() - a fast DB read -
        since they don't change what a cluster of *already-clustered* faces
        looks like, only which faces are still pending."""
        if self.people_recluster_worker is not None and self.people_recluster_worker.isRunning():
            return
        # Disabled for the duration - a second connection to the same sqlite db while
        # this one holds it (EXCLUSIVE locking mode) would fail, not just double-run.
        self.btn_refresh_recluster.setEnabled(False)
        self.btn_find_matches.setEnabled(False)
        # Hidden rather than just disabled, matching the selection-based show/hide
        # pattern - nothing stays selected once _load_people_lists() reloads after
        # this finishes, so there's nothing to correctly re-show them for anyway.
        for btn in (self.btn_confirm_label, self.btn_discard_cluster, self.btn_rename_person,
                    self.btn_remove_from_person, self.btn_apply_matches):
            btn.setVisible(False)
        self.people_groups_list.setEnabled(False)
        self.people_status_label.setText("Reclustering... can take a couple of minutes over a real library.")

        self.people_recluster_worker = ReclusterWorker()
        self.people_recluster_worker.finished_ok.connect(self._on_recluster_finished)
        self.people_recluster_worker.recluster_failed.connect(self._on_recluster_failed)
        self.people_recluster_worker.start()

    def _on_recluster_finished(self, cluster_count):
        self.btn_refresh_recluster.setEnabled(True)
        self.btn_find_matches.setEnabled(True)
        self.people_groups_list.setEnabled(True)
        self._load_people_lists()

    def _on_recluster_failed(self, message):
        self.btn_refresh_recluster.setEnabled(True)
        self.btn_find_matches.setEnabled(True)
        self.people_groups_list.setEnabled(True)
        self.people_status_label.setText(f"Reclustering failed: {message}")

    def _load_people_lists(self):
        """Fast, DB-only reload of both lists - does NOT recompute
        clustering (see _refresh_people_tab for the slow, explicit-only
        path). Used at startup and after every labeling action. Safe to
        call with zero faces indexed yet (a fresh install, or before
        build_face_index.py has ever been run) - everything below degrades
        to empty lists rather than erroring."""
        import face_index
        self.people_groups_list.clear()
        clusters = face_index.list_unlabeled_clusters()
        people = face_index.list_people()

        for cluster, suffix in self._ordered_pending_clusters(clusters):
            label = "Unclustered" if cluster["cluster_id"] == -1 else f"C {cluster['cluster_id']}"
            item = QListWidgetItem(f"{label} ({cluster['count']}){suffix}")
            item.setData(Qt.ItemDataRole.UserRole, {"type": "cluster", "cluster_id": cluster["cluster_id"]})
            self.people_groups_list.addItem(item)

        for p in people:
            item = QListWidgetItem(f"👤 {p['name']} ({p['face_count']})")
            item.setData(Qt.ItemDataRole.UserRole, {"type": "person", "person_id": p["person_id"], "name": p["name"]})
            self.people_groups_list.addItem(item)

        self._fit_people_groups_list_width()
        # Re-set (not mutated in place) so the completer's popup reflects renames/merges/new
        # people immediately - stale autocomplete suggestions would be actively misleading.
        self.people_name_completer.setModel(QStringListModel([p["name"] for p in people], self))

    def _fit_people_groups_list_width(self):
        """Caps the pending/people list to its longest current line instead of
        sharing stretch space with the rest of the tab - user's explicit ask
        (2026-08-21), since a wide list (hundreds of clusters, one with a long
        suggested name) was squeezing the name box and preview pane. Recomputed
        every reload since the real text (cluster ids, suggested names) changes."""
        metrics = self.people_groups_list.fontMetrics()
        widths = [metrics.horizontalAdvance(self.people_groups_list.item(i).text())
                  for i in range(self.people_groups_list.count())]
        widths.append(metrics.horizontalAdvance(self.people_groups_header_label.text()))
        longest = max(widths, default=0)
        # Padding for the list's own frame margins, item icon indent, and a
        # vertical scrollbar - measured to comfortably fit them, not exact.
        self.people_groups_list.setMaximumWidth(longest + 48)

    def _ordered_pending_clusters(self, clusters):
        """Reorders the pending-clusters list so likely-related ones sit
        next to each other, instead of scattered purely by size - directly
        answers "many small clusters are the same person" (2026-08-20/21
        diary notes) by surfacing that adjacency instead of leaving it to be
        found by luck while scrolling. Three tiers, each falling through to
        the next: (1) clusters matching an already-labeled person, grouped
        by that person's name; (2) clusters matching each other (neither
        matches a labeled person yet) via suggest_cluster_groupings(),
        placed adjacent in pairs; (3) everything else, original size order.
        "Unclustered" (-1) is excluded from all of this - a large mixed bag
        with no single identity to compare - and always appended last.
        Returns (cluster_dict, label_suffix) tuples."""
        import face_index
        real_clusters = [c for c in clusters if c["cluster_id"] != -1]
        unclustered = [c for c in clusters if c["cluster_id"] == -1]
        by_id = {c["cluster_id"]: c for c in real_clusters}

        person_suggestions = face_index.suggest_people_for_all_clusters()
        cluster_suggestions = {
            s["cluster_id"]: s for s in face_index.suggest_cluster_groupings()
            if s["cluster_id"] not in person_suggestions
        }

        tier1_ids = [cid for cid in by_id if cid in person_suggestions]
        name_group_size = {}
        for cid in tier1_ids:
            name = person_suggestions[cid]["name"]
            name_group_size[name] = name_group_size.get(name, 0) + 1
        # Biggest suggested-group first (most pending clusters to clear for one name in
        # one pass), then within a name group, most-confident suggestion first.
        tier1_ids.sort(key=lambda cid: (-name_group_size[person_suggestions[cid]["name"]],
                                         person_suggestions[cid]["name"],
                                         -person_suggestions[cid]["similarity"]))

        remaining_ids = [cid for cid in by_id if cid not in person_suggestions]  # original (size) order preserved
        tier2_ordered, visited = [], set()
        for cid in remaining_ids:
            if cid in visited or cid not in cluster_suggestions:
                continue
            tier2_ordered.append(cid)
            visited.add(cid)
            sibling_id = cluster_suggestions[cid]["suggested_cluster_id"]
            if sibling_id in by_id and sibling_id not in visited and sibling_id not in person_suggestions:
                tier2_ordered.append(sibling_id)
                visited.add(sibling_id)
        tier3_ids = [cid for cid in remaining_ids if cid not in visited]

        # Suffix stays short - list width is at a premium with hundreds of pending
        # clusters on screen at once; the similarity score reappears in the status
        # label once a cluster is actually selected, so it isn't needed twice.
        ordered = []
        for cid in tier1_ids:
            s = person_suggestions[cid]
            ordered.append((by_id[cid], f" → {s['name']}?"))
        for cid in tier2_ordered:
            s = cluster_suggestions[cid]
            ordered.append((by_id[cid], f" ≈ C{s['suggested_cluster_id']}"))
        for cid in tier3_ids:
            ordered.append((by_id[cid], ""))
        for c in unclustered:
            ordered.append((c, ""))
        return ordered

    def _on_people_group_selected(self, item):
        import face_index
        data = item.data(Qt.ItemDataRole.UserRole)
        self.people_current_group = data
        self.people_excluded_faces = set()
        self.people_current_matches_plan = []
        self.people_accepted_matches = set()
        self.people_name_box.clear()

        is_cluster = data["type"] == "cluster"
        is_person = data["type"] == "person"
        # Hidden, not just disabled - Confirm/Discard belong to reviewing a pending
        # cluster, Rename/Remove belong to browsing an already-labeled person; only
        # one pair is ever relevant to what's currently selected (2026-08-24 request).
        self.btn_confirm_label.setVisible(is_cluster)
        self.btn_discard_cluster.setVisible(is_cluster)
        self.btn_rename_person.setVisible(is_person)
        self.btn_remove_from_person.setVisible(is_person)
        self.btn_apply_matches.setVisible(False)
        if is_cluster:
            faces = face_index.get_faces_for_cluster(data["cluster_id"])
        else:
            faces = face_index.get_faces_for_person(data["person_id"])
        # Cluster faces start all-checked (opt-out exclude, most faces genuinely belong).
        # A labeled person's faces start all-UNCHECKED instead (2026-08-24 request) -
        # nothing's presumed wrong, so nothing starts marked; checking one is how you flag
        # it for Remove from Person. people_excluded_faces means "unchecked" either way, so
        # starting a person's set as everything-displayed achieves "all unchecked" using the
        # exact same toggle/act-on-what's-not-excluded logic Confirm already uses.
        self._show_people_faces(faces, default_selected=is_cluster)
        if is_person:
            self.people_excluded_faces = {f["face_vector_index"] for f in self.people_displayed_faces}

        if is_person:
            # Pre-fill with the current name so it's a quick edit, not a
            # retype - Rename (or Enter) sends whatever's in the box.
            self.people_name_box.setText(data["name"])

        # Pre-fill the name box with a suggested existing person, cluster-vs-labeled-person
        # (the cluster-level counterpart to Find Matches, which works per-face) - not
        # meaningful for "Unclustered" (-1), a large mixed bag of many different people,
        # not one identity to compare as a single centroid. Still just a suggestion - the
        # name box stays fully editable, and nothing is written unless Confirm is pressed.
        if is_cluster and data["cluster_id"] != -1:
            suggestion = face_index.suggest_person_for_cluster(data["cluster_id"])
            if suggestion:
                self.people_name_box.setText(suggestion["name"])
                self.people_status_label.setText(
                    f"{self.people_status_label.text()} - suggested: {suggestion['name']} "
                    f"({suggestion['similarity']:.2f} similarity, confirm or type a different name)")

        # Clicking a list item leaves keyboard focus on the list itself, so Enter
        # would activate the list (not the name box) rather than confirm/rename -
        # move focus into the box now so Enter works immediately after selecting,
        # with no extra click needed. selectAll() so a pre-filled suggestion/name
        # can be either accepted as-is (just press Enter) or typed straight over.
        self.people_name_box.setFocus()
        self.people_name_box.selectAll()

    def _find_people_matches(self):
        """Runs propose_matches() (writes nothing) and shows every suggestion
        for review - matches start CHECKED (opt-out reject, 2026-08-24
        request), same default as a cluster's faces: propose_matches() only
        ever proposes something above DEFAULT_MATCH_THRESHOLD in the first
        place, so most suggestions genuinely are correct - uncheck the rare
        wrong one rather than having to check every good one by hand. Apply
        Selected Matches still only writes what's checked at click time, so
        this only changes the starting point, not the actual safety net."""
        import face_index
        self.people_current_group = {"type": "matches"}
        self.people_excluded_faces = set()
        self.people_groups_list.clearSelection()
        # Only Apply Selected Matches belongs to this mode - hide the cluster/person
        # buttons rather than just disabling them, same pattern as _on_people_group_selected.
        self.btn_confirm_label.setVisible(False)
        self.btn_discard_cluster.setVisible(False)
        self.btn_rename_person.setVisible(False)
        self.btn_remove_from_person.setVisible(False)
        self.btn_apply_matches.setVisible(True)

        self.people_current_matches_plan = face_index.propose_matches()
        # Pre-accept exactly what's about to be displayed (respecting the same
        # MAX_FACES_TO_DISPLAY cap _show_match_cards() itself applies) - never the
        # full plan, so Apply can never act on a match that was never actually shown.
        self.people_accepted_matches = {
            m["face_vector_index"] for m in self.people_current_matches_plan[:MAX_FACES_TO_DISPLAY]
        }
        self.btn_apply_matches.setEnabled(bool(self.people_current_matches_plan))
        total = len(self.people_current_matches_plan)
        if total > MAX_FACES_TO_DISPLAY:
            self.people_status_label.setText(
                f"Showing {MAX_FACES_TO_DISPLAY} of {total} suggested matches (all checked) - uncheck any that are "
                f"wrong, Apply, then Find Matches again for more.")
        elif total:
            self.people_status_label.setText(f"{total} suggested match(es), all checked - uncheck any that are wrong, then Apply.")
        else:
            self.people_status_label.setText("No suggested matches right now (label a few people first, or nothing new is close enough).")
        self._show_match_cards(self.people_current_matches_plan)

    def _show_people_faces(self, faces, default_selected=True):
        import face_index
        for card in self.people_face_cards:
            card.setParent(None)
        self.people_face_cards = []
        # The exact set Confirm/Discard/Remove are allowed to act on - never the full
        # underlying group, only what actually got rendered and reviewable. See
        # face_index.label_faces()/discard_faces() for why this distinction exists.
        self.people_displayed_faces = faces[:MAX_FACES_TO_DISPLAY]
        crops_dir = face_index.get_face_crops_dir()
        for face in self.people_displayed_faces:
            self.people_face_cards.append(self._build_face_card(face, crops_dir, default_selected))
        self._relayout_people_grid()
        if faces:
            if len(faces) > MAX_FACES_TO_DISPLAY:
                self.people_status_label.setText(
                    f"Showing {MAX_FACES_TO_DISPLAY} of {len(faces)} faces - too many to display at once. "
                    f"Label or discard these, then reopen the group to see more.")
            else:
                self.people_status_label.setText(f"Showing {len(faces)} face(s).")

    def _relayout_people_grid(self):
        columns = columns_for_width(self.people_scroll_area.viewport().width())
        for i, card in enumerate(self.people_face_cards):
            self.people_faces_grid.addWidget(card, i // columns, i % columns)

    def _build_face_card(self, face, crops_dir, default_selected=True):
        card = QWidget()
        card.setFixedWidth(RESULT_CARD_WIDTH)
        card.setStyleSheet("background-color: #1c1c1c; border-radius: 8px;")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)

        pixmap = QPixmap(str(crops_dir / face["crop_filename"]))
        thumbnail = ResultThumbnail(pixmap, RESULT_THUMB_WIDTH, RESULT_THUMB_HEIGHT)
        thumbnail.preview_clicked.connect(lambda f=face: self._preview_people_face(f))
        # For a pending cluster: checked = "keep this face in the label" (the default -
        # most faces in a real cluster genuinely belong), unchecking excludes the rare
        # stray one. For browsing an already-labeled person (default_selected=False,
        # 2026-08-24 request): the opposite - nothing's wrong by default, so nothing
        # starts checked; checking a face marks it for Remove from Person instead.
        # Same underlying people_excluded_faces set either way (see
        # _on_people_group_selected's is_person branch for how the starting state differs).
        thumbnail.select_toggled.connect(lambda checked, f=face: self._toggle_people_face_excluded(f, checked))
        thumbnail.set_selected(default_selected)
        card_layout.addWidget(thumbnail)

        source = self._source_badge(face["file_path"])
        info_label = QLabel(f"{source} score {face['det_score']:.2f} · {face['width']}x{face['height']}")
        info_label.setStyleSheet("color: #777; font-size: 10px;")
        card_layout.addWidget(info_label)

        return card

    def _source_badge(self, file_path):
        """Drive-portable paths (config.to_portable_path) start with 'DRIVE::' -
        anything else is local (Mac Photos library etc.) - both feed the same
        list_library_media() pool for indexing, this is purely a display hint."""
        return "💾 Drive" if str(file_path).startswith("DRIVE::") else "💻 Local"

    def _toggle_people_face_excluded(self, face, checked):
        if checked:
            self.people_excluded_faces.discard(face["face_vector_index"])
        else:
            self.people_excluded_faces.add(face["face_vector_index"])

    def _show_match_cards(self, plan):
        import face_index
        for card in self.people_face_cards:
            card.setParent(None)
        self.people_face_cards = []
        crops_dir = face_index.get_face_crops_dir()
        # Rendering is capped for the same performance reason as _show_people_faces -
        # apply_matches() itself stays correct either way, since it only ever acts on
        # whatever's in people_accepted_matches, which can only contain faces that were
        # actually displayed and ticked.
        for match in plan[:MAX_FACES_TO_DISPLAY]:
            self.people_face_cards.append(self._build_match_card(match, crops_dir))
        self._relayout_people_grid()

    def _build_match_card(self, match, crops_dir):
        card = QWidget()
        card.setFixedWidth(RESULT_CARD_WIDTH)
        card.setStyleSheet("background-color: #1c1c1c; border-radius: 8px;")
        card_layout = QVBoxLayout(card)
        card_layout.setContentsMargins(8, 8, 8, 8)

        pixmap = QPixmap(str(crops_dir / match["crop_filename"]))
        thumbnail = ResultThumbnail(pixmap, RESULT_THUMB_WIDTH, RESULT_THUMB_HEIGHT)
        thumbnail.preview_clicked.connect(lambda m=match: self._preview_people_face(m))
        # Starts checked, matching people_accepted_matches being pre-populated in
        # _find_people_matches() - visual state and the actual accepted set agree.
        thumbnail.select_toggled.connect(lambda checked, m=match: self._toggle_people_match_accepted(m, checked))
        thumbnail.set_selected(True)
        card_layout.addWidget(thumbnail)

        source = self._source_badge(match["file_path"])
        info_label = QLabel(f"{source} → {match['proposed_person_name']} ({match['similarity']:.2f})")
        info_label.setStyleSheet("color: #9cf; font-size: 10px;")
        card_layout.addWidget(info_label)

        return card

    def _toggle_people_match_accepted(self, match, checked):
        if checked:
            self.people_accepted_matches.add(match["face_vector_index"])
        else:
            self.people_accepted_matches.discard(match["face_vector_index"])

    def _apply_people_matches(self):
        import face_index
        if not self.people_current_matches_plan:
            return
        self._snapshot_for_undo(self.people_accepted_matches, f"applied {len(self.people_accepted_matches)} match(es)")
        count = face_index.apply_matches(self.people_current_matches_plan, self.people_accepted_matches)
        self.people_status_label.setText(f"Applied {count} accepted match(es).")
        self.people_current_matches_plan = []
        self.people_accepted_matches = set()
        self.btn_apply_matches.setEnabled(False)
        self._load_people_lists()

    def _preview_people_face(self, face):
        # file_path in the faces table is stored in the same portable "DRIVE::..." form
        # as everything else in the shared index (see config.to_portable_path) for files
        # that live on the external drive - most of them, in practice. Has to be resolved
        # back to a real, currently-mounted path before opening, same as the Search tab's
        # results already do via media_search - a raw "DRIVE::..." string isn't a real
        # filesystem path on its own.
        real_path = resolve_portable_path(face["file_path"], EXTERNAL_DRIVE_LABEL)
        if real_path is None:
            self.people_preview_image.setText(f"Drive '{EXTERNAL_DRIVE_LABEL}' not connected")
            return

        # PIL, not QPixmap directly - a source photo can be HEIC, which QPixmap has no
        # codec for (the same reason media_search.get_thumbnail()/get_thumbnail_dir()
        # exist for the Search tab) - converted through QImage instead of relying on a
        # pre-generated thumbnail, since faces don't have one of their own yet.
        from PIL import Image
        import pillow_heif
        pillow_heif.register_heif_opener()
        try:
            img = Image.open(real_path).convert("RGB")
        except Exception:
            self.people_preview_image.setText("Preview unavailable")
            return
        qimage = QImage(img.tobytes("raw", "RGB"), img.width, img.height, img.width * 3, QImage.Format.Format_RGB888)
        pixmap = QPixmap.fromImage(qimage.copy())
        self.people_preview_image.setPixmap(pixmap.scaled(
            self.people_preview_image.size(), Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation))

    def _on_people_name_box_enter(self):
        """Enter in the name box means different things depending on what's
        selected - label a pending cluster, or rename the already-labeled
        person currently being browsed. Mirrors whichever button is actually
        enabled right now rather than duplicating that logic here."""
        if self.people_current_group and self.people_current_group.get("type") == "person":
            self._rename_people_person()
        else:
            self._confirm_people_label()

    def _rename_people_person(self):
        # people.name is UNIQUE, so renaming to a name already used by someone
        # else merges the two - see face_index.rename_person()'s docstring.
        import face_index
        if not self.people_current_group or self.people_current_group["type"] != "person":
            return
        new_name = self.people_name_box.text().strip()
        if not new_name:
            self.people_status_label.setText("Type a name first.")
            return
        old_name = self.people_current_group["name"]
        person_id = self.people_current_group["person_id"]
        try:
            result_id = face_index.rename_person(person_id, new_name)
        except ValueError as e:
            self.people_status_label.setText(str(e))
            return
        if result_id != person_id:
            self.people_status_label.setText(f"Merged '{old_name}' into existing person '{new_name}'.")
        else:
            self.people_status_label.setText(f"Renamed '{old_name}' to '{new_name}'.")
        self._load_people_lists()

    def _confirm_people_label(self):
        # Acts only on self.people_displayed_faces (what's actually on screen right
        # now), never on the full cluster in the database - a pending group can be
        # bigger than MAX_FACES_TO_DISPLAY, and this must never label faces nobody
        # actually looked at. See face_index.label_faces()'s docstring.
        import face_index
        if not self.people_current_group or self.people_current_group["type"] != "cluster":
            return
        name = self.people_name_box.text().strip()
        if not name:
            self.people_status_label.setText("Type a name first.")
            return
        to_label = [f["face_vector_index"] for f in self.people_displayed_faces
                    if f["face_vector_index"] not in self.people_excluded_faces]
        self._snapshot_for_undo(to_label, f"labeled {len(to_label)} face(s) as {name}")
        try:
            person_id, count = face_index.label_faces(to_label, name)
        except ValueError as e:
            self.people_status_label.setText(str(e))
            self.people_last_action = None  # nothing actually written - no undo to offer
            self.btn_undo.setEnabled(False)
            return
        self.people_status_label.setText(f"Labeled {count} face(s) as {name}.")
        self._load_people_lists()

    def _discard_people_cluster(self):
        # Same displayed-only scoping as _confirm_people_label - see face_index.discard_faces().
        import face_index
        if not self.people_current_group or self.people_current_group["type"] != "cluster":
            return
        to_discard = [f["face_vector_index"] for f in self.people_displayed_faces]
        self._snapshot_for_undo(to_discard, f"discarded {len(to_discard)} face(s)")
        count = face_index.discard_faces(to_discard)
        self.people_status_label.setText(f"Discarded {count} face(s) - not a real person or not worth labeling.")
        self._load_people_lists()

    def _remove_from_person(self):
        # Opposite selection polarity from a cluster: faces start UNCHECKED here, so
        # "checked" (not in people_excluded_faces) means "flagged to remove" instead
        # of "keep" - see _on_people_group_selected's is_person branch.
        import face_index
        if not self.people_current_group or self.people_current_group["type"] != "person":
            return
        to_remove = [f["face_vector_index"] for f in self.people_displayed_faces
                     if f["face_vector_index"] not in self.people_excluded_faces]
        if not to_remove:
            self.people_status_label.setText("Check at least one face to remove first.")
            return
        self._snapshot_for_undo(to_remove, f"removed {len(to_remove)} face(s) from this person")
        count = face_index.unlabel_faces(to_remove)
        self.people_status_label.setText(f"Removed {count} face(s) - back in the unlabeled pool.")
        self._load_people_lists()

    def _snapshot_for_undo(self, face_vector_indices, description):
        """Captures face state right before a write action so _undo_last_people_
        action() can reverse exactly that one action - single-level only, this
        overwrites whatever was snapshotted before, it doesn't stack into a
        history (2026-08-24 request, deliberately scoped to "undo my last click"
        rather than a full log)."""
        import face_index
        self.people_last_action = {
            "description": description,
            "faces": face_index.snapshot_face_states(face_vector_indices),
        }
        self.btn_undo.setEnabled(True)

    def _undo_last_people_action(self):
        import face_index
        if not self.people_last_action:
            return
        face_index.restore_face_states(self.people_last_action["faces"])
        self.people_status_label.setText(f"Undid: {self.people_last_action['description']}.")
        self.people_last_action = None
        self.btn_undo.setEnabled(False)
        self._load_people_lists()

    def add_button(self, layout, text, handler, primary=False):
        """Cuts the repeated create/style/connect/add boilerplate that every
        button in this UI otherwise needed on its own."""
        btn = QPushButton(text)
        if primary:
            btn.setStyleSheet("background-color: #28a745; color: white; font-weight: bold; padding: 15px;")
        else:
            btn.setStyleSheet("background-color: #333; color: white; padding: 5px;")
        btn.clicked.connect(handler)
        layout.addWidget(btn)
        return btn

    def get_manual_items(self):
        """(display_name, path) for every manually-imported item currently in the list."""
        items = []
        for i in range(self.media_list.count()):
            item = self.media_list.item(i)
            data = item.data(Qt.ItemDataRole.UserRole)
            if data and data.get("manual"):
                items.append((item.text(), data["path"]))
        return items

    def remove_selected_from_list(self):
        """Removes the selected row(s) from the media list only - never
        touches the underlying file on disk. If a removed item was a manual
        import, it's dropped from persistence too so it doesn't reappear
        after a refresh/restart; an HDD/Photos-scanned item just reappears
        on the next full scan, which is expected for a list-only removal."""
        selected_items = self.media_list.selectedItems()
        if not selected_items:
            return
        for item in selected_items:
            self.media_list.takeItem(self.media_list.row(item))
        save_manual_imports_to_disk(self.profile["app_data_dir"], self.get_manual_items())

    def load_iphone_photos(self):
        # Preserve manually-imported items already in the list (e.g. across a Refresh click)
        manual_items = self.get_manual_items()

        # On a fresh app launch the list starts empty, so nothing above was found -
        # fall back to what was saved to disk from previous sessions.
        if not manual_items:
            manual_items = load_manual_imports_from_disk(self.profile["app_data_dir"])

        self.media_list.clear()

        # Restore them at the top, before the (potentially huge) Photos/iCloud list below
        for i, (display_name, real_path) in enumerate(manual_items):
            self.add_media_item(display_name, real_path, manual=True, index=i)

        # The shared external drive, if connected, is now the primary media source -
        # the same library the search index scans, so what's browsable here always
        # matches what's searchable (see media-search-shared-drive-architecture memory).
        # Falls back to the original Photos/iCloud pipelines below only when it's not
        # plugged in, rather than requiring it.
        drive_root = find_volume_by_label(EXTERNAL_DRIVE_LABEL)
        if drive_root:
            self.status_label.setText(f"Scanning {EXTERNAL_DRIVE_LABEL}...")
            QApplication.processEvents()
            hdd_videos = list_hdd_media(drive_root)
            for name, path in hdd_videos:
                self.add_media_item(name, path)
            self.status_label.setText(f"Loaded {len(hdd_videos)} videos from {EXTERNAL_DRIVE_LABEL}.")
            return

        mode = self.profile["pipeline_mode"]

        # --- PIPELINE 1: NATIVE MAC PHOTOS DATABASE BRIDGE ---
        if mode == "native_db":
            try:
                import osxphotos
                self.status_label.setText("Connecting directly to Mac Photos Database...")
                QApplication.processEvents()
                
                # Scan the internal SQLite database mappings
                photosdb = osxphotos.PhotosDB()
                downloaded_count = 0
                cloud_only_count = 0

                # Split into downloaded vs cloud-only first, so downloaded videos always list first;
                # newest-first ordering is kept within each group.
                local_videos = []
                cloud_videos = []
                for video in photosdb.photos(images=False, movies=True):
                    path = video.path or video.path_edited
                    if path and Path(path).exists():
                        local_videos.append((video, path))
                    else:
                        cloud_videos.append(video)

                sort_key = lambda v: v.date_added or v.date
                local_videos.sort(key=lambda pair: sort_key(pair[0]), reverse=True)
                cloud_videos.sort(key=sort_key, reverse=True)

                for video, path in local_videos:
                    self.add_media_item(video.original_filename or Path(path).name, path)
                    downloaded_count += 1

                for video in cloud_videos:
                    # Cloud-only original (not downloaded to this Mac yet).
                    # Tag it with the PhotoInfo so we can fetch it from iCloud on demand when selected.
                    label = f"☁️ {video.original_filename or video.uuid}"
                    item = QListWidgetItem(label)
                    item.setData(Qt.ItemDataRole.UserRole, {"photo": video})
                    self.media_list.addItem(item)
                    cloud_only_count += 1

                if downloaded_count or cloud_only_count:
                    self.status_label.setText(
                        f"Loaded {downloaded_count} downloaded + {cloud_only_count} cloud-only iPhone videos. "
                        f"Selecting a ☁️ item downloads it from iCloud first."
                    )

                if self.media_list.count() == 0:
                    # FALLBACK: If osxphotos returns nothing, scan the standard Pictures folder recursively
                    import platform
                    if platform.system() == "Darwin":
                        home = Path.home()
                        fallback_folder = home / "Pictures" / "Photos Library.photoslibrary" / "originals"
                        if fallback_folder.exists():
                            for file_path in fallback_folder.rglob("*"):
                                if file_path.suffix.lower() in ['.mp4', '.mov', '.m4v']:
                                    if not file_path.name.startswith('.'):
                                        self.add_media_item(file_path.name, file_path)
                                        
                if self.media_list.count() == 0:
                    self.media_list.addItem("Database synced! No local video clips found.")
                elif not (downloaded_count or cloud_only_count):
                    self.status_label.setText(f"Loaded {self.media_list.count()} iPhone videos natively!")
            except Exception as e:
                self.media_list.addItem("Could not read the Photos database.")
                self.status_label.setText(f"Photos database error: {e}")
                
        # --- PIPELINE 2: CLOUD ICLOUD FETCH FOR WINDOWS PC ---
        elif mode == "cloud":
            folder = self.profile["photos_path"]
            if folder and folder.exists():
                video_files = list(folder.glob("*.mp4")) + list(folder.glob("*.mov"))
                video_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)  # Newest added first
                for video_file in video_files:
                    self.add_media_item(video_file.name, video_file)
            
            if self.media_list.count() == 0:
                self.media_list.addItem("iCloud folder empty. Drag files here manually.")

    def add_media_item(self, display_name, real_path, manual=False, index=None):
        """Show a friendly filename in the list while keeping the real (often ugly,
        UUID-named) on-disk path tucked away for actual playback/editing use.
        Pass index=0 (etc.) to insert at the top instead of appending at the end -
        important for manual imports, since the Photos list can have thousands of
        entries and an appended item would be scrolled off-screen at the bottom."""
        item = QListWidgetItem(display_name)
        item.setData(Qt.ItemDataRole.UserRole, {"path": str(real_path), "manual": manual})
        if index is None:
            self.media_list.addItem(item)
        else:
            self.media_list.insertItem(index, item)
        return item

    def filter_media_list(self, text):
        text = text.lower()
        for i in range(self.media_list.count()):
            item = self.media_list.item(i)
            item.setHidden(text not in item.text().lower())

    def resolve_local_path(self, item):
        """Return a real on-disk path for this media item. For a cloud-only Photos
        original (Mac native_db mode), re-checks Photos fresh in case it has since
        been downloaded, and if so, copies it (with its real filename) into the
        shared Import Folder - a pure local file copy, no Photos automation involved."""
        data = item.data(Qt.ItemDataRole.UserRole)
        if not data:
            return item.text()  # No tag at all - the displayed text is already the real path
        if data.get("path"):
            return data["path"]  # Friendly display name shown, real path stashed here (e.g. downloaded Mac videos)
        if not data.get("photo"):
            return item.text()

        photo = data["photo"]

        # Re-check fresh - the snapshot we loaded the list from may be out of date
        import osxphotos
        fresh = osxphotos.PhotosDB().get_photo(photo.uuid)
        fresh_path = fresh.path or fresh.path_edited if fresh else None

        if fresh_path and Path(fresh_path).exists():
            IMPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            try:
                exported = fresh.export(str(IMPORT_CACHE_DIR))
                local_path = exported[0] if exported else fresh_path
            except Exception:
                local_path = fresh_path
            item.setText(fresh.original_filename or Path(local_path).name)
            item.setData(Qt.ItemDataRole.UserRole, {"path": local_path})
            self.status_label.setText(f"'{fresh.original_filename}' is downloaded - copied into the Import Folder.")
            return local_path

        QMessageBox.information(
            self,
            "Video Not Downloaded Yet",
            f"'{photo.original_filename}' is stored in iCloud only and hasn't been downloaded to this Mac.\n\n"
            f"1. Open Photos\n"
            f"2. Select '{photo.original_filename}'\n"
            f"3. Menu bar → File → Export → Export Unmodified Original(s)... (any destination is fine)\n"
            f"4. Come back here and select '{photo.original_filename}' in this list again\n"
            f"   (the app will automatically grab the correctly-named copy for you)"
        )
        self.status_label.setText(f"'{photo.original_filename}' needs downloading in Photos first.")
        return None

    def play_selected_video(self, item):
        local_path = self.resolve_local_path(item)
        if not local_path:
            # Without this, video_duration_cached stays stale from whatever video played
            # last, so nudging the margin/sensitivity sliders afterward would silently
            # re-trigger resolve_local_path on this failed item (repeating the "not
            # downloaded" popup) instead of correctly doing nothing.
            self.video_duration_cached = 0
            return
        self.player.setSource(QUrl.fromLocalFile(local_path))
        self.player.play()

    def update_timeline_playhead(self, position_ms):
        self.timeline_tracker.set_progress(position_ms)

    def update_video_duration(self, duration_ms):
        self.video_duration_cached = duration_ms
        if duration_ms > 0:
            # Auto-load the real cut preview as soon as the newly selected video is ready
            self.generate_timeline_preview()

    def generate_timeline_preview(self):
        selected_items = self.media_list.selectedItems()
        if not selected_items or self.video_duration_cached == 0:
            return

        input_file = self.resolve_local_path(selected_items[0])
        if not input_file:
            return
        self.status_label.setText("Calculating vocal coordinates metrics sheets...")
        QApplication.processEvents()

        try:
            margin_seconds = self.margin_box.value()
            sensitivity = self.sensitivity_box.value() / 100.0  # UI shows %, auto-editor/levels use a 0-1 fraction

            # Real per-frame audio loudness from auto-editor - no more fake/random preview data
            levels_cmd = [sys.executable, "-m", "auto_editor", "levels", input_file]
            result = subprocess.run(levels_cmd, check=True, capture_output=True, text=True)
            levels = [float(line) for line in result.stdout.splitlines() if line.strip() and line.strip() != "@start"]

            if not levels:
                raise RuntimeError("No audio level data returned for this file.")

            is_loud = [lvl > sensitivity for lvl in levels]
            frame_count = len(is_loud)
            frame_duration_ms = self.video_duration_cached / frame_count
            margin_frames = max(0, int((margin_seconds * 1000) / frame_duration_ms))

            # Dilate loud sections by the margin on each side - matches auto-editor's --margin behavior
            expanded = is_loud[:]
            for idx, loud in enumerate(is_loud):
                if loud:
                    lo = max(0, idx - margin_frames)
                    hi = min(frame_count, idx + margin_frames + 1)
                    expanded[lo:hi] = [True] * (hi - lo)

            # Collapse the per-frame flags into contiguous (start_pct, end_pct, is_speech) chunks
            chunks_data = []
            seg_start = 0
            for idx in range(1, frame_count + 1):
                if idx == frame_count or expanded[idx] != expanded[seg_start]:
                    chunks_data.append((seg_start / frame_count, idx / frame_count, expanded[seg_start]))
                    seg_start = idx

            self.timeline_tracker.set_timeline_data(chunks_data, self.video_duration_cached)
            self.status_label.setText("Ready.")

        except Exception as e:
            QMessageBox.warning(self, "Error", str(e))

    def probe_video(self, path):
        """One ffmpeg probe covering codec + duration, instead of two separate
        subprocess calls that each re-read the same file's stream info - halves
        the ffmpeg-spawning overhead per file probed."""
        result = subprocess.run([self.profile["ffmpeg_binary"], "-i", path], capture_output=True, text=True)
        codec_match = re.search(r"Video:\s*(\w+)", result.stderr)
        duration_match = re.search(r"Duration:\s*(\d+):(\d+):(\d+\.\d+)", result.stderr)
        codec = codec_match.group(1).lower() if codec_match else None
        duration = None
        if duration_match:
            h, m, s = duration_match.groups()
            duration = int(h) * 3600 + int(m) * 60 + float(s)
        return codec, duration

    @staticmethod
    def format_duration(seconds):
        if seconds is None:
            return "unknown"
        m, s = divmod(int(round(seconds)), 60)
        h, m = divmod(m, 60)
        return f"{h:02d}:{m:02d}:{s:02d}" if h else f"{m:02d}:{s:02d}"

    def transcribe_selected_video(self):
        selected_items = self.media_list.selectedItems()
        if not selected_items:
            QMessageBox.warning(self, "No Video Selected", "Please select a video from the list first!")
            return

        input_file = self.resolve_local_path(selected_items[0])
        if not input_file:
            return
        # Use the friendly display name shown in the list, not the underlying file's own
        # name - for videos already on this Mac, the real file is UUID-named internally
        # by Photos, which would otherwise leak into the .srt filename.
        display_name = selected_items[0].text()
        output_dir = EXPORT_CACHE_DIR

        if self.transcribe_location_checkbox.isChecked():
            suggested = str(get_unique_path(EXPORT_CACHE_DIR / (Path(display_name).stem + ".srt")))
            chosen_path, _ = QFileDialog.getSaveFileName(self, "Save Transcript", suggested, "SubRip Subtitle (*.srt)")
            if not chosen_path:
                return
            output_dir = Path(chosen_path).parent
            display_name = Path(chosen_path).name

        self.transcribe_file(input_file, display_name, output_dir=output_dir)

    def unload_whisper_model(self):
        """Drops the loaded Whisper model so its ~1.4GB can be reclaimed -
        CPU-only, so releasing the Python reference is enough, no explicit
        cache-clearing needed the way CLIP's GPU/MPS memory requires."""
        self.whisper_model = None

    def transcribe_file(self, input_file, display_name, output_dir=None):
        """Transcribe a specific file and save its .srt - shared by the manual
        'Transcribe Audio' button and the optional post-render auto-transcribe.
        output_dir defaults to the shared Export Folder; the post-render path
        passes the rendered video's own folder instead, so the captions land
        next to whatever output location the user actually chose."""
        output_dir = output_dir or EXPORT_CACHE_DIR
        try:
            from faster_whisper import WhisperModel
        except ImportError:
            QMessageBox.critical(
                self, "Missing Dependency",
                "faster-whisper isn't installed.\n\nRun: pip install faster-whisper"
            )
            return False

        language = TRANSCRIPTION_LANGUAGES.get(self.transcribe_lang_box.currentText())
        multilingual = (language == MULTILINGUAL_SENTINEL)
        if multilingual:
            language = None
        self.transcript_view.clear()

        try:
            if self.whisper_model is None:
                self.status_label.setText("🎙️ Loading transcription model (downloads once, ~1.4GB the first time)...")
                QApplication.processEvents()
                self.whisper_model = WhisperModel("medium", device="cpu", compute_type="int8")

            self.status_label.setText(f"🎙️ Transcribing '{display_name}'...")
            QApplication.processEvents()

            task = "translate" if self.translate_checkbox.isChecked() else "transcribe"

            start = time.time()
            segments_gen, info = self.whisper_model.transcribe(
                input_file, word_timestamps=True, language=language, multilingual=multilingual, task=task
            )

            segments = []
            text_parts = []
            for seg in segments_gen:
                segments.append(seg)
                text_parts.append(seg.text.strip())
                elapsed = int(time.time() - start)
                self.status_label.setText(f"🎙️ Transcribing... {elapsed}s elapsed ({seg.end:.0f}s of audio processed)")
                self.transcript_view.setPlainText(" ".join(text_parts))
                QApplication.processEvents()

            # Model's actual work is done as of here (generator fully consumed) -
            # restart the idle clock now, not when transcription began, so a long
            # transcribe never races its own unload timer mid-run.
            self.whisper_unload_timer.start(WHISPER_IDLE_UNLOAD_MS)

            if not segments:
                self.status_label.setText("No speech detected in this video.")
                QMessageBox.information(self, "No Speech Detected", "No speech was found in this video's audio.")
                return False

            # Export as .srt captions for use in a finishing-touches editor like CapCut
            output_dir.mkdir(parents=True, exist_ok=True)
            srt_path = get_unique_path(output_dir / (Path(display_name).stem + ".srt"))
            with open(srt_path, "w", encoding="utf-8") as f:
                for i, seg in enumerate(segments, start=1):
                    f.write(f"{i}\n")
                    f.write(f"{format_srt_timestamp(seg.start)} --> {format_srt_timestamp(seg.end)}\n")
                    f.write(f"{seg.text.strip()}\n\n")

            self.status_label.setText(f"🎉 Transcription complete! Captions saved to {srt_path.name}")
            return True

        except Exception as e:
            self.status_label.setText("❌ Transcription failed.")
            QMessageBox.critical(self, "Error", str(e))
            return False

    def start_magic_cut(self):
        selected_items = self.media_list.selectedItems()
        if not selected_items:
            return

        self.player.stop()

        # Grab the local path, downloading from iCloud first if needed
        first_clicked_item = selected_items[0]
        input_file = self.resolve_local_path(first_clicked_item)
        if not input_file:
            return
        display_name = first_clicked_item.text()

        if self.render_location_checkbox.isChecked():
            suggested = str(get_unique_path(EXPORT_CACHE_DIR / (Path(display_name).stem + ".mp4")))
            output_video, _ = QFileDialog.getSaveFileName(self, "Save Video", suggested, "Video Files (*.mp4)")
        else:
            EXPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            output_video = str(get_unique_path(EXPORT_CACHE_DIR / (Path(display_name).stem + ".mp4")))

        if output_video:
            margin_str = f"{self.margin_box.value()}s"
            transcoded_temp = None
            stage_times = []

            try:
                # One probe covers both the codec check below and the original-length figure
                # in the final summary - previously two separate ffmpeg calls on this same file.
                input_codec, original_duration = self.probe_video(input_file)

                # auto-editor has a real bug decoding HEVC (H.265) source video whenever it
                # actually has to cut and splice multiple segments together - it silently
                # produces a fully black video track (audio is unaffected). Confirmed by
                # testing directly: forcing the *output* codec doesn't help, but pre-converting
                # the HEVC *source* to H.264 first fixes it completely. Many iPhone videos are
                # HEVC, so auto-detect and work around it transparently.
                actual_input = input_file
                if not self.skip_hevc_check_checkbox.isChecked() and input_codec == "hevc":
                    transcoded_temp = str(Path(output_video).with_name("temp_h264_source.mp4"))
                    transcode_cmd = [self.profile["ffmpeg_binary"], "-y"]
                    if self.profile["os"] == "Darwin":
                        # Hardware encode (videotoolbox) confirmed ~4.7x faster than software libx264
                        # for the same conversion, with identical correctness (verified: rotation
                        # still comes out correctly baked into the output pixels either way).
                        transcode_cmd += ["-hwaccel", "videotoolbox"]
                        transcode_cmd += ["-i", input_file, "-c:v", "h264_videotoolbox", "-b:v", "12M"]
                    else:
                        transcode_cmd += ["-i", input_file, "-c:v", "libx264", "-preset", "fast", "-b:v", "12M"]
                    transcode_cmd += ["-c:a", "copy", transcoded_temp]
                    elapsed = self.run_with_live_progress(transcode_cmd, "🎞️ Converting HEVC video for reliable editing")
                    stage_times.append(("Converting video", elapsed))
                    actual_input = transcoded_temp

                # Cut silence, keeping the original shot orientation untouched (no rotation).
                # Some iPhone videos decode far slower than others on this Mac - a long render
                # is expected for those, not a hang, so this stays non-blocking and shows live
                # elapsed time instead of freezing the window.
                sensitivity_pct = self.sensitivity_box.value()
                cut_instruction = [
                    sys.executable, "-m", "auto_editor",
                    actual_input, "--output", output_video,
                    "--margin", margin_str,
                    "--edit", f"audio:threshold={sensitivity_pct}%"
                ]
                elapsed = self.run_with_live_progress(cut_instruction, "⚙️ Cutting Video Assets")
                stage_times.append(("Processing silences", elapsed))

                if self.transcribe_after_render_checkbox.isChecked():
                    t0 = time.time()
                    self.transcribe_file(output_video, Path(output_video).name, output_dir=Path(output_video).parent)
                    stage_times.append(("Transcribing", time.time() - t0))

                total = sum(t for _, t in stage_times)
                breakdown = "\n".join(f"  {name}: {t:.1f}s" for name, t in stage_times)

                _, rendered_duration = self.probe_video(output_video)
                if original_duration is not None and rendered_duration is not None:
                    cut_amount = original_duration - rendered_duration
                    length_summary = (
                        f"Original length: {self.format_duration(original_duration)}\n"
                        f"Rendered length: {self.format_duration(rendered_duration)}\n"
                        f"Time cut: {self.format_duration(cut_amount)}"
                    )
                else:
                    length_summary = "Length comparison unavailable."

                self.status_label.setText(f"🎉 Rendered successfully! Total: {total:.1f}s\n{breakdown}\n{length_summary}")
                QMessageBox.information(
                    self, "Success!",
                    f"Video compiled completely!\n\n{length_summary}\n\nTime breakdown:\n{breakdown}\n\nTotal: {total:.1f}s"
                )

            except Exception as e:
                self.status_label.setText("❌ Render pipeline failed.")
                QMessageBox.critical(self, "Error", str(e))
            finally:
                if transcoded_temp and Path(transcoded_temp).exists():
                    Path(transcoded_temp).unlink()

    def run_with_live_progress(self, cmd, label):
        """Run a subprocess without freezing the UI, showing elapsed time so a
        slow render (e.g. HEVC video) doesn't look like a crash. Force-quitting
        mid-render is what produces a black-video/audio-only broken output file,
        so this exists specifically to stop that from happening."""
        process = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        start = time.time()
        while process.poll() is None:
            elapsed = int(time.time() - start)
            self.status_label.setText(
                f"{label}... {elapsed}s elapsed - HEVC/large videos can take several minutes, please don't force-quit"
            )
            QApplication.processEvents()
            time.sleep(0.3)
        if process.returncode != 0:
            raise subprocess.CalledProcessError(process.returncode, cmd)
        return time.time() - start

    def manually_import_video(self):
        # Open up a clean dialogue explorer, starting in the shared import folder if it exists
        start_dir = str(IMPORT_CACHE_DIR) if IMPORT_CACHE_DIR.exists() else ""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Videos to Edit",
            start_dir,
            "Video Files (*.mp4 *.mov *.m4v);;All Files (*)",
            options=QFileDialog.Option.DontUseNativeDialog
        )

        if file_paths:
            for i, path in enumerate(file_paths):
                # Insert at the top (not appended) - the Photos list can have
                # thousands of entries, so an appended item would be invisible
                # without scrolling all the way to the bottom.
                self.add_media_item(Path(path).name, path, manual=True, index=i)

            # Persist all current manual imports to disk so they survive an app restart
            current_manual = self.get_manual_items()
            save_manual_imports_to_disk(self.profile["app_data_dir"], current_manual)

            self.status_label.setText(f"Added {len(file_paths)} video tracks manually!")

    def open_import_folder(self):
        IMPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(IMPORT_CACHE_DIR)))

    def open_export_folder(self):
        EXPORT_CACHE_DIR.mkdir(parents=True, exist_ok=True)
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(EXPORT_CACHE_DIR)))


# --- MAIN APP ENTRY BLOCK ---
if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle("Fusion")
    window = ProfessionalAIEditor()
    window.show()
    sys.exit(app.exec())

