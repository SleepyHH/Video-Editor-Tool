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
                             QTabWidget, QScrollArea, QDateEdit, QLineEdit)
from PyQt6.QtCore import QDate, QThread, QTimer, QUrl, Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QFont, QPen, QDesktopServices, QPixmap
from PyQt6.QtMultimedia import QMediaPlayer, QAudioOutput
from PyQt6.QtMultimediaWidgets import QVideoWidget
from config import EXTERNAL_DRIVE_LABEL, VIDEO_EXTENSIONS, find_volume_by_label, get_os_profile, walk_media_files

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

    def __init__(self, query, top_k, after, before, file_types):
        super().__init__()
        self.query = query
        self.top_k = top_k
        self.after = after
        self.before = before
        self.file_types = file_types

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
            results = media_search.search(
                self.query, top_k=self.top_k, after=self.after,
                before=self.before, file_types=self.file_types,
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
        self.setCentralWidget(self.tabs)
        # Pausing both unconditionally on every switch is simpler than tracking
        # which tab was just left, and harmless - pausing a player that isn't
        # currently playing is a no-op.
        self.tabs.currentChanged.connect(lambda _: (self.player.pause(), self.search_preview_player.pause()))

        self.load_iphone_photos()

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

        self.btn_search.setEnabled(False)
        self.search_status_label.setText(f'🔍 Searching for "{query}"...')

        self.search_worker = SearchWorker(query, self.search_top_box.value(), after, before, file_types)
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

        self.search_status_label.setText(f"{len(results)} results")
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
        ranked = sorted(self.selected_search_results.values(), key=lambda r: r["score"], reverse=True)

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

