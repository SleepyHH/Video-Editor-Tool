"""
Mode A engine, lightweight local GUI: a search box that stays open in the
browser, backed by the same search() engine as the CLI. No new dependency -
just Python's built-in http.server, no Flask/Django/etc.

    python3 media_search_server.py
    -> opens http://localhost:8765 automatically, Ctrl+C in the terminal to stop

Keeps the CLIP model loaded once for the whole session (load_clip_model()'s
module-level cache in media_index.py), so repeated queries are fast - the
one-shot CLI re-pays some model setup cost on every single invocation,
this doesn't.

Serves media/thumbnail bytes itself over HTTP (rather than pointing the
browser at file:// paths, like the CLI gallery does) - required, not a
style choice: a page loaded over http:// is blocked by the browser from
loading file:// resources at all, a security boundary, confirmed live as
the actual cause of thumbnails/video staying black through this server.
"""
import html
import threading
import webbrowser
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path
from urllib.parse import parse_qs, unquote, urlparse

from config import to_portable_path
from media_index import EXTERNAL_DRIVE_LABEL, init_db, unload_clip_model
from media_search import GALLERY_STYLE, build_gallery_cards, get_thumbnail_dir, smart_search

PORT = 8765

# Auto-unload CLIP after this long with no completed search, so the model's
# ~GBs don't sit resident in memory for a GUI left open and idle. Timer is
# (re)started only once a search actually finishes - not when one begins -
# so a slow search can't get its own model unloaded out from under it.
CLIP_IDLE_UNLOAD_SECONDS = 10 * 60

_unload_timer = None
_unload_timer_lock = threading.Lock()


def _schedule_clip_unload():
    global _unload_timer
    with _unload_timer_lock:
        if _unload_timer is not None:
            _unload_timer.cancel()
        _unload_timer = threading.Timer(CLIP_IDLE_UNLOAD_SECONDS, unload_clip_model)
        _unload_timer.daemon = True
        _unload_timer.start()

CONTENT_TYPES = {
    ".mp4": "video/mp4", ".m4v": "video/mp4", ".mov": "video/quicktime",
    ".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".png": "image/png", ".heic": "image/heic",
}

SEARCH_FORM = """
<form class="searchform" action="/search" method="get">
  <input type="text" name="q" placeholder="Search your library..." value="{query}" autofocus>
  <select name="type" title="Filter by file type">
    <option value="" {sel_all}>All types</option>
    <option value="video" {sel_video}>Video only</option>
    <option value="image" {sel_image}>Image only</option>
  </select>
  <select name="person" title="Filter by labeled person">
    {person_options}
  </select>
  <input type="number" name="top" value="{top}" min="1" max="200" title="How many results">
  <input type="date" name="after" value="{after}" title="Only after this date">
  <input type="date" name="before" value="{before}" title="Only before this date">
  <button type="submit">Search</button>
</form>
"""


def _person_options_html(selected):
    """<option> tags for the Person filter, "All people" first, populated
    from face_index's labeled people - local import, same reasoning as
    media_search.search()'s own person filter (keeps face_index's cv2/hdbscan
    out of a request that doesn't use this filter's underlying data, even
    though this particular server already pays torch's import cost at
    startup unconditionally, unlike main.py's lazier Search tab)."""
    import face_index
    parts = [f'<option value="" {"selected" if not selected else ""}>All people</option>']
    for p in face_index.list_people():
        name = p["name"]
        sel = "selected" if name == selected else ""
        parts.append(f'<option value="{html.escape(name)}" {sel}>{html.escape(name)}</option>')
    return "".join(parts)


def _is_indexed_file(real_file_path):
    """Only ever serve a path that's actually in our own index - closes off
    /media?path=... as an arbitrary-file-read endpoint. Cheap check, real
    boundary, appropriate for a single-user localhost-only tool without
    needing full auth machinery.

    real_file_path is a real, resolved absolute path (what search() hands
    back) - converted to the same portable form used in storage before
    checking, since that's what's actually in the database for anything
    on the shared drive."""
    storage_path = to_portable_path(real_file_path, EXTERNAL_DRIVE_LABEL)
    conn = init_db()  # not a raw sqlite3.connect() - needs the same EXCLUSIVE-locking pragma
    row = conn.execute("SELECT 1 FROM indexed_files WHERE file_path = ?", (storage_path,)).fetchone()
    conn.close()
    return row is not None


class SearchHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)

        if parsed.path.startswith("/thumb/"):
            self._serve_thumbnail(parsed.path[len("/thumb/"):])
        elif parsed.path == "/media":
            self._serve_media(parse_qs(parsed.query).get("path", [""])[0])
        else:
            self._serve_search_page(parsed, parse_qs(parsed.query))

    def _serve_search_page(self, parsed, params):
        query = (params.get("q", [""])[0]).strip()
        top = params.get("top", ["12"])[0] or "12"
        after = params.get("after", [""])[0] or None
        before = params.get("before", [""])[0] or None
        file_type = params.get("type", [""])[0] or None
        person = params.get("person", [""])[0] or None

        form = SEARCH_FORM.format(
            query=html.escape(query), top=html.escape(top),
            after=after or "", before=before or "",
            sel_all="selected" if not file_type else "",
            sel_video="selected" if file_type == "video" else "",
            sel_image="selected" if file_type == "image" else "",
            person_options=_person_options_html(person),
        )

        if parsed.path == "/search" and query:
            results = smart_search(query, top_k=int(top), after=after, before=before, file_types=file_type, explicit_person=person)
            _schedule_clip_unload()
            status_line = f"<p>{len(results)} results for &quot;{html.escape(query)}&quot;</p>" if results else \
                "<p>No results - try a different query, or check the index has been built.</p>"
            body = form + status_line + f'<div class="grid">{build_gallery_cards(results, for_server=True)}</div>'
        else:
            body = form + "<p>Type a query above to search your library.</p>"

        page = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Huys Media Search</title>
<style>{GALLERY_STYLE}</style></head>
<body><h1>Huys Media Search - local, nothing leaves this machine</h1>{body}</body></html>"""

        self._send_bytes(page.encode("utf-8"), "text/html; charset=utf-8")

    def _serve_thumbnail(self, filename):
        thumbnail_dir = get_thumbnail_dir()
        path = thumbnail_dir / filename
        if path.parent != thumbnail_dir or not path.exists():
            self.send_response(404)
            self.end_headers()
            return
        self._send_bytes(path.read_bytes(), "image/jpeg")

    def _serve_media(self, file_path_str):
        file_path_str = unquote(file_path_str)
        path = Path(file_path_str)
        if not _is_indexed_file(file_path_str) or not path.exists():
            self.send_response(404)
            self.end_headers()
            return

        content_type = CONTENT_TYPES.get(path.suffix.lower(), "application/octet-stream")
        file_size = path.stat().st_size
        range_header = self.headers.get("Range")

        # Video needs Range request support to seek/scrub properly - without it,
        # browsers either can't jump to the matched timestamp or refuse to play
        # at all until the whole file downloads (a real problem for 100-200MB clips).
        if range_header:
            start_str, _, end_str = range_header.replace("bytes=", "").partition("-")
            start = int(start_str) if start_str else 0
            end = min(int(end_str), file_size - 1) if end_str else file_size - 1
            self.send_response(206)
            self.send_header("Content-Type", content_type)
            self.send_header("Content-Range", f"bytes {start}-{end}/{file_size}")
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(end - start + 1))
            self.end_headers()
            with open(path, "rb") as f:
                f.seek(start)
                self.wfile.write(f.read(end - start + 1))
        else:
            self.send_response(200)
            self.send_header("Content-Type", content_type)
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Content-Length", str(file_size))
            self.end_headers()
            with open(path, "rb") as f:
                self.wfile.write(f.read())

    def _send_bytes(self, data, content_type):
        self.send_response(200)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def log_message(self, format, *args):
        pass  # quiet - browser noise (favicon requests etc.) would otherwise clutter the terminal


if __name__ == "__main__":
    server = HTTPServer(("localhost", PORT), SearchHandler)
    url = f"http://localhost:{PORT}"
    print(f"Search GUI running at {url} - Ctrl+C to stop")
    webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nStopped.")
