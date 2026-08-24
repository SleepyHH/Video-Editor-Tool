"""
Mode A engine, query half: turns a short text query into a CLIP text
embedding and ranks it against the index built by media_index.py.
No GUI - run directly to playtest:

    python3 media_search.py "dog jumping in the lake"                # prints ranked file paths
    python3 media_search.py "dog jumping in the lake" --gallery      # visual HTML gallery (recommended)
    python3 media_search.py "dog jumping in the lake" --open         # opens every result individually
    python3 media_search.py "dog jumping in the lake" --top 30       # more/fewer results
"""
import argparse
import hashlib
import html
import subprocess
import tempfile
import webbrowser
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from config import get_os_profile, resolve_portable_path
from media_index import (
    EXTERNAL_DRIVE_LABEL, IMAGE_EXTENSIONS, VIDEO_EXTENSIONS,
    get_db_path, get_embeddings_path, get_index_dir, init_db, load_clip_model,
)


def get_thumbnail_dir():
    return get_index_dir() / "thumbnails"


def embed_text(query, model, tokenizer, device):
    tokens = tokenizer([query]).to(device)
    with torch.no_grad():
        features = model.encode_text(tokens)
        features = features / features.norm(dim=-1, keepdim=True)
    return features.cpu().numpy().astype("float32")[0]


def resolve_file_types(file_types):
    """Normalizes a file-type filter into a set of lowercase extensions
    (with leading dot), or None for no filter. Accepts a shorthand
    ("video"/"image", expanding to the same extension sets media_index.py
    already indexes by) or specific extensions (e.g. "mov", ".mp4"), as a
    comma-separated string or a list - either works from the CLI or the
    GUI's query string."""
    if not file_types:
        return None
    if isinstance(file_types, str):
        file_types = file_types.split(",")
    resolved = set()
    for t in file_types:
        t = t.strip().lower()
        if not t:
            continue
        if t in ("video", "videos"):
            resolved |= VIDEO_EXTENSIONS
        elif t in ("image", "images", "photo", "photos"):
            resolved |= IMAGE_EXTENSIONS
        else:
            resolved.add(t if t.startswith(".") else f".{t}")
    return resolved or None


def search(query, top_k=12, after=None, before=None, file_types=None, person=None):
    """Returns up to top_k results, ranked by cosine similarity, one per
    source file (multiple matching frames from the same video collapse
    into a single result at its best-scoring timestamp).

    after/before (optional "YYYY-MM-DD" strings) filter by EXIF/metadata
    capture date - a file with no known date is excluded whenever a date
    filter is active, since there's no way to verify it's actually in
    range. This is a manual, explicit filter for now - not yet automatic
    from something like "last summer" in the query text itself, which
    would need its own query-understanding step on top of this.

    file_types (optional, see resolve_file_types) filters by extension,
    e.g. "video" for just .mov/.mp4/.m4v, or "mov,mp4" for specific ones -
    added 05-08-2026 since the user is mostly hunting for footage to edit,
    not photos.

    person (optional, an exact face_index people.name string) restricts
    results to files with at least one face labeled as that person - added
    2026-08-24, the first real join between this CLIP-based search index and
    the separate face_index.py identity system. An unknown name (or one with
    no labeled faces) short-circuits to an empty result before the CLIP
    model even loads, same as the missing-embeddings-file check below.

    Results on the shared external drive are stored path-portably (see
    config.to_portable_path) and resolved back to a real path here, on
    whichever machine happens to be running - a result whose drive isn't
    currently connected is skipped rather than returned as a broken path,
    same treatment as a cloud-only file that hasn't been downloaded yet."""
    embeddings_path = get_embeddings_path()
    if not embeddings_path.exists():
        return []

    file_types = resolve_file_types(file_types)

    allowed_paths = None
    if person:
        # Local import - keeps face_index's own dependencies (cv2, hdbscan) out of
        # every plain-text search that doesn't use this filter, same lazy-import
        # discipline already used for faster_whisper/insightface elsewhere.
        import face_index
        allowed_paths = face_index.get_file_paths_for_person(person)
        if not allowed_paths:
            return []  # unknown name, or genuinely no labeled faces - nothing to rank

    model, _, tokenizer, device = load_clip_model()
    query_vec = embed_text(query, model, tokenizer, device)

    embeddings = np.load(embeddings_path)
    scores = embeddings @ query_vec  # cosine similarity - both sides are L2-normalized

    conn = init_db()
    rows = conn.execute(
        "SELECT items.vector_index, items.file_path, items.media_type, items.timestamp_seconds, "
        "indexed_files.date_taken, indexed_files.lat, indexed_files.lon "
        "FROM items JOIN indexed_files ON items.file_path = indexed_files.file_path"
    ).fetchall()
    conn.close()

    best_per_file = {}
    for vector_index, stored_path, media_type, timestamp_seconds, date_taken, lat, lon in rows:
        if file_types and Path(stored_path).suffix.lower() not in file_types:
            continue
        if after and (not date_taken or date_taken < after):
            continue
        if before and (not date_taken or date_taken > before):
            continue
        if allowed_paths is not None and stored_path not in allowed_paths:
            continue

        real_path = resolve_portable_path(stored_path, EXTERNAL_DRIVE_LABEL)
        if not real_path or not real_path.exists():
            continue  # drive not connected right now, or file genuinely missing

        score = float(scores[vector_index])
        current = best_per_file.get(stored_path)
        if current is None or score > current["score"]:
            best_per_file[stored_path] = {
                "file_path": str(real_path),
                "media_type": media_type,
                "timestamp_seconds": timestamp_seconds,
                "date_taken": date_taken,
                "lat": lat,
                "lon": lon,
                "score": score,
            }

    ranked = sorted(best_per_file.values(), key=lambda r: r["score"], reverse=True)
    return ranked[:top_k]


def smart_search(query, top_k=12, after=None, before=None, file_types=None, explicit_person=None):
    """search(), but with any labeled people's names typed directly into the
    query text detected and pulled out first - "huy and cats" means "photos
    of Huy, ranked by how well they match 'cats'", not a literal CLIP search
    for the string "huy and cats" (2026-08-24 request - see
    face_index.extract_mentioned_people() for the detection rules and their
    real-data-driven caveats, e.g. why "an"/"Dad" behave differently).

    explicit_person (the Search tab's Person dropdown) is merged in with
    whatever's detected in the text - lets a name that doesn't parse
    reliably from free text (excluded ones - see
    face_index._QUERY_AMBIGUOUS_NAMES) still combine with a typed query.

    - Nobody mentioned (dropdown on "All people", no name detected) ->
      behaves exactly like search().
    - One person mentioned -> the same hard filter search(person=...)
      already does: only their files, ranked by whatever text is left.
    - Multiple people mentioned -> a file needs only ONE of them, not all
      (photos of everyone together are a much narrower, rarer category than
      any one of them alone) - but files matching MORE of the mentioned
      people are ranked ahead of files matching fewer, CLIP relevance to
      the leftover text as the tiebreaker within each tier. Every result
      carries "matched_people" (who) and "match_count" (how many).
    - If nothing but names is left after extraction (e.g. query was just
      "huy"), there's nothing left to rank by - the CLIP model isn't even
      loaded; results are that person's/those people's files, newest first,
      with "score": None (callers displaying a relevance bar should treat
      that as "not applicable", not zero)."""
    import face_index
    remainder, detected = face_index.extract_mentioned_people(query)
    mentioned = list(dict.fromkeys(detected + ([explicit_person] if explicit_person else [])))

    if not mentioned:
        return search(query, top_k=top_k, after=after, before=before, file_types=file_types)

    paths_by_person = {name: face_index.get_file_paths_for_person(name) for name in mentioned}
    relevant_paths = set().union(*paths_by_person.values())
    if not relevant_paths:
        return []

    def _match_count(portable_path):
        return sum(1 for paths in paths_by_person.values() if portable_path in paths)

    if not remainder.strip():
        file_types_resolved = resolve_file_types(file_types)
        conn = init_db()
        placeholders = ",".join("?" * len(relevant_paths))
        rows = conn.execute(
            f"SELECT items.file_path, items.media_type, items.timestamp_seconds, "
            f"indexed_files.date_taken, indexed_files.lat, indexed_files.lon "
            f"FROM items JOIN indexed_files ON items.file_path = indexed_files.file_path "
            f"WHERE items.file_path IN ({placeholders})",
            list(relevant_paths),
        ).fetchall()
        conn.close()

        results = []
        for stored_path, media_type, timestamp_seconds, date_taken, lat, lon in rows:
            if file_types_resolved and Path(stored_path).suffix.lower() not in file_types_resolved:
                continue
            if after and (not date_taken or date_taken < after):
                continue
            if before and (not date_taken or date_taken > before):
                continue
            real_path = resolve_portable_path(stored_path, EXTERNAL_DRIVE_LABEL)
            if not real_path or not real_path.exists():
                continue
            results.append({
                "file_path": str(real_path), "media_type": media_type,
                "timestamp_seconds": timestamp_seconds, "date_taken": date_taken,
                "lat": lat, "lon": lon, "score": None,
                "matched_people": mentioned, "match_count": _match_count(stored_path),
            })
        results.sort(key=lambda r: (r["match_count"], r["date_taken"] or ""), reverse=True)
        return results[:top_k]

    # There's real text to rank by - reuse search()'s full pipeline (type/date
    # filtering, portable-path resolution, existence checks) with an
    # effectively unbounded top_k (cheap: it only changes the final slice
    # size, CLIP scores the whole library either way), then restrict to the
    # union of mentioned people's files and re-rank by (match_count, score).
    candidates = search(remainder, top_k=100000, after=after, before=before, file_types=file_types)

    real_to_portable = {}
    for portable in relevant_paths:
        real = resolve_portable_path(portable, EXTERNAL_DRIVE_LABEL)
        if real:
            real_to_portable[str(real)] = portable

    results = []
    for r in candidates:
        portable = real_to_portable.get(r["file_path"])
        if portable is None:
            continue
        r["matched_people"] = mentioned
        r["match_count"] = _match_count(portable)
        results.append(r)
    results.sort(key=lambda r: (r["match_count"], r["score"]), reverse=True)
    return results[:top_k]


def open_files(paths, profile=None):
    """Opens every given file in the OS's default viewer for visual
    playtesting. On Mac, `open` with multiple paths launches Preview/
    QuickTime once with all of them browsable via the sidebar rather than
    a separate window per file. Windows has no equivalent single command,
    so each file gets its own os.startfile() call instead."""
    profile = profile or get_os_profile()
    if profile["os"] == "Darwin":
        subprocess.run(["open", *paths])
    else:
        import os
        for path in paths:
            os.startfile(path)


GALLERY_STYLE = """
  body { background:#111; color:#eee; font-family:-apple-system,sans-serif; padding:20px; margin:0; }
  h1 { font-size:15px; font-weight:normal; color:#999; margin:0 0 16px; }
  .grid { display:grid; grid-template-columns:repeat(auto-fill, minmax(260px, 1fr)); gap:16px; }
  .card { background:#1c1c1c; border-radius:8px; padding:8px; }
  .card img, .card video { width:100%; max-height:240px; object-fit:contain; background:#000; border-radius:4px; display:block; }
  .relevance { display:flex; align-items:center; gap:6px; margin-bottom:6px; }
  .bar { flex:1; height:6px; background:#333; border-radius:3px; overflow:hidden; }
  .fill { height:100%; background:#4caf50; }
  .pct { font-weight:bold; color:#4caf50; font-family:monospace; font-size:12px; min-width:32px; text-align:right; }
  .raw { font-size:10px; color:#666; font-family:monospace; }
  .info { display:flex; justify-content:space-between; gap:8px; font-size:11px; color:#aaa; margin-top:6px; }
  .info .type { color:#777; text-transform:capitalize; }
  .path { font-size:10px; color:#777; word-break:break-all; margin-top:4px; }
  .people { font-size:11px; color:#9cf; margin-top:4px; }
  .searchform { display:flex; gap:8px; margin-bottom:20px; flex-wrap:wrap; }
  .searchform input[type=text] { flex:1; min-width:200px; padding:8px; font-size:14px; background:#1c1c1c; border:1px solid #333; color:#eee; border-radius:4px; }
  .searchform input[type=number], .searchform input[type=date], .searchform select { padding:8px; background:#1c1c1c; border:1px solid #333; color:#eee; border-radius:4px; }
  .searchform button { padding:8px 16px; background:#4caf50; color:#fff; border:none; border-radius:4px; cursor:pointer; font-size:14px; }
"""


def get_thumbnail(file_path, media_type, timestamp_seconds, profile=None):
    """A browser-renderable JPEG thumbnail for one result, generated once
    and cached (keyed by path+timestamp, so repeat searches are instant).

    Needed because raw files often aren't directly displayable: browsers
    can't show HEIC in an <img> tag at all (only Pillow/pillow_heif can
    decode it - confirmed live, HEIC results showed a broken-image icon),
    and a bare <video> tag shows a black box with no frame until playback
    actually starts, defeating the point of a visual gallery. Returns None
    (caller falls back to the source file) if generation fails."""
    profile = profile or get_os_profile()
    thumbnail_dir = get_thumbnail_dir()
    thumbnail_dir.mkdir(parents=True, exist_ok=True)
    key = hashlib.sha1(f"{file_path}:{timestamp_seconds}".encode()).hexdigest()
    thumb_path = thumbnail_dir / f"{key}.jpg"
    if thumb_path.exists():
        return thumb_path

    try:
        if media_type == "video":
            cmd = [
                profile["ffmpeg_binary"], "-y", "-ss", str(timestamp_seconds or 0),
                "-i", str(file_path), "-frames:v", "1", "-qscale:v", "4", str(thumb_path),
            ]
            subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        else:
            img = Image.open(file_path).convert("RGB")
            img.thumbnail((480, 480))
            img.save(thumb_path, "JPEG", quality=85)
        return thumb_path if thumb_path.exists() else None
    except Exception:
        return None


def build_gallery_cards(results, for_server=False):
    """Renders just the result cards (score bar + thumbnail/video + path) -
    the reusable part shared between the one-shot CLI gallery and the
    live search GUI server, so there's one place that defines what a
    result card looks like, not two.

    for_server=True builds relative HTTP URLs (/media, /thumb) instead of
    file:// URIs. This matters, not just a style choice: the CLI gallery
    opens a real file:// page, where file:// asset references are same-
    origin and allowed. The live server serves the page over http://, and
    browsers block a page loaded over http(s) from reaching file:// paths
    at all (security boundary, not a caching quirk) - confirmed live, this
    is why thumbnails/video stayed black through the server until this was
    added. Serving the actual bytes back over HTTP is the real fix.

    Relevance bar: the raw cosine score is meaningless on its own (its
    "good" range shifts depending on the model/query), so each result is
    also shown as a percentage of THIS query's own top score - relative
    ranking within the result set, not an absolute quality claim. Real
    absolute quality bands (e.g. "strong match") come later, once a model
    is settled on and there's enough playtesting to calibrate real
    thresholds against."""
    # Real scores exist for every result unless smart_search() (2026-08-24) had
    # nothing left to rank by after pulling a person's name out of the query
    # (e.g. searching just "huy") - those results carry score=None instead of
    # a meaningless embed-of-nothing number, so top_score/the relevance bar
    # both need to tolerate it, not just the individual score below.
    scored = [r for r in results if r["score"] is not None]
    top_score = scored[0]["score"] if scored else 1.0
    profile = get_os_profile()
    cards = []
    for r in results:
        thumb_path = get_thumbnail(r["file_path"], r["media_type"], r["timestamp_seconds"], profile)

        if for_server:
            from urllib.parse import quote
            source_uri = f"/media?path={quote(r['file_path'])}"
            thumb_uri = f"/thumb/{thumb_path.name}" if thumb_path else source_uri
        else:
            source_uri = Path(r["file_path"]).resolve().as_uri()
            thumb_uri = thumb_path.resolve().as_uri() if thumb_path else source_uri

        if r["media_type"] == "video":
            ts = r["timestamp_seconds"] or 0
            media_tag = f'<video src="{source_uri}#t={ts}" poster="{thumb_uri}" controls preload="metadata"></video>'
        else:
            media_tag = f'<img src="{thumb_uri}" loading="lazy">'

        if r["score"] is None:
            relevance_html = '<div class="relevance"><span class="raw">no query text to rank by - sorted by date</span></div>'
        else:
            relative_pct = max(0.0, r["score"] / top_score * 100) if top_score else 0.0
            relevance_html = f"""<div class="relevance">
                <div class="bar"><div class="fill" style="width:{relative_pct:.0f}%"></div></div>
                <span class="pct">{relative_pct:.0f}%</span>
                <span class="raw">raw {r['score']:.3f}</span>
            </div>"""

        # matched_people/match_count only exist on smart_search() results that
        # detected at least one person - makes the "prioritizing" behavior
        # visible rather than a silent black box.
        people_html = ""
        if r.get("matched_people"):
            people_html = f'<div class="people">👤 {html.escape(", ".join(r["matched_people"]))} ({r["match_count"]} matched)</div>'

        cards.append(f"""
        <div class="card">
            {relevance_html}
            {media_tag}
            <div class="info">
                <span class="date">{html.escape(r['date_taken']) if r.get('date_taken') else 'no date on file'}</span>
                <span class="type">{r['media_type']}{f" · GPS {r['lat']:.4f},{r['lon']:.4f}" if r.get('lat') else ""}</span>
            </div>
            {people_html}
            <div class="path">{html.escape(r['file_path'])}</div>
        </div>""")
    return "".join(cards)


def build_gallery_page(query, results, extra_top_html=""):
    """Full HTML page: style + optional extra content above the results
    (the search server injects a search box here) + the result grid."""
    header = f'<h1>Query: &quot;{html.escape(query)}&quot; &mdash; {len(results)} results &mdash; % is relative to this query\'s own top match, not an absolute quality score</h1>' if query else ""
    return f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Search: {html.escape(query)}</title>
<style>{GALLERY_STYLE}</style></head>
<body>
  {extra_top_html}
  {header}
  <div class="grid">{build_gallery_cards(results)}</div>
</body></html>"""


def render_gallery(query, results):
    """Builds a single local HTML page showing every result as an actual
    visual thumbnail with its score, instead of opening a pile of separate
    Preview/QuickTime windows. Videos are shown as a seekable <video>
    jumped straight to the matched timestamp (via the #t= media fragment)
    rather than starting from 0:00 - the whole point is seeing WHY a
    result matched, not hunting through the clip to find it. One-shot:
    writes a file and opens it. For repeated interactive querying without
    dropping back to the terminal each time, see media_search_server.py."""
    page = build_gallery_page(query, results)
    gallery_path = Path(tempfile.gettempdir()) / "huys_search_gallery.html"
    gallery_path.write_text(page)
    webbrowser.open(gallery_path.as_uri())
    return gallery_path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Playtest the media search engine.")
    parser.add_argument("query", nargs="+", help="Search text, e.g. \"dog jumping in the lake\"")
    parser.add_argument("--top", type=int, default=12, help="How many results to return (default: 12)")
    parser.add_argument("--after", help="Only files dated on/after this date, e.g. 2026-06-01")
    parser.add_argument("--before", help="Only files dated on/before this date, e.g. 2026-08-31")
    parser.add_argument("--type", help="Filter by file type: video, image, or specific extensions e.g. mov,mp4")
    parser.add_argument("--gallery", action="store_true", help="Open a visual HTML gallery of all results (recommended)")
    parser.add_argument("--open", action="store_true", help="Open every result individually in the default viewer")
    args = parser.parse_args()

    query_text = " ".join(args.query)
    results = smart_search(query_text, top_k=args.top, after=args.after, before=args.before, file_types=args.type)

    if not results:
        print("No results - has the index been built yet? Run: python3 media_index.py")
    for r in results:
        ts = f" @ {int(r['timestamp_seconds'])}s" if r["timestamp_seconds"] is not None else ""
        date = f"  [{r['date_taken']}]" if r["date_taken"] else ""
        score_str = f"{r['score']:.3f}" if r["score"] is not None else "  -  "
        people = f"  ({', '.join(r['matched_people'])})" if r.get("matched_people") else ""
        print(f"{score_str}  {r['file_path']}{ts}{date}{people}")

    if args.gallery and results:
        render_gallery(query_text, results)
    elif args.open and results:
        open_files([r["file_path"] for r in results])
