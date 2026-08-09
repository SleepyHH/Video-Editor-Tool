# Testing checklist

Two mini-projects share this repo - the video editor (`main.py`) and the
step-2 media search engine (`config.py`'s portable-path helpers,
`media_index.py`, `media_search.py`, `media_search_server.py`). Each has its
own automated test file; run the one(s) covering whatever you touched.

## 1. Automated (fast, run every time)

```
python3 test_helpers.py        # main.py - the video editor
python3 test_media_search.py   # config.py, media_index.py, media_search.py, media_search_server.py
```

`test_helpers.py` covers the pure logic and ffmpeg probes without needing the
GUI, a real Photos library, or a slow Whisper/render pass: SRT timestamp
formatting, duplicate filename handling, manual-import persistence, the
transcription language list, duration formatting, HEVC/duration detection on
real files already in the Imports folder, the Whisper idle-unload wiring, and
the OS profile in `config.py`.

`test_media_search.py` covers the same kind of pure logic for the search
engine without downloading/loading the real ~1.6GB CLIP model or needing a
real index: portable-path (`DRIVE::...`) round-tripping, the
thumbnail-cache-folder self-exclusion regression guard, file-type filter
parsing, device selection, and the CLIP idle-unload timer's scheduling
mechanics (using a near-zero interval, not the real 10-minute window).

Both finish in ~1-2 seconds combined and should all pass before considering
any change done.

## 2. Manual checklist - video editor (the full feature list)

Automated tests can't drive the actual GUI or wait through a real
transcription/render, so these need a human click-through after any change
that touches the relevant area. Launch with `python3 main.py` - opens on the
"🎬 Editor" tab; the "🔍 Prompt-Style Search" tab is covered separately below.

- [ ] **HDD-first media source** — with the shared drive connected, the media
      list populates from it (not Photos/iCloud); unplugging it and hitting
      "🔄 Refresh Media List" falls back to the original Photos/cloud scan
- [ ] **Photos library scan** — app launches, media list populates with your
      real videos, downloaded videos listed before cloud-only (☁️) ones,
      newest-first within each group
- [ ] **Cloud-only video handling** — selecting a ☁️ item that isn't
      downloaded shows the "not downloaded yet" message (not a crash/freeze);
      re-selecting after downloading it in Photos picks it up automatically
- [ ] **Search box** — typing filters the list live, clearing shows everything again
- [ ] **Manual import** — "Add Video Files Manually" adds a file to the *top*
      of the list; it's still there after "🔄 Refresh Media List" and after
      fully quitting and relaunching the app
- [ ] **Video preview** — clicking a video plays it; timeline scrubbing by
      clicking the bottom track works
- [ ] **Auto cut-preview** — selecting a video automatically loads the
      real (not random) silence-detection preview; nudging the margin or
      sensitivity sliders live-updates it without needing a manual refresh
- [ ] **Transcription** — pick a language from the dropdown (100 languages +
      "Mixed languages"), hit "🎙️ Transcribe Audio", confirm text streams into
      the preview box and a `.srt` lands in the Export folder (or wherever
      "📍 Choose save location" was pointed)
- [ ] **Translate to English** — with the checkbox on, output is a real
      English translation, not just the original-language transcript
- [ ] **Auto-Cut render** — "🚀 Run Auto-Cut" produces a real video; for an
      HEVC source, conversion happens automatically (or is skipped if
      "⏭️ Skip conversion" is checked) and the video is NOT black and NOT
      sideways either way
- [ ] **Post-render auto-transcribe** — with "🎙️ Also transcribe rendered
      output" checked, the `.srt` lands right next to the rendered video
- [ ] **Success summary** — after a render, the popup shows original length,
      rendered length, time cut, and a per-stage timing breakdown
- [ ] **Import/Export folder buttons** — both open the correct real folder in Finder
- [ ] **Remove from list** — selecting a row and hitting "🗑️ Remove Selected
      from List" removes it from the list only; the underlying file is
      untouched (still exists, still playable elsewhere)
- [ ] **Tab-switch pauses playback** — start playing a video on this tab,
      switch to "🔍 Prompt-Style Search", confirm playback paused (and vice
      versa for a video playing in the search tab's preview pane)

## 3. Manual checklist - Prompt-Style Search tab (native, in main.py)

Built 08-08-2026 as Qt widgets directly in `main.py`, not the browser gallery -
same reasons anything Qt/CLIP-model-dependent can't be automated. Launch
`python3 main.py` and switch to the "🔍 Prompt-Style Search" tab.

- [ ] **Search returns real results** — a query returns relevant result
      cards with a real (non-black) thumbnail, date/type/GPS info, video
      length for video results, and file name
- [ ] **Filters** — type dropdown (All/Video/Image) and result count (default
      20) actually narrow the results
- [ ] **Date fields start empty** — "From"/"To" show placeholder text, apply
      no filter until clicked; clicking either jumps it to a real default
      (a year ago / today) rather than a blank or 1752 date, and *that* date
      then filters correctly
- [ ] **App stays responsive during a search** — the rest of the app (e.g.
      switching back to the Editor tab, or starting a transcription) doesn't
      freeze while a search is running - confirms the QThread worker is
      actually keeping this off the UI thread
- [ ] **Hover-only overlay** — a result thumbnail shows plain, unobscured
      image with the cursor away from it; hovering reveals "Preview" (top)
      and "Select"/"✓ Selected" (bottom), both fairly transparent so the
      image stays visible underneath; a checked card's "✓ Selected" stays
      visible even without hovering
- [ ] **Preview vs. select zones** — clicking the top ~60% of a thumbnail
      loads it into the shared preview pane (seeked to the matched timestamp
      for video); clicking the bottom ~40% toggles selection instead
- [ ] **Shared preview pane sizing** — roughly half the tab's width, video
      plays with sound and seeks correctly
- [ ] **Preview timeline** — a scrub bar appears below the preview for video
      results (hidden for image results); clicking anywhere on it jumps
      playback there, and the playhead moves live during playback
- [ ] **Responsive grid** — resizing the app window changes how many result
      cards fit per row, without needing to re-run the search
- [ ] **Confirm Selection** — stays disabled with nothing selected, enables
      the moment one card is selected; clicking it switches to the Editor tab
- [ ] **Confirm Selection dedupes** — selecting a result that's already
      somewhere in the media list (not a new import) moves it to the top
      instead of creating a second entry for the same file
- [ ] **Confirm Selection hides the rest** — after confirming, only the
      just-confirmed item(s) are visible in the Editor tab's list; hitting
      "🔄 Refresh Media List" reveals everything again with the confirmed
      item(s) still pinned at the top
- [ ] **Selection persists after confirming** — the imported items are still
      at the top after fully quitting/relaunching the app (same persistence
      as manual imports, since it's the same mechanism)
- [ ] **CLIP idle-unload** — after ~10 minutes with no search from this tab,
      memory usage for the app drops back down (Activity Monitor/Task
      Manager) rather than staying pinned at the loaded-model level
- [ ] **App startup stays fast** — launching `main.py` without ever touching
      the search tab doesn't load torch/open_clip (check startup time/memory
      hasn't regressed) - confirms the lazy-import stayed lazy

## 4. Manual checklist - media search engine (standalone browser GUI)

Automated tests don't load the real CLIP model, scan the real library, or
drive a browser, so these need a human check after any change that touches
the relevant area. Launch the GUI with `python3 media_search_server.py`.

- [ ] **Search returns real results** — a query returns relevant items, not
      an empty grid or a crash; relevance bar and date/GPS info line render
      per card
- [ ] **File-type filter** — the type dropdown (All/Video/Image) actually
      narrows results to that type
- [ ] **Date filter** — after/before date fields narrow results correctly
- [ ] **Thumbnails and playback** — thumbnails load (not black/broken) for
      both HEIC photos and video; clicking a video result seeks to the
      matched timestamp and plays
- [ ] **Cross-machine path resolution** — a search run on whichever machine
      didn't just index correctly resolves `DRIVE::...` paths to real,
      playable files (not "file not found")
- [ ] **CLIP idle-unload** — after ~10 minutes with no search, memory usage
      for the server process drops back down (check Activity Monitor/Task
      Manager) rather than staying pinned at the loaded-model level
- [ ] **Indexing end-to-end** — `python3 media_index.py` (or however it's
      invoked) picks up new/changed files, doesn't re-embed unchanged ones,
      and a subsequent search finds the newly-indexed content

## 5. When adding a new feature

- If it's pure logic (no GUI, no slow external call) → add a test to
  `test_helpers.py` (video editor) or `test_media_search.py` (search engine),
  not just this checklist.
- If it's GUI-driven or depends on Whisper/Photos/rendering/the real CLIP
  model → add a line to the relevant manual checklist above instead.
