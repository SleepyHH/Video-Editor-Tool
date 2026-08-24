# Testing checklist

Two mini-projects share this repo - the video editor (`main.py`) and the
step-2 media search engine (`config.py`'s portable-path helpers,
`media_index.py`, `media_search.py`, `media_search_server.py`), plus the
face-recognition pipeline (`face_index.py`, its own People tab in `main.py`).
Each has its own automated test file; run the one(s) covering whatever you
touched.

## 1. Automated (fast, run every time)

```
python3 test_helpers.py        # main.py - the video editor
python3 test_media_search.py   # config.py, media_index.py, media_search.py, media_search_server.py
python3 test_face_index.py     # face_index.py - detection filter, clustering, labeling
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
- [ ] **Person filter narrows to just that person** — the "All people"
      dropdown is empty (just "All people") until this tab is actually
      opened once, then lists every labeled name; picking one and searching
      returns only files with a face labeled as them; picking a name with
      no matching content returns 0 results rather than erroring; switching
      back to this tab later keeps the same person selected if it's still a
      real label, or falls back to "All people" if it was renamed/merged away
- [ ] **Typing a person's name into the query itself works, no dropdown
      needed - but it has to be capitalized** — searching e.g. "Huy and
      cats" (leave the dropdown on "All people") returns only Huy's files,
      ranked by "cats"; the status line under the search box says "matched:
      Huy" so the detection isn't invisible; searching just a name alone
      (no other words) still returns that person's files, newest first,
      with no relevance % shown on the cards (there's nothing left to rank
      by)
- [ ] **Lowercase falls straight through to a normal search** — the same
      query typed lowercase ("huy and cats") should NOT filter by person at
      all - no "matched:" note in the status line, and results come from a
      completely ordinary CLIP search of the literal text, same as if
      nobody were labeled "Huy"
- [ ] **Multiple people in one query prioritizes files with more of them**
      — searching e.g. "Huy and An with cats" detects both names (status
      line says "matched: Huy, An"); results are files with EITHER person
      (not just files with both), but any file containing both should sit
      above files with only one, and each card shows which of the mentioned
      people it actually matched
- [ ] **The dropdown and typed names combine** — pick a person from the
      dropdown AND type a different, unrelated query (no name in the text) -
      results should still be restricted to the dropdown person, same as
      before this feature existed
- [ ] **Capitalization is the only disambiguation rule now, by design** —
      "an old photo" (lowercase) should NOT match An; "An old photo"
      (capitalized) should; same test with any other real name that happens
      to also be an English word (e.g. a color or food word) - only the
      properly-capitalized version triggers the person filter, confirming
      there's no separate hardcoded list of "risky" names anymore
      (face_index.py's `extract_mentioned_people()` comment explains why)
- [ ] **A compound family-relation name doesn't fragment into its own
      component names** — for someone like "Fraser's Mum," who has "Fraser"
      and "Mum" separately labeled too, searching "Fraser's Mum at the
      park" must detect just the one person "Fraser's Mum," not fragment
      into two separate matches for "Fraser" and "Mum"; searching "Fraser
      and Mum at the park" (no possessive) should still correctly detect
      both of them as two separate people - all labels have been normalized
      to consistent Title Case specifically so this resolves correctly
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

## 4. Manual checklist - People tab (native, in main.py)

Built 2026-08-19 as the labeling UI for `face_index.py`'s detect/cluster
pipeline - Phase 1 of the face-recognition plan. Reuses the Search tab's
`ResultThumbnail`/`ResultsScrollArea`/`columns_for_width` directly, so most
of section 3's rendering checks apply here too; this list only covers what's
actually new. Launch `python3 main.py`, switch to the "👤 People" tab. Needs
`build_face_index()` (`python3 face_index.py`) to have been run at least
once against a real library first - an empty index just shows "0 pending
group(s)", which is correct behavior, not a bug.

`build_face_index()` itself is CLI-only, same as `media_index.build_index()`
- no GUI trigger for indexing exists for either pipeline. Since Phase 3
(video, 2026-08-24) added a genuinely new algorithm (`consolidate_face_runs()`
- collapsing a video's per-frame detections into one row per continuous
appearance instead of one per sampled second), it needs its own real-data
check, not just unit tests against synthetic frame sequences:
- [ ] **Video face-indexing smoke test** — run `build_face_index()` against
      a small real video-containing folder (or let a full run reach a few
      real videos), then open a resulting video-sourced pending cluster in
      the People tab; the representative crop should be a real, sane face
      (not garbled/misaligned), and a person present continuously across
      several seconds should show up as ONE cluster entry, not several
      near-duplicate ones a second apart

Verified for real, not just synthetic tests, 2026-08-24: ran the actual
detection+consolidation pipeline (not the full `build_face_index()`, just
its core steps) against two short real videos - one produced two correctly-
separated multi-second runs (5s and 3s) for two different real people in the
same clip, both representative crops visually confirmed as real, clearly-
cropped faces, not garbage. A `build_face_index()` run over the whole real
library hasn't been done yet - that's the natural next step whenever it's
wanted, this only validated the pipeline itself works correctly.

- [ ] **App startup stays fast** — launching `main.py` without opening this
      tab doesn't load `insightface`/`onnxruntime` (check startup time/memory
      hasn't regressed) - confirms the lazy-import stayed lazy, same as the
      Search tab's CLIP model
- [ ] **Groups list populates** — pending clusters (biggest first, an
      "Unclustered" entry last if any noise faces exist) and already-labeled
      people (👤 prefixed) both show with correct face counts
- [ ] **Selecting a cluster** — loads its faces into the grid; "✅ Confirm"
      and "🗑️ Discard cluster" appear (not just enable); "✏️ Rename" and "🚫
      Remove from Person" stay hidden
- [ ] **Selecting an already-labeled person** — loads their faces into the
      grid; "✏️ Rename" and "🚫 Remove from Person" appear; "✅ Confirm" and
      "🗑️ Discard cluster" stay hidden
- [ ] **Face cards default checked** — every card starts "✓ Selected" (kept);
      clicking a card's bottom zone unchecks it (excluded from the label),
      clicking again rechecks it
- [ ] **Preview zone** — clicking a card's top zone loads the *source photo*
      (not just the crop) into the right-hand preview pane, including for a
      HEIC source
- [ ] **Confirm labels correctly** — typing a name and hitting Confirm labels
      every *checked* face as that person and leaves unchecked ones alone
      (still pending, reviewable again later); the cluster disappears from
      the pending list and the person appears/grows in the people list
- [ ] **Confirm reuses an existing name** — labeling a second cluster with a
      name that already exists adds to that person's count rather than
      creating a duplicate
- [ ] **Blank name is rejected** — hitting Confirm with an empty name field
      shows a message and labels nothing
- [ ] **Discard cluster** — removes it from the pending list without creating
      a person; those faces don't resurface unless reclustered from scratch
- [ ] **Refresh & Recluster** — re-runs clustering over whatever's still
      unlabeled and refreshes both lists, without disturbing already-labeled
      people
- [ ] **Tab-switch pauses playback** — if a video was playing in another tab,
      switching to People pauses it (same shared-pause wiring as the other
      two tabs)
- [ ] **Enter confirms a label immediately after selecting** — clicking a
      pending cluster in the list, then pressing Enter with no extra click
      into the name box, labels it (or renames, for an already-labeled
      person) - selecting an item moves keyboard focus straight into the
      name box
- [ ] **Suggested name pre-fills on selecting a cluster** — opening a real
      pending cluster (not "Unclustered") that's similar to an already-labeled
      person pre-fills the name box with that person's name (selected, so
      typing overwrites it) and shows a similarity score in the status line;
      still fully editable, and nothing is written unless Confirm is
      actually pressed
- [ ] **Pending list is grouped, not just size-sorted** — clusters suggesting
      the same existing person appear adjacent (biggest suggested-group
      first, shown as e.g. "C 909 (6) → Huy?"); clusters that look like each
      other but match no labeled person yet are paired next to each other
      (e.g. "C 442 (3) ≈ C 218"); everything else falls back to the original
      size order; "Unclustered" always stays last
- [ ] **List labels stay short** — pending clusters show as "C {id} ({count})"
      and labeled people as "👤 {name} ({count})" - no "faces"/"likely"/score
      text bloating the list, so the list column doesn't have to be wide to
      stay readable and the name box below it isn't squeezed out
- [ ] **Pending/people list is only as wide as it needs to be** — the left
      column stops right around its longest current line of text (not a
      fixed fraction of the window); the preview pane on the right is
      noticeably wider than before as a result, roughly matching the
      face-grid panel's width instead of being much narrower
- [ ] **Discard cluster / Rename are stacked, not side by side** — "🗑️
      Discard cluster" and "✏️ Rename" appear one above the other in their
      own column between Confirm and Apply Selected Matches, not spread
      across the row
- [ ] **Name box spans the full tab width, not just the middle column** — the
      "Name this person..." row sits below the three-column area (list /
      face grid / preview), not squeezed inside the face-grid column; it
      should look comfortably wide, not a sliver next to the buttons
- [ ] **Preview image is centered, not stuck top-left** — for a photo whose
      aspect ratio doesn't match the preview pane's (e.g. a portrait photo
      in a wide pane, or vice versa), the empty space splits evenly on both
      sides/top-bottom instead of collecting entirely on one side
- [ ] **Labeled people list is alphabetical** — the 👤-prefixed part of the
      list is sorted A-Z by name (case-insensitive), not by face count, and
      stays alphabetical as more people get labeled
- [ ] **Rename an existing person** — selecting an already-labeled person
      pre-fills the name box with their current name; editing it and hitting
      "✏️ Rename" (or Enter) updates their name everywhere and the people list
      re-sorts to match; Confirm/Discard stay disabled the whole time since
      this isn't a pending cluster
- [ ] **Rename onto an existing name merges** — renaming a person to a name
      that's already used by someone else moves all their faces onto that
      existing person instead of erroring or creating a duplicate; the old
      (now-empty) person entry disappears from the list
- [ ] **Find Matches shows suggestions, all pre-checked, highest score first**
      — with at least one person already labeled, "🔍 Find Matches" shows
      unlabeled faces that resemble them, each card already "✓ Selected" and
      each labeled with the proposed name + similarity score; scores read
      highest-to-lowest across the grid, not scattered; Confirm/Discard stay
      disabled since this isn't a pending cluster
- [ ] **Unchecking a match excludes just that one** — unchecking a suggested
      match's card and hitting "✅ Apply Selected Matches" applies every
      *other* checked match but leaves the unchecked one's face untouched -
      still unlabeled, still available next time Find Matches runs
- [ ] **Apply Selected Matches updates both lists** — after applying, matched
      people's face counts grow and the status line reports how many were
      applied; re-running Find Matches no longer shows the ones just applied
- [ ] **Name box autocompletes from existing people** — typing into "Name
      this person..." pops up a narrowing list of already-labeled names as
      you type (case-insensitive); picking one or continuing to type a new
      name both still work, nothing is forced
- [ ] **Browsing a labeled person starts all-unchecked** — unlike a pending
      cluster (starts all-checked), opening an already-labeled person's
      group shows every face card as unselected; "🚫 Remove from Person"
      stays clickable but does nothing useful until at least one is checked
- [ ] **Remove from Person only touches checked faces** — checking a couple
      of a labeled person's faces and hitting "🚫 Remove from Person" sends
      just those back to the unlabeled pool (they reappear as pending once
      reclustered); every other face keeps its label untouched
- [ ] **Undo reverses the last action exactly** — after Confirm, Discard,
      Remove from Person, or Apply Selected Matches, "↩️ Undo" is enabled;
      clicking it puts the affected faces back exactly as they were
      (including deleting a person if Undo removes their only face) and
      then disables itself again - it does NOT go back further than one step
- [ ] **A new action replaces what Undo can reverse** — do one action, then
      a second one, then Undo - only the second action is undone, the first
      one stays applied (single-level undo, not a history)
- [ ] **Action buttons are hidden, not greyed out, when they don't apply** —
      nothing selected: no Confirm/Discard/Rename/Remove/Apply Selected
      Matches buttons appear at all; a pending cluster: only Confirm/Discard
      show; a labeled person: only Rename/Remove show; Find Matches mode:
      only Apply Selected Matches shows (disabled if there's nothing to
      apply). Switching between these should visibly add/remove buttons,
      not just toggle their greyed-out look
- [ ] **Buttons reappear correctly after Refresh & Recluster** — with a
      cluster or person selected (so its buttons are showing), hit "🔄
      Refresh && Recluster" - those buttons hide during the recluster and
      stay hidden afterward (nothing is selected anymore); clicking a fresh
      item in the list shows the correct pair again, not stuck disabled

## 5. Manual checklist - media search engine (standalone browser GUI)

Automated tests don't load the real CLIP model, scan the real library, or
drive a browser, so these need a human check after any change that touches
the relevant area. Launch the GUI with `python3 media_search_server.py`.

- [ ] **Search returns real results** — a query returns relevant items, not
      an empty grid or a crash; relevance bar and date/GPS info line render
      per card
- [ ] **File-type filter** — the type dropdown (All/Video/Image) actually
      narrows results to that type
- [ ] **Person filter** — the dropdown lists every labeled person; picking
      one and searching narrows results to just their files; the selection
      round-trips correctly through the page reload (`?person=...` in the URL)
- [ ] **Natural-language person detection works here too, same capitalization
      rule** — typing e.g. "Huy and cats" into the search box (dropdown left
      on "All people") narrows to Huy's files and shows a "👤 Huy (1
      matched)" line on each result card; the same query lowercase ("huy
      and cats") does NOT filter, matching the native Search tab's behavior
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

## 6. When adding a new feature

- If it's pure logic (no GUI, no slow external call) → add a test to
  `test_helpers.py` (video editor), `test_media_search.py` (search engine),
  or `test_face_index.py` (face recognition), not just this checklist.
- If it's GUI-driven or depends on Whisper/Photos/rendering/the real CLIP
  model → add a line to the relevant manual checklist above instead.
