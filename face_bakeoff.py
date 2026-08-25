"""
face_bakeoff.py - one-time evaluation tool, NOT part of the indexing pipeline.

Runs two face-recognition backends (insightface's buffalo_l = SCRFD detector +
ArcFace recognizer, and facenet-pytorch's MTCNN + InceptionResnetV1 trained on
VGGFace2 = the FaceNet family) over the same random sample of real Camera Roll
photos, clusters each backend's face embeddings independently (HDBSCAN,
unsupervised - no labels needed), and writes:
  - report.html       - faces grouped by cluster, for judging cluster quality
  - diagnostics.html  - every detected face sorted by detection score, crop
                         size, and blur, worst first - for picking real
                         filter thresholds by eye rather than guessing

Read-only against the drive - only ever opens images, never writes/moves/deletes
anything there. All output goes to face_bakeoff_output/ in the project root
(gitignored - contains real face crops).

The insightface backend's detection/quality-filter logic (passes_quality_filter,
blur_variance, load_insightface_backend, detect_insightface) now lives in
face_index.py, the productionized pipeline - imported back here rather than
duplicated, so any future model bake-off compares against the exact same
tuned logic the real app actually uses, with no risk of the two drifting apart.

    python3 face_bakeoff.py                    # default sample of 300 images
    python3 face_bakeoff.py --sample-size 500
    python3 face_bakeoff.py --seed 7            # different reproducible sample
"""
import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
from PIL import Image
import pillow_heif

pillow_heif.register_heif_opener()

from config import EXTERNAL_DRIVE_LABEL, find_volume_by_label, walk_media_files
from media_index import IMAGE_EXTENSIONS
from face_index import (
    MIN_CLUSTER_SIZE, blur_variance, detect_insightface, load_image_rgb,
    load_insightface_backend, passes_quality_filter,
)

APP_DIR = Path(__file__).resolve().parent
OUTPUT_DIR = APP_DIR / "face_bakeoff_output"
DEFAULT_SAMPLE_SIZE = 300
DEFAULT_SEED = 42


def find_camera_roll_root(drive_root):
    return drive_root / "Camera Roll"


def sample_images(camera_roll_root, n, seed=DEFAULT_SEED):
    """Random sample of real image files (not video - kept out of scope for this
    first pass) across the whole Camera Roll, so the sample spans many months/
    people rather than clustering within one narrow time slice."""
    all_images = sorted(p for p in walk_media_files(camera_roll_root) if p.suffix.lower() in IMAGE_EXTENSIONS)
    rng = random.Random(seed)
    if len(all_images) <= n:
        return all_images
    return rng.sample(all_images, n)


# --- Backend: facenet-pytorch (MTCNN detection + InceptionResnetV1/VGGFace2 recognition) ---

def load_facenet_backend():
    from facenet_pytorch import MTCNN, InceptionResnetV1
    mtcnn = MTCNN(image_size=160, margin=20, keep_all=True, device="cpu")
    resnet = InceptionResnetV1(pretrained="vggface2").eval()
    return mtcnn, resnet


def detect_facenet(backend, pil_img):
    import torch
    mtcnn, resnet = backend
    boxes, probs = mtcnn.detect(pil_img)
    aligned = mtcnn(pil_img)  # standardized tensors ready for the recognizer, or None
    if aligned is None or boxes is None:
        return []
    if aligned.dim() == 3:
        aligned = aligned.unsqueeze(0)
    with torch.no_grad():
        embeddings = resnet(aligned).numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    results = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = [max(int(v), 0) for v in box]
        crop = pil_img.crop((x1, y1, x2, y2))
        if crop.size[0] == 0 or crop.size[1] == 0:
            continue
        prob = probs[i] if probs is not None and i < len(probs) and probs[i] is not None else 0.0
        results.append({
            "bbox": (x1, y1, x2, y2),
            "embedding": embeddings[i].astype(np.float32),
            "crop": crop,
            "det_score": float(prob),
            "blur": blur_variance(crop),
            "width": crop.size[0],
            "height": crop.size[1],
        })
    return results


BACKENDS = {
    "insightface_buffalo_l": (load_insightface_backend, detect_insightface),
}

# facenet-pytorch (detect_facenet/load_facenet_backend below) is kept in the file but dropped
# from BACKENDS - real bake-off evidence (2026-08-17) showed it slower and less discriminative
# (merged a different person into insightface's dominant cluster), and its embeddings can't be
# combined with insightface's anyway (different, incompatible vector spaces) - no real ensemble
# benefit identified, so effort now goes into tuning insightface rather than re-testing facenet.


def run_backend(name, image_paths, output_dir):
    load_fn, detect_fn = BACKENDS[name]
    backend_dir = output_dir / name
    crops_dir = backend_dir / "crops"
    crops_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{name}] loading model...")
    backend = load_fn()

    faces = []  # list of dicts: source, bbox, embedding, crop_filename, det_score, blur, width, height
    start = time.time()
    for i, path in enumerate(image_paths):
        try:
            img = load_image_rgb(path)
        except Exception as e:
            print(f"[{name}] skipped unreadable {path.name}: {e}")
            continue
        for j, face in enumerate(detect_fn(backend, img)):
            crop_name = f"{i:05d}_{j}.jpg"
            face["crop"].save(crops_dir / crop_name, "JPEG", quality=85)
            faces.append({
                "source": str(path),
                "bbox": face["bbox"],
                "embedding": face["embedding"],
                "crop_filename": crop_name,
                "det_score": face["det_score"],
                "blur": face["blur"],
                "width": face["width"],
                "height": face["height"],
            })
        if (i + 1) % 50 == 0:
            print(f"[{name}] {i + 1}/{len(image_paths)} images, {len(faces)} faces so far")
    elapsed = time.time() - start
    print(f"[{name}] done: {len(image_paths)} images -> {len(faces)} faces in {elapsed:.1f}s")

    if not faces:
        return {"name": name, "faces": [], "elapsed": elapsed, "cluster_count": 0, "noise_count": 0}

    for face in faces:
        face["passes_filter"] = passes_quality_filter(face)
    kept = [f for f in faces if f["passes_filter"]]
    filtered_out = len(faces) - len(kept)

    cluster_count = noise_count = 0
    if kept:
        kept_embeddings = np.stack([f["embedding"] for f in kept])
        import hdbscan
        clusterer = hdbscan.HDBSCAN(min_cluster_size=MIN_CLUSTER_SIZE, metric="euclidean")
        labels = clusterer.fit_predict(kept_embeddings)
        for face, label in zip(kept, labels):
            face["cluster"] = int(label)
        cluster_count = len(set(labels[labels >= 0]))
        noise_count = int((labels == -1).sum())
    for face in faces:
        if not face["passes_filter"]:
            face["cluster"] = -2  # excluded by the quality filter - never attempted clustering

    print(f"[{name}] {filtered_out} faces filtered out by quality gate, "
          f"{cluster_count} clusters, {noise_count} unclustered (of {len(kept)} kept)")

    write_report(name, kept, backend_dir, elapsed, len(image_paths), cluster_count, noise_count, filtered_out)
    write_diagnostics_report(name, faces, backend_dir)  # full set, including filtered-out, for threshold review
    with open(backend_dir / "results.json", "w") as f:
        json.dump([{k: v for k, v in face.items() if k != "embedding"} for face in faces], f, indent=2)
    np.save(backend_dir / "embeddings.npy", np.stack([f["embedding"] for f in faces]))

    return {"name": name, "faces": faces, "elapsed": elapsed, "cluster_count": cluster_count,
            "noise_count": noise_count, "filtered_out": filtered_out}


REPORT_STYLE = """
<style>
body { font-family: -apple-system, sans-serif; background: #111; color: #eee; margin: 2rem; }
h1 { font-size: 1.2rem; } h2 { font-size: 1rem; color: #9cf; margin-top: 2rem; }
.stats { color: #aaa; margin-bottom: 1.5rem; }
.grid { display: flex; flex-wrap: wrap; gap: 6px; }
.face { display: flex; flex-direction: column; align-items: center; width: 100px; }
.face img { width: 100px; height: 100px; object-fit: cover; border-radius: 4px; }
.cap { font-size: 0.65rem; color: #9a9; text-align: center; line-height: 1.2; }
</style>
"""


def _face_tile(face, crops_rel="crops"):
    return (f'<div class="face"><img src="{crops_rel}/{face["crop_filename"]}" title="{Path(face["source"]).name}">'
            f'<div class="cap">score {face["det_score"]:.2f}<br>{face["width"]}x{face["height"]}<br>blur {face["blur"]:.0f}</div></div>')


def write_report(name, faces, backend_dir, elapsed, image_count, cluster_count, noise_count, filtered_out=0):
    by_cluster = {}
    for face in faces:
        by_cluster.setdefault(face["cluster"], []).append(face)
    ordered = sorted((c for c in by_cluster if c != -1), key=lambda c: -len(by_cluster[c]))

    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>Face bake-off: {name}</title>{REPORT_STYLE}</head><body>
<h1>{name}</h1>
<div class="stats">{image_count} images scanned - {len(faces) + filtered_out} faces detected
({filtered_out} dropped by the quality filter, {len(faces)} kept) - {cluster_count} clusters - {noise_count} unclustered - {elapsed:.1f}s
- <a href="diagnostics.html" style="color:#9cf">diagnostics (sorted by score/size/blur)</a></div>
"""]
    for cluster_id in ordered:
        members = by_cluster[cluster_id]
        parts.append(f'<h2>Cluster {cluster_id} ({len(members)} faces)</h2><div class="grid">')
        parts.extend(_face_tile(face) for face in members)
        parts.append("</div>")

    if -1 in by_cluster:
        members = by_cluster[-1]
        parts.append(f'<h2>Unclustered ({len(members)} faces)</h2><div class="grid">')
        parts.extend(_face_tile(face) for face in members)
        parts.append("</div>")

    parts.append("</body></html>")
    (backend_dir / "report.html").write_text("".join(parts), encoding="utf-8")


def write_diagnostics_report(name, faces, backend_dir):
    """Every detected face, sorted three ways (worst first), so real
    thresholds for det_score / crop size / blur can be picked by looking at
    where quality actually visibly drops off - not guessed at."""
    sections = [
        ("Sorted by detection score (ascending - low confidence first)", lambda f: f["det_score"]),
        ("Sorted by crop size (ascending - smallest first)", lambda f: f["width"] * f["height"]),
        ("Sorted by blur variance (ascending - most blurred first)", lambda f: f["blur"]),
    ]
    parts = [f"""<!doctype html><html><head><meta charset="utf-8">
<title>Face bake-off diagnostics: {name}</title>{REPORT_STYLE}</head><body>
<h1>{name} - diagnostics</h1>
<div class="stats">{len(faces)} faces total - <a href="report.html" style="color:#9cf">back to cluster report</a></div>
"""]
    for title, key in sections:
        parts.append(f'<h2>{title}</h2><div class="grid">')
        parts.extend(_face_tile(face) for face in sorted(faces, key=key))
        parts.append("</div>")
    parts.append("</body></html>")
    (backend_dir / "diagnostics.html").write_text("".join(parts), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-size", type=int, default=DEFAULT_SAMPLE_SIZE)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    args = parser.parse_args()

    drive_root = find_volume_by_label(EXTERNAL_DRIVE_LABEL)
    if not drive_root:
        raise RuntimeError(f"Drive '{EXTERNAL_DRIVE_LABEL}' not connected - plug it in and try again.")

    camera_roll = find_camera_roll_root(drive_root)
    images = sample_images(camera_roll, args.sample_size, args.seed)
    print(f"Sampled {len(images)} images (seed={args.seed}) from {camera_roll}")

    OUTPUT_DIR.mkdir(exist_ok=True)
    summary = []
    for name in BACKENDS:
        summary.append(run_backend(name, images, OUTPUT_DIR))

    print("\n=== Summary ===")
    for s in summary:
        print(f"{s['name']}: {len(s['faces'])} faces, {s['cluster_count']} clusters, "
              f"{s['noise_count']} unclustered, {s['elapsed']:.1f}s")
    print(f"\nReports written under {OUTPUT_DIR} - open each backend's report.html / diagnostics.html to compare.")


if __name__ == "__main__":
    main()
