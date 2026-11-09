"""
Proof-of-concept v2: can a face regenerated from landmarks + one photo
(a) still match the original identity and (b) slip past deepfake detectors?

What the generator receives (SD 1.5 + ControlNet + IP-Adapter, unchanged model):
  1. ControlNet   : MediaPipe landmark drawing  -> pose, expression, face geometry
  2. IP-Adapter   : face crop                   -> identity / appearance
  3. Text prompt  : BLIP caption of the photo   -> glasses, beard, clothes, background
                    (+ optional --extra text)
  4. Best-of-N    : N seeds, keep the candidate FaceNet likes most

Defences applied to the result:
  A. Identity verifier  : FaceNet cosine similarity (+ impostor baseline in batch mode)
  B. Learned detectors  : one older face-deepfake ViT + one diffusion-era detector
  C. Frequency analysis : radial power spectrum and its slope

Fairness fixes vs v1: replicate-padding instead of black borders, and both real and
generated faces go through the same crop / resize / JPEG pipeline before analysis.

Usage:
  python seminar_demo.py my_face.jpg
  python seminar_demo.py faces_folder/            # batch mode -> summary.csv + summary.png
"""
import argparse
import csv
import gc
import io
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import mediapipe as mp  # noqa: E402

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
mp_det = mp.solutions.face_detection
mp_mesh = mp.solutions.face_mesh

FAKE_LABELS = {"fake", "deepfake", "artificial", "ai", "ai-generated", "generated", "synthetic"}
DEFAULT_DETECTORS = ["dima806/deepfake_vs_real_image_detection",  # older face-deepfake data
                     "Organika/sdxl-detector"]                    # trained on diffusion output
NEGATIVE = ("cartoon, illustration, painting, 3d render, airbrushed, plastic skin, blurry, "
            "deformed, extra eyes, lowres, watermark, text")

MESH_GROUPS = [
    (mp_mesh.FACEMESH_FACE_OVAL, (10, 200, 10)),
    (mp_mesh.FACEMESH_LIPS, (10, 180, 10)),
    (mp_mesh.FACEMESH_RIGHT_EYE, (10, 200, 180)),
    (mp_mesh.FACEMESH_RIGHT_EYEBROW, (10, 220, 180)),
    (mp_mesh.FACEMESH_RIGHT_IRIS, (10, 200, 250)),
    (mp_mesh.FACEMESH_LEFT_EYE, (180, 200, 10)),
    (mp_mesh.FACEMESH_LEFT_EYEBROW, (180, 220, 10)),
    (mp_mesh.FACEMESH_LEFT_IRIS, (250, 200, 10)),
]


class NoFace(Exception):
    pass


# --------------------------------------------------------------------------
# Face detection, cropping, landmarks
# --------------------------------------------------------------------------
def detect_face(img):
    """Return (cx, cy, side) of the most confident face, in pixels."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    with mp_det.FaceDetection(model_selection=1, min_detection_confidence=0.5) as d:
        res = d.process(arr)
    if not res.detections:
        raise NoFace("no face detected")
    det = max(res.detections, key=lambda x: x.score[0])
    b = det.location_data.relative_bounding_box
    bw, bh = b.width * w, b.height * h
    return b.xmin * w + bw / 2, b.ymin * h + bh / 2, max(bw, bh)


def pad_crop(img, face, scale):
    """Square crop around the face. Out-of-frame areas replicate edge pixels (no black bars)."""
    cx, cy, side = face
    half = side * scale / 2
    x0, y0, x1, y1 = (int(round(v)) for v in (cx - half, cy - half, cx + half, cy + half))
    arr = np.array(img)
    h, w = arr.shape[:2]
    pl, pt, pr, pb = max(0, -x0), max(0, -y0), max(0, x1 - w), max(0, y1 - h)
    if pl or pt or pr or pb:
        arr = cv2.copyMakeBorder(arr, pt, pb, pl, pr, cv2.BORDER_REPLICATE)
    return Image.fromarray(arr[y0 + pt:y1 + pt, x0 + pl:x1 + pl])


def extract_landmarks(crop, size=512):
    arr = np.array(crop.resize((size, size), Image.LANCZOS))
    with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                          refine_landmarks=True, min_detection_confidence=0.5) as m:
        res = m.process(arr)
    if not res.multi_face_landmarks:
        raise NoFace("face mesh failed")
    pts = np.array([(p.x * size, p.y * size) for p in res.multi_face_landmarks[0].landmark])
    control = np.zeros((size, size, 3), np.uint8)
    for conns, colour in MESH_GROUPS:
        for a, b in conns:
            cv2.line(control, tuple(map(int, pts[a])), tuple(map(int, pts[b])), colour, 2)
    overlay = arr.copy()
    for x, y in pts:
        cv2.circle(overlay, (int(x), int(y)), 1, (0, 255, 0), -1)
    return Image.fromarray(control), Image.fromarray(overlay), pts


# --------------------------------------------------------------------------
# Text conditioning: describe the photo with a captioner
# --------------------------------------------------------------------------
def clean_caption(text):
    text = text.replace("arafed", "").replace("  ", " ").strip(" ,.")
    return text


def build_prompt(caption, extra=""):
    parts = ["RAW photo", caption if caption else "of a person"]
    if extra:
        parts.append(extra)
    parts += ["looking at the camera", "natural skin texture, visible pores",
              "sharp focus, soft natural lighting"]
    return ", ".join(parts)


def caption_images(images):
    from transformers import BlipForConditionalGeneration, BlipProcessor
    name = "Salesforce/blip-image-captioning-base"
    dtype = torch.float16 if DEVICE == "cuda" else torch.float32
    proc = BlipProcessor.from_pretrained(name)
    model = BlipForConditionalGeneration.from_pretrained(name, torch_dtype=dtype).to(DEVICE)
    out = []
    for im in images:
        inputs = proc(images=im, return_tensors="pt").to(DEVICE, dtype)
        ids = model.generate(**inputs, max_new_tokens=30, num_beams=3)
        out.append(clean_caption(proc.decode(ids[0], skip_special_tokens=True)))
    del model, proc
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# Identity verifier (FaceNet, as in Park et al. [4])
# --------------------------------------------------------------------------
def load_facenet():
    from facenet_pytorch import InceptionResnetV1
    return InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def embed(model, img, face=None):
    face = face or detect_face(img)
    crop = pad_crop(img, face, 1.15).convert("RGB").resize((160, 160))
    x = (torch.from_numpy(np.array(crop)).float().permute(2, 0, 1) - 127.5) / 128.0
    with torch.no_grad():
        e = model(x[None].to(DEVICE))
    return torch.nn.functional.normalize(e, dim=1)[0].cpu()


# --------------------------------------------------------------------------
# Generation: landmarks + identity reference + caption, best-of-N by identity
# --------------------------------------------------------------------------
def load_generator(args):
    from diffusers import (ControlNetModel, StableDiffusionControlNetPipeline,
                           UniPCMultistepScheduler)
    controlnet = ControlNetModel.from_pretrained(
        "CrucibleAI/ControlNetMediaPipeFace", subfolder="diffusion_sd15",
        torch_dtype=torch.float16, variant="fp16")
    pipe = StableDiffusionControlNetPipeline.from_pretrained(
        args.base, controlnet=controlnet, torch_dtype=torch.float16, safety_checker=None)
    pipe.scheduler = UniPCMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.load_ip_adapter("h94/IP-Adapter", subfolder="models",
                         weight_name="ip-adapter-plus-face_sd15.bin")
    pipe.set_ip_adapter_scale(args.ip_scale)
    return pipe.to(DEVICE)


def generate_best(pipe, fn, sample, prompt, args):
    """Generate N candidates (different seeds); keep the one with highest identity cosine."""
    best, best_cos, all_cos = None, -2.0, []
    for i in range(args.candidates):
        img = pipe(
            prompt=prompt, negative_prompt=NEGATIVE,
            image=sample["control"], ip_adapter_image=sample["ref"],
            num_inference_steps=args.steps, guidance_scale=7.0,
            controlnet_conditioning_scale=1.0,
            generator=torch.Generator(DEVICE).manual_seed(args.seed + i),
        ).images[0]
        try:
            cos = float(torch.dot(sample["emb"], embed(fn, img)))
        except NoFace:
            all_cos.append(float("nan"))
            continue
        all_cos.append(cos)
        if cos > best_cos:
            best, best_cos = img, cos
    if best is None:
        raise NoFace("no candidate contained a detectable face")
    return best, best_cos, all_cos


# --------------------------------------------------------------------------
# Analysis view: identical processing for real and generated faces
# --------------------------------------------------------------------------
def analysis_view(img, jpeg_q):
    """Face crop -> 512x512 -> optional JPEG round-trip, so capture pipelines match."""
    view = pad_crop(img, detect_face(img), 1.3).convert("RGB").resize((512, 512), Image.LANCZOS)
    if jpeg_q > 0:
        buf = io.BytesIO()
        view.save(buf, format="JPEG", quality=jpeg_q)
        buf.seek(0)
        view = Image.open(buf).convert("RGB")
    return view


def load_detectors(ids):
    from transformers import pipeline
    return {i.split("/")[-1]: pipeline("image-classification", model=i,
                                       device=0 if DEVICE == "cuda" else -1) for i in ids}


def p_fake(clf, view):
    scores = clf(view, top_k=None)
    for s in scores:
        if s["label"].lower().strip() in FAKE_LABELS:
            return float(s["score"])
    raise RuntimeError(f"Unknown detector labels {[s['label'] for s in scores]}; "
                       f"add the 'fake' one to FAKE_LABELS")


def spectrum(view, n=256):
    """Radially averaged log power spectrum and its log-log slope (r from 4 to Nyquist)."""
    g = np.array(view.convert("L").resize((n, n), Image.LANCZOS), dtype=np.float32)
    g = (g - g.mean()) * np.outer(np.hanning(n), np.hanning(n))
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    yy, xx = np.indices(power.shape)
    r = np.hypot(yy - n // 2, xx - n // 2).astype(int)
    prof = (np.bincount(r.ravel(), power.ravel()) / np.bincount(r.ravel()))[: n // 2]
    logp = np.log10(prof + 1e-8)
    radius = np.arange(len(prof))
    sel = radius >= 4
    slope = float(np.polyfit(np.log10(radius[sel]), logp[sel], 1)[0])
    return logp, slope


# --------------------------------------------------------------------------
# Verdicts and summaries (pure functions)
# --------------------------------------------------------------------------
def verdict(cos, det_scores, thr):
    """det_scores: {name: (p_fake_real, p_fake_generated)}.
    A detector is 'sane' only if it calls the real photo real."""
    match = cos >= thr
    sane = [k for k, (pr, _) in det_scores.items() if pr < 0.5]
    if not sane:
        return match, sane, None
    return match, sane, bool(match and all(det_scores[k][1] < 0.5 for k in sane))


def summarise(rows, thr):
    n = len(rows)
    out = {"n": n, "id_match_rate": sum(r["cos"] >= thr for r in rows) / n, "detectors": {}}
    for name in rows[0]["det"]:
        pr = [r["det"][name][0] for r in rows]
        pg = [r["det"][name][1] for r in rows]
        out["detectors"][name] = {
            "false_positive_rate_on_real": sum(p >= 0.5 for p in pr) / n,
            "miss_rate_on_generated": sum(p < 0.5 for p in pg) / n,
        }
    decided = [r["success"] for r in rows if r["success"] is not None]
    out["attack_success_rate"] = (sum(decided) / len(decided)) if decided else None
    out["n_decided"] = len(decided)
    return out


def impostor_scores(embs_real, embs_gen):
    """Cosine between different people: original_i vs original_j and original_i vs generated_j."""
    scores = []
    for i, a in enumerate(embs_real):
        for j in range(len(embs_real)):
            if i != j:
                scores.append(float(torch.dot(a, embs_real[j])))
                scores.append(float(torch.dot(a, embs_gen[j])))
    return scores


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def make_report(path, s, args):
    fig = plt.figure(figsize=(15, 9.5))
    gs = fig.add_gridspec(2, 4, height_ratios=[1.3, 1])
    panels = [(s["wide"].resize((512, 512)), "1-2. Input + detected face"),
              (s["overlay"], "3. 478 MediaPipe landmarks"),
              (s["control"], "ControlNet condition"),
              (s["gen"], f"4. Regenerated (best of {args.candidates})")]
    for i, (im, title) in enumerate(panels):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im)
        ax.set_title(title, fontsize=11)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(s["spec_real"], label=f"original (slope {s['slope_real']:.2f})")
    ax.plot(s["spec_gen"], label=f"generated (slope {s['slope_gen']:.2f})")
    ax.set_xlabel("spatial frequency (radius)")
    ax.set_ylabel("log10 power")
    ax.set_title("C. Radial power spectrum", fontsize=11)
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1:])
    ax.axis("off")
    match, sane, success = s["verdict"]
    cands = ", ".join("nan" if c != c else f"{c:.2f}" for c in s["all_cos"])
    lines = [f"Prompt: {s['prompt'][:95]}", "",
             "A. Identity verifier (FaceNet)",
             f"   cosine = {s['cos']:.3f} (threshold {args.threshold})  ->  "
             f"{'MATCH' if match else 'NO MATCH'}   | candidates: {cands}", "",
             "B. Learned detectors   P(fake): real photo / generated"]
    for name, (pr, pg) in s["det"].items():
        note = "" if pr < 0.5 else "   [uninformative: calls the REAL photo fake]"
        lines.append(f"   {name[:34]:34s} {pr:.3f} / {pg:.3f}  "
                     f"{'evaded' if pg < 0.5 else 'caught'}{note}")
    lines.append("")
    if success is None:
        lines.append("Outcome: INCONCLUSIVE - no detector scored the real photo as real")
    elif success:
        lines.append("Outcome: passes verification AND evades every informative detector")
    else:
        lines.append("Outcome: stopped by at least one layer")
    lines += ["", "Single image: a demonstration, not a measurement."]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def make_summary_figure(path, rows, imp, thr):
    names = list(rows[0]["det"])
    fig, axes = plt.subplots(1, 1 + len(names), figsize=(5 * (1 + len(names)), 4.5))
    x = np.arange(len(rows))
    ax = axes[0]
    ax.bar(x, [r["cos"] for r in rows], color="tab:orange")
    ax.axhline(thr, color="k", ls="--", label=f"threshold {thr}")
    if imp:
        ax.axhline(max(imp), color="tab:red", ls=":", label=f"impostor max {max(imp):.2f}")
    ax.set_xticks(x, [r["name"][:8] for r in rows], rotation=45, ha="right")
    ax.set_ylabel("cosine similarity")
    ax.set_title("Identity: original vs generated")
    ax.legend(fontsize=8)
    for ax, name in zip(axes[1:], names):
        ax.scatter(x, [r["det"][name][0] for r in rows], label="real", color="tab:blue")
        ax.scatter(x, [r["det"][name][1] for r in rows], label="generated", color="tab:orange")
        ax.axhline(0.5, color="k", ls="--")
        ax.set_ylim(-0.05, 1.05)
        ax.set_xticks(x, [r["name"][:8] for r in rows], rotation=45, ha="right")
        ax.set_ylabel("P(fake)")
        ax.set_title(name[:32], fontsize=10)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------
def collect_images(p):
    p = Path(p)
    if p.is_dir():
        exts = {".jpg", ".jpeg", ".png", ".webp"}
        return sorted(f for f in p.iterdir() if f.suffix.lower() in exts)
    return [p]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image", help="an image file or a folder of images (batch mode)")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--candidates", type=int, default=4, help="best-of-N by identity cosine")
    ap.add_argument("--ip-scale", type=float, default=0.8)
    ap.add_argument("--threshold", type=float, default=0.5)
    ap.add_argument("--no-caption", action="store_true", help="skip BLIP, use generic prompt")
    ap.add_argument("--extra", default="", help="extra prompt text, e.g. 'wearing glasses'")
    ap.add_argument("--detectors", nargs="+", default=DEFAULT_DETECTORS)
    ap.add_argument("--jpeg-q", type=int, default=90,
                    help="JPEG quality applied to real AND generated before analysis (0 = off)")
    ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)
    fn = load_facenet()

    # ---- Stage A: load, detect, landmarks, identity embedding of the original
    samples = []
    for path in collect_images(args.image):
        try:
            img = Image.open(path).convert("RGB")
            face = detect_face(img)
            wide = pad_crop(img, face, 1.7)
            control, overlay, _ = extract_landmarks(wide)
            samples.append(dict(name=path.stem, img=img, face=face, wide=wide,
                                control=control, overlay=overlay,
                                ref=pad_crop(img, face, 1.3).resize((512, 512), Image.LANCZOS),
                                emb=embed(fn, img, face)))
            print(f"[A] {path.name}: face + landmarks ok")
        except NoFace as e:
            print(f"[A] {path.name}: skipped ({e})")
    if not samples:
        raise SystemExit("No usable images.")

    # ---- Stage B: describe each photo (text conditioning)
    if args.no_caption:
        captions = [""] * len(samples)
    else:
        print("[B] captioning photos")
        captions = caption_images([s["wide"] for s in samples])
    for s, c in zip(samples, captions):
        s["caption"], s["prompt"] = c, build_prompt(c, args.extra)
        print(f"    {s['name']}: {c!r}")

    # ---- Stage C: generate (model loaded once)
    print("[C] generating (first run downloads ~6 GB)")
    pipe = load_generator(args)
    kept = []
    for s in samples:
        try:
            s["gen"], s["cos"], s["all_cos"] = generate_best(pipe, fn, s, s["prompt"], args)
            kept.append(s)
            print(f"    {s['name']}: best cosine {s['cos']:.3f}")
        except NoFace as e:
            print(f"    {s['name']}: skipped ({e})")
    del pipe
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    samples = kept

    # ---- Stage D: detection + reports
    print("[D] running detectors")
    dets = load_detectors(args.detectors)
    rows = []
    for s in samples:
        d = out / s["name"]
        d.mkdir(exist_ok=True)
        s["gen"].save(d / "generated.png")
        s["control"].save(d / "control_landmarks.png")
        v_real, v_gen = analysis_view(s["img"], args.jpeg_q), analysis_view(s["gen"], args.jpeg_q)
        s["det"] = {n: (p_fake(c, v_real), p_fake(c, v_gen)) for n, c in dets.items()}
        s["spec_real"], s["slope_real"] = spectrum(v_real)
        s["spec_gen"], s["slope_gen"] = spectrum(v_gen)
        s["verdict"] = verdict(s["cos"], s["det"], args.threshold)
        s["success"] = s["verdict"][2]
        make_report(d / "report.png", s, args)
        rows.append(s)

    # ---- Summary
    imp = []
    if len(rows) >= 2:
        imp = impostor_scores([r["emb"] for r in rows],
                              [embed(fn, r["gen"]) for r in rows])
    with open(out / "summary.csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["image", "caption", "cosine", "slope_real", "slope_gen"] +
                   [f"{n}_{k}" for n in dets for k in ("p_real", "p_gen")] + ["attack_success"])
        for r in rows:
            w.writerow([r["name"], r["caption"], f"{r['cos']:.3f}",
                        f"{r['slope_real']:.2f}", f"{r['slope_gen']:.2f}"] +
                       [f"{r['det'][n][k]:.3f}" for n in dets for k in (0, 1)] +
                       [r["success"]])
    sm = summarise(rows, args.threshold)
    print("\n=== SUMMARY ===")
    print(f"images: {sm['n']}   identity match rate: {sm['id_match_rate']:.0%}")
    if imp:
        print(f"impostor cosine (different people): mean {np.mean(imp):.2f}, max {max(imp):.2f}"
              f"   <- a threshold below the max would let impostors in")
    for n, v in sm["detectors"].items():
        print(f"{n}: flags real photos as fake {v['false_positive_rate_on_real']:.0%}, "
              f"misses generated {v['miss_rate_on_generated']:.0%}")
    if sm["attack_success_rate"] is None:
        print("attack success: inconclusive (no informative detector)")
    else:
        print(f"attack success (match + evades all informative detectors): "
              f"{sm['attack_success_rate']:.0%} of {sm['n_decided']} decided images")
    make_summary_figure(out / "summary.png", rows, imp, args.threshold)
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()