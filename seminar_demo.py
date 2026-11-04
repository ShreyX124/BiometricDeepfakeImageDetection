"""
Proof-of-concept: can a face regenerated from landmarks + one photo
(a) still match the original identity and (b) slip past a deepfake detector?

Pipeline
  1. Load user photo
  2. Detect face                      (MediaPipe Face Detection)
  3. Extract 478 landmarks            (MediaPipe Face Mesh)
  4. Regenerate the face              (Stable Diffusion 1.5 + ControlNet-MediaPipeFace
                                       for geometry + IP-Adapter-Face for identity)
  5. Check the generated image with three "defences":
       A. Identity verifier   FaceNet embedding, cosine similarity to the original
       B. Learned detector    ViT real-vs-fake classifier (Hugging Face)
       C. Frequency analysis  radially averaged power spectrum

Usage:  python seminar_demo.py my_face.jpg
"""
import argparse
import gc
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

# Colours follow the convention the ControlNet was trained on (approximate).
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


# --------------------------------------------------------------------------
# Steps 1-3: detect face, crop, landmarks
# --------------------------------------------------------------------------
def detect_face(img):
    """Return (cx, cy, side) of the most confident face, in pixels."""
    arr = np.array(img)
    h, w = arr.shape[:2]
    with mp_det.FaceDetection(model_selection=1, min_detection_confidence=0.5) as d:
        res = d.process(arr)
    if not res.detections:
        raise SystemExit("No face detected. Try a clearer, front-facing photo.")
    det = max(res.detections, key=lambda x: x.score[0])
    b = det.location_data.relative_bounding_box
    bw, bh = b.width * w, b.height * h
    cx, cy = b.xmin * w + bw / 2, b.ymin * h + bh / 2
    return cx, cy, max(bw, bh)


def square_crop(img, face, scale):
    cx, cy, side = face
    half = side * scale / 2
    return img.crop((int(cx - half), int(cy - half), int(cx + half), int(cy + half)))


def extract_landmarks(crop, size=512):
    """Run Face Mesh on the crop. Returns (control_image, overlay_image, points[478,2])."""
    arr = np.array(crop.resize((size, size), Image.LANCZOS))
    with mp_mesh.FaceMesh(static_image_mode=True, max_num_faces=1,
                          refine_landmarks=True, min_detection_confidence=0.5) as m:
        res = m.process(arr)
    if not res.multi_face_landmarks:
        raise SystemExit("Face Mesh failed on the crop.")
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
# Step 4: regenerate (landmarks -> geometry, IP-Adapter -> identity)
# --------------------------------------------------------------------------
def generate(control_img, face_ref, args):
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
    pipe.to(DEVICE)

    out = pipe(
        prompt=args.prompt,
        negative_prompt="cartoon, painting, blurry, deformed, extra eyes, lowres, watermark",
        image=control_img,
        ip_adapter_image=face_ref,
        num_inference_steps=args.steps,
        guidance_scale=7.0,
        controlnet_conditioning_scale=1.0,
        generator=torch.Generator(DEVICE).manual_seed(args.seed),
    ).images[0]

    del pipe, controlnet
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()
    return out


# --------------------------------------------------------------------------
# Step 5A: identity verification (FaceNet, as in Park et al. [4])
# --------------------------------------------------------------------------
def load_facenet():
    from facenet_pytorch import InceptionResnetV1
    return InceptionResnetV1(pretrained="vggface2").eval().to(DEVICE)


def embed(model, img):
    crop = square_crop(img, detect_face(img), 1.15).convert("RGB").resize((160, 160))
    x = (torch.from_numpy(np.array(crop)).float().permute(2, 0, 1) - 127.5) / 128.0
    with torch.no_grad():
        e = model(x[None].to(DEVICE))
    return torch.nn.functional.normalize(e, dim=1)[0].cpu()


# --------------------------------------------------------------------------
# Step 5B: learned deepfake detector (ViT)
# --------------------------------------------------------------------------
def load_detector(model_id):
    from transformers import pipeline
    return pipeline("image-classification", model=model_id,
                    device=0 if DEVICE == "cuda" else -1)


def p_fake(clf, img):
    face = square_crop(img, detect_face(img), 1.3).convert("RGB")
    scores = clf(face, top_k=None)
    return next(s["score"] for s in scores if "fake" in s["label"].lower())


# --------------------------------------------------------------------------
# Step 5C: frequency analysis (cf. GAN fingerprints [1], spectral peaks [9])
# --------------------------------------------------------------------------
def spectrum(img, n=256):
    face = square_crop(img, detect_face(img), 1.3).convert("L").resize((n, n), Image.LANCZOS)
    g = np.array(face, dtype=np.float32)
    g = (g - g.mean()) * np.outer(np.hanning(n), np.hanning(n))
    power = np.abs(np.fft.fftshift(np.fft.fft2(g))) ** 2
    yy, xx = np.indices(power.shape)
    r = np.hypot(yy - n // 2, xx - n // 2).astype(int)
    prof = np.bincount(r.ravel(), power.ravel()) / np.bincount(r.ravel())
    prof = prof[: n // 2]
    hf_ratio = prof[n // 4:].sum() / prof.sum()  # share of energy in upper half of frequencies
    return np.log10(prof + 1e-8), float(hf_ratio)


# --------------------------------------------------------------------------
# Report figure for the slides
# --------------------------------------------------------------------------
def make_report(path, crop, overlay, gen, res, args):
    fig = plt.figure(figsize=(14, 8.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[1.3, 1])
    for i, (im, title) in enumerate([(crop, "1-2. Input + detected face"),
                                     (overlay, "3. 478 MediaPipe landmarks"),
                                     (gen, "4. Regenerated face")]):
        ax = fig.add_subplot(gs[0, i])
        ax.imshow(im)
        ax.set_title(title, fontsize=12)
        ax.axis("off")

    ax = fig.add_subplot(gs[1, 0])
    ax.plot(res["spec_real"], label=f"original (HF {res['hf_real']:.3f})")
    ax.plot(res["spec_fake"], label=f"generated (HF {res['hf_fake']:.3f})")
    ax.set_xlabel("spatial frequency (radius)")
    ax.set_ylabel("log10 power")
    ax.set_title("5C. Radial power spectrum")
    ax.legend(fontsize=8)

    ax = fig.add_subplot(gs[1, 1:])
    ax.axis("off")
    match = res["cos"] >= args.threshold
    evades = res["pf_fake"] < 0.5
    lines = [
        "5A. Identity verifier (FaceNet)",
        f"    cosine similarity original vs generated = {res['cos']:.3f}"
        f"   (threshold {args.threshold})  ->  {'MATCH' if match else 'NO MATCH'}",
        "",
        "5B. Learned deepfake detector (ViT)",
        f"    P(fake) original  = {res['pf_real']:.3f}",
        f"    P(fake) generated = {res['pf_fake']:.3f}  ->  "
        f"{'classified REAL (evaded)' if evades else 'flagged FAKE (caught)'}",
        "",
        "Outcome: " + ("generated face passes verification AND evades the detector"
                       if match and evades else
                       "attack stopped by at least one layer" if not (match and evades) else ""),
        "",
        "Single image, uncalibrated threshold: a demonstration, not a measurement.",
    ]
    ax.text(0, 1, "\n".join(lines), va="top", family="monospace", fontsize=10.5)
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("image")
    ap.add_argument("--out", default="results")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--steps", type=int, default=30)
    ap.add_argument("--ip-scale", type=float, default=0.8,
                    help="identity strength (higher = closer likeness, less prompt control)")
    ap.add_argument("--threshold", type=float, default=0.5,
                    help="cosine threshold for 'same person' (illustrative, not calibrated)")
    ap.add_argument("--base", default="stable-diffusion-v1-5/stable-diffusion-v1-5")
    ap.add_argument("--detector", default="dima806/deepfake_vs_real_image_detection")
    ap.add_argument("--prompt", default="photo of a person, natural lighting, sharp focus, "
                                        "realistic skin texture, high detail")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(exist_ok=True)

    print("[1/5] loading image")
    img = Image.open(args.image).convert("RGB")

    print("[2/5] detecting face")
    face = detect_face(img)
    crop = square_crop(img, face, 1.7)  # generous margin (hair, ears) for generation

    print("[3/5] extracting landmarks")
    control, overlay, _ = extract_landmarks(crop)
    control.save(out / "control_landmarks.png")

    print("[4/5] generating (first run downloads ~6 GB of weights)")
    ref = square_crop(img, face, 1.3).resize((512, 512), Image.LANCZOS)
    gen = generate(control, ref, args)
    gen.save(out / "generated.png")

    print("[5/5] running verification + detection")
    fn = load_facenet()
    cos = float(torch.dot(embed(fn, img), embed(fn, gen)))
    clf = load_detector(args.detector)
    spec_real, hf_real = spectrum(img)
    spec_fake, hf_fake = spectrum(gen)
    res = dict(cos=cos, pf_real=p_fake(clf, img), pf_fake=p_fake(clf, gen),
               spec_real=spec_real, spec_fake=spec_fake, hf_real=hf_real, hf_fake=hf_fake)

    make_report(out / "report.png", crop.resize((512, 512)), overlay, gen, res, args)
    print(f"cosine={cos:.3f}  P(fake) real={res['pf_real']:.3f} gen={res['pf_fake']:.3f}")
    print(f"saved to {out.resolve()}")


if __name__ == "__main__":
    main()
