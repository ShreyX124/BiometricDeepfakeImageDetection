<<<<<<< HEAD
deactivate
winget install Python.Python.3.12        # skip if you already have 3.12
cd "D:\Mtech Notes\Semester 1\Seminar\Implementation"
Remove-Item -Recurse -Force venv
py -3.12 -m venv venv
.\venv\Scripts\Activate.ps1
python --version                          # should say 3.12.x

pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
pip install mediapipe==0.10.14 opencv-python matplotlib pillow numpy
pip install diffusers transformers accelerate safetensors
pip install facenet-pytorch --no-deps
pip install requests tqdm


# Face regeneration vs. verification and detection: seminar demo

## Install (Windows or Linux, Python 3.10 to 3.12)

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate

# RTX 50-series (Blackwell) needs a CUDA 12.8 build of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install mediapipe==0.10.14 opencv-python matplotlib pillow numpy
pip install diffusers transformers accelerate safetensors
pip install facenet-pytorch --no-deps      # --no-deps stops it downgrading your torch
pip install requests tqdm
```

Check the GPU is seen: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

## Run

```bash
python seminar_demo.py my_face.jpg
```

The first run downloads about 6 GB of weights, so do it before the seminar and then use it offline.
Output goes to `results/`: `report.png` (the slide), `generated.png`, `control_landmarks.png`.

## Useful flags

| Flag | Effect |
|---|---|
| `--ip-scale 0.8` | Identity strength. Raise it for a closer likeness, lower it for more variety. |
| `--threshold 0.5` | Cosine cut-off for "same person". It is illustrative, not calibrated. |
| `--seed N` | Try several seeds and keep the most convincing result. |
| `--detector <hf-id>` | Swap in any Hugging Face image-classification deepfake detector. |

## Troubleshooting

- **Base model 404**: pass another SD 1.5 mirror with `--base`.
- **CUDA out of memory**: close other GPU apps. As a last resort add `pipe.enable_model_cpu_offload()` instead of `pipe.to(DEVICE)` in `generate()`.
- **Face does not follow the landmarks**: the ControlNet expects its own colour convention for the mesh drawing. Adjust `MESH_GROUPS`.
- **Detector label error**: print `clf(img)` once and check the label name for "fake".

## Mapping to the literature review

| Demo component | Review section |
|---|---|
| MediaPipe landmarks and ControlNet, so geometry is fixed and identity comes from a reference | Sec. 3: Deng et al. [7], geometry-aware generation |
| FaceNet verifier | Sec. 1: Park et al. [4] attack FaceNet pipelines |
| ViT detector, trained on older data and likely to fail on diffusion output | Sec. 1 and 4: generalisation gap [1], [2], [8] |
| Power spectrum plot | Sec. 1 and 5: GAN fingerprints absent in diffusion [1], spectral cues [9] |
| Pass verification AND evade detector | The gap in Sec. 6: untested whether identity-preserving synthetic faces pass verification and detection together |

## Limits to state in the seminar

- One image, one run: an anecdote, not a measurement. A real study needs many subjects, APCER/BPCER/EER and several detectors.
- The FaceNet threshold is not calibrated, and this is not Aadhaar's matcher.
- There is no liveness test (blink, depth, challenge-response). Only a static image check is done.
- Use your own face or a consenting volunteer.
=======
# BiometricDeepfakeImageDetection
Generation and detection of images for Biometric Deepfake Detection
>>>>>>> a8181d7c986360d5327d9e81dda260ebf93a1044


# Face regeneration vs. verification and detection: seminar demo

## Install (Windows or Linux, Python 3.10 to 3.12)

```bash
python -m venv venv && source venv/bin/activate      # Windows: venv\Scripts\activate

# RTX 50-series (Blackwell) needs a CUDA 12.8 build of PyTorch
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128

pip install mediapipe==0.10.14 opencv-python matplotlib pillow numpy
pip install diffusers transformers accelerate safetensors
pip install facenet-pytorch --no-deps      # --no-deps stops it downgrading your torch
pip install requests tqdm
```

Check the GPU is seen: `python -c "import torch; print(torch.cuda.get_device_name(0))"`

## Run

```bash
python seminar_demo.py my_face.jpg            # one image
python seminar_demo.py faces_folder/          # batch mode: 5-10 consenting faces
```

Per image, `results/<name>/` gets `report.png` (the slide), `generated.png`, `control_landmarks.png`.
Batch mode also writes `results/summary.csv` and `results/summary.png`, and prints a summary table.
Extra downloads this version needs: BLIP captioner (about 1 GB) and the Organika detector (about 350 MB).

## What the generator is given

| Input | Source | Gives the model |
|---|---|---|
| Landmark drawing | MediaPipe Face Mesh, via ControlNet | pose, expression, face geometry |
| Face crop | IP-Adapter-Face | identity and appearance |
| Text prompt | BLIP caption of the photo (+ `--extra`) | glasses, beard, clothes, background |
| N seeds | best-of-N by FaceNet cosine | picks the best likeness |

## Flags

| Flag | Effect |
|---|---|
| `--candidates 4` | Best-of-N seeds. Higher is a better likeness but slower (about 6 to 10 s each). |
| `--extra "wearing thick black glasses"` | Hand-add detail the captioner missed. |
| `--no-caption` | Generic prompt, for an ablation: show what the caption adds. |
| `--ip-scale 0.8` | Identity strength. |
| `--threshold 0.5` | Cosine cut-off. Compare with the impostor stats printed in batch mode. |
| `--detectors id1 id2 ...` | Any Hugging Face image-classification detector. |
| `--jpeg-q 90` | JPEG quality applied to real and generated before analysis (0 = off). |

## Troubleshooting

- **Base model 404**: pass another SD 1.5 mirror with `--base`.
- **CUDA out of memory**: use `pipe.enable_model_cpu_offload()` instead of `.to(DEVICE)` in `load_generator()`.
- **Face does not follow the landmarks**: adjust `MESH_GROUPS` colours.
- **"Unknown detector labels"**: add that detector's "fake" label to `FAKE_LABELS`.
- **A detector shows "[uninformative]"**: it flagged your real photo as fake. Its verdict on the generated image is ignored.

## Mapping to the literature review

| Demo component | Review section |
|---|---|
| MediaPipe landmarks and ControlNet, so geometry is fixed and identity comes from a reference | Sec. 3: Deng et al. [7], geometry-aware generation |
| FaceNet verifier | Sec. 1: Park et al. [4] attack FaceNet pipelines |
| Two detectors: an older face-deepfake ViT and a diffusion-era one; compare their verdicts | Sec. 1 and 4: generalisation gap [1], [2], [8] |
| Power spectrum and slope, on identically processed real and generated crops | Sec. 1 and 5: GAN fingerprints absent in diffusion [1], spectral cues [9] |
| Pass verification AND evade detector | The gap in Sec. 6: untested whether identity-preserving synthetic faces pass verification and detection together |

## Limits to state in the seminar

- Batch mode over 5 to 10 faces is still a small pilot, not a benchmark. A real study needs many subjects, APCER/BPCER/EER and more detectors.
- The FaceNet threshold is not calibrated, and this is not Aadhaar's matcher.
- There is no liveness test (blink, depth, challenge-response). Only a static image check is done.
- Use your own face or a consenting volunteer.