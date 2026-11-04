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
