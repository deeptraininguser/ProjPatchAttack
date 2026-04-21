# Physical Adversarial Patch Attack – Complete Pipeline

[![License: CC BY-NC 4.0](https://licensebuttons.net/l/by-nc/4.0/88x31.png)](https://creativecommons.org/licenses/by-nc/4.0/)

This project implements a **projector-based physical adversarial patch attack** pipeline.  
A latent-space patch is optimised to fool image classifiers when projected onto a real-world scene, then evaluated via live camera inference.

![Projector-based adversarial attack on everyday objects across viewpoints](docs/assets/teaser.jpg)
> **Our method in action.** A lightweight projection framework enables robust adversarial attacks through rapid calibration **(15 seconds, 33 patterns)** and real-time adaptation under viewpoint and scene changes.

### Overview Video
<video src="docs/assets/demo_video.mp4" controls="controls" muted="muted" playsinline="playsinline"></video>

## Abstract
Projector-based physical adversarial attacks offer a contact-free and reconfigurable way to manipulate camera inputs, but prior work often either relies on lengthy calibration or overstates real-world generality. We present a projector-camera attack pipeline for indoor settings with local planar, mostly diffuse target regions. Our method uses 33 projected calibration patterns and less than 15 seconds of capture to fit a differentiable per-pixel photometric model and an ArUco-based geometric mapping, then optimizes latent-space patches across viewpoints, models, and sensor-like augmentations.
Relative to the learned PCNet surrogate used by CAPAA, our results show lower reconstruction error (0.077±0.052 vs. 0.120±0.062 RMSE), 33 rather than 500 calibration patterns, and under 1 second rather than roughly 1800 seconds of simulator fitting on the same machine. In physical evaluation across five objects and five novel viewpoints, combined attack success reaches 86.8% on seen models, 79.7% on a DINOv2-based classifier, and 71.7% on unseen models; ablations show that removing simulation or multi-view training sharply degrades performance.
We do not claim outdoor viability or human imperceptibility. Instead, we frame the system as a credible indoor white-box threat, state the assumptions explicitly, analyze failure modes, and discuss defenses such as projection-aware training, temporal light-signature checks, and multi-view consistency tests.

## Pipeline
![Three-stage attack pipeline overview](docs/assets/pipeline3.png)

**Attack Pipeline.** Left: A patch location is selected and an ArUco marker is projected while capturing a short multi-view video, followed by projecting 33 flat color patterns for photometric calibration. Middle: A latent vector is optimized using a downstream classifier under attack by simulating how the decoded patch would appear in camera-space under viewpoint changes and the full project-and-capture process. Right: The optimized patch is projected onto the target, causing the downstream classifier to fail; the attack is robust to viewpoint and object changes.

### Simulator Fidelity
![Qualitative simulation comparison](docs/assets/sim_supp_small.jpg)

**Qualitative Simulation Results.** Various patterns are Projected on a surface and Captured. The process is simulated using PCNet (CAPAA) and our affine model.

### Physical Attack Success
![Bar charts comparing attack success rates](docs/assets/all_methods_comparison.png)

**Quantitative Comparisons.** Attack success rates vs. CAPAA and a random Gaussian baseline over 5 scenes and 5 novel viewpoints (20 captures per viewpoint). Seen: classifiers used during optimization; DINO: DINOv2-ViT-L14; Unseen: 5 classifiers not used during optimization.

## Dynamic Object Attack
![Dynamic attack on a moving remote-controlled vehicle](docs/assets/dynamic.jpg)

**Dynamic Scenario.** The optimized patch is reprojected onto a moving remote-controlled vehicle using real-time tracking (YOLOv11 + SIFT + RANSAC). The attack maintains alignment under slow motion; effectiveness degrades under rapid movement.

---

## Repository layout

| File / module | Purpose |
|---|---|
| `config.yaml` | **All tuneable parameters** – classifier choice, training hyper-parameters, paths, scheduler, etc. |
| `.env` | Comet ML secrets (`COMET_API_KEY`, `COMET_PROJECT_NAME`, `COMET_WORKSPACE`). Ignored by git. |
| `complete_attack.ipynb` | **Main notebook** – runs the full pipeline end-to-end (Acquisition → Training → Inference). |
| `attack_config.py` | Loads `config.yaml` + `.env`, resolves `on_remote`, exposes helpers. |
| `classifier_loader.py` | Dynamically imports the classifier module specified in config. |
| `experiment_tracking.py` | Optional Comet ML wrapper; becomes a no-op when `comet.enabled: false`. |
| `data_preparation.py` | Loads captured frames, detects ArUco markers, computes homographies, builds DataLoaders. |
| `vae_utils.py` | Stable Diffusion VAE helpers (encode / decode latents). |
| `augmentation.py` | Augmentation transforms (jitter, blur, photometric augmentor) and the `warp()` helper. |
| `training.py` | The main adversarial-patch training loop. |
| `evaluation.py` | Post-training CSV export (ablation metrics) and top-patch saving. |
| `visualization.py` | Matplotlib summary plots for the best patches. |
| `consts.py` | Low-level constants (ArUco code, border size, latent dimensions, forbidden classes). |
| `classfier_mobilenet.py` | MobileNet V3 Small classifier. |
| `classfier_ensemble_v2.py` | Ensemble classifier (ConvNeXt + EfficientNet + MobileNet + Swin). |
| `classfier.py` | Inception V3 classifier. |
| `classfier_test.py` | EfficientNet B0 test classifier. |
| `capture_utils_v2.py` | Camera / projector capture utilities. |
| `tracking_utils.py` | ArUco-based tracking and projection helper. |
| `aruco_pose.py` | Camera-pose estimation from ArUco markers. |
| `classfier_ensemble.py` | Ensemble classifier (earlier variant). |
| `interp_comp_torch.py` | Interpolation / comparison utilities (PyTorch). |
| `complete_attack_v2.ipynb` | Alternate version of the full pipeline notebook. |
| `single_debug_v2_untargeted.ipynb` | Single-patch debug / untargeted attack notebook. |

---

## Prerequisites

```
Python ≥ 3.10
CUDA-capable GPU (tested with CUDA 11.8 / 12.x)
Physical setup: camera + projector + printed ArUco marker
```

### Python dependencies

```bash
pip install -r requirements.txt
```

See [`requirements.txt`](./requirements.txt) for the full list. For optional experiment tracking, uncomment `comet-ml` in that file or install it separately:

```bash
pip install comet_ml
```

---

## Quick-start: running the full pipeline

### 1. Configure

Edit **`config.yaml`** to match your setup:

| Key | What to set |
|---|---|
| `on_remote` | `null` (auto-detect), `true`, or `false` |
| `remote_project_path` | Path to `chdir` into on the remote machine |
| `classifier_train` | Module name, e.g. `"classfier_mobilenet"` or `"classfier_ensemble_v2"` |
| `classifier_dev` | Module used during per-patch validation |
| `classifier_test` | Module used for final evaluation |
| `training.num_epochs` | Number of training epochs |
| `training.target_classes` | List of ImageNet class indices (empty = untargeted) |
| `comet.enabled` | `true` to log to Comet ML, `false` to skip |

If using Comet ML, fill in **`.env`**:

```
COMET_API_KEY=<your-key>
COMET_PROJECT_NAME=physicaladvproj
COMET_WORKSPACE=<your-workspace>
```

### 2. Open the notebook

```
code complete_attack.ipynb
```

### 3. Run the cells in order

The notebook has four clearly marked sections:

#### Section 0 – Setup & Configuration
- Loads `config.yaml` and `.env`.
- Initialises the (optional) Comet ML tracker.

#### Section 1 – Acquisition (Camera + Calibration)
- Opens the camera via `CaptureSystem`.
- Displays the projector drawer and detects ArUco corners interactively.
- Runs photometric calibration and saves the state to `capture_system_state.pkl`.

> **Physical setup required**: camera must see the projected ArUco marker.

#### Section 2 – Training
- Loads the classifier(s) specified in config.
- Loads captured multi-view frames, filters by ArUco + class, computes homographies.
- Builds train / val / test DataLoaders.
- Loads the Stable Diffusion VAE and the photometric augmentor.
- Runs the adversarial latent-space optimisation loop.
- Saves ablation CSVs and top-performing patch images to `./results/`.

> On a remote GPU machine, set `on_remote: true` (or leave `null` for auto-detect).

#### Section 3 – Inference
- Picks the best patch from the training results.
- Projects it via `system.plot_on_screen()`.
- Captures live frames, classifies each, and overlays predictions.
- Saves inference captures to `infer_caps_dir`.

#### Section 4 – Cleanup
- Ends the Comet experiment and closes OpenCV windows.

---

## License

This project is licensed under the **Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0)** license.

See the full license text in [LICENSE](./LICENSE) or visit:
https://creativecommons.org/licenses/by-nc/4.0/

## Running on a remote machine

1. Set `on_remote: true` in `config.yaml` (or leave `null` and let the code auto-detect from `cwd`).
2. Set `remote_project_path` to the project directory on the remote host.
3. The acquisition step (Section 1) should ideally be run locally where the camera is attached.  
   After acquisition, copy `capture_system_state.pkl` and the `captures_frames_multiview/` directory to the remote machine.
4. Run Section 2 (Training) on the remote machine.
5. Copy the resulting patches back and run Section 3 (Inference) locally.

---

## Choosing a classifier

Change `classifier_train` in `config.yaml` to any of:

| Value | Model |
|---|---|
| `classfier_mobilenet` | MobileNet V3 Small |
| `classfier_ensemble_v2` | Ensemble (ConvNeXt + EfficientNet B0 + MobileNet V3 Large + Swin-B) |
| `classfier_ensemble` | Ensemble (earlier variant) |
| `classfier` | Inception V3 |
| `classfier_test` | EfficientNet B0 |

When using the ensemble, you can also tweak per-model weights under `ensemble_weights` in the config.

---

## Disabling Comet ML

Set `comet.enabled: false` in `config.yaml`.  The `ExperimentTracker` will silently absorb all logging calls so no code changes are needed.

---

## Output files

After a successful run you will find:

```
./results/
├── best_latent_final_<model>.pt          # Best latent tensor
├── <model>_<size>_<timestamp>_top_patches/
│   ├── 0_0.85.png                        # Individual patch PNGs
│   └── …
├── ablation_tracking/
│   ├── …_epoch_metrics.csv
│   ├── …_patch_metrics.csv
│   └── …_summary.csv
└── infer_caps/
    └── <patch_name>/<timestamp>/
        ├── caps.pkl
        ├── caps_with_text.pkl
        ├── results.pkl
        └── raw_predictions.pkl
```
