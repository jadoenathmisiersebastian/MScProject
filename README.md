# Synthetic RGB-D Perception for Prosthetic Grasp Planning

A simulation-driven RGB-D perception pipeline for prosthetic grasp planning,
combining Unity-based domain-randomised data generation, automatic annotation,
YOLO instance segmentation, metric geometry estimation and real-world iPhone
RGB-D evaluation.

This repository contains the software developed for an MSc project
investigating whether models trained exclusively on synthetic RGB-D data can
detect and geometrically characterise everyday grasp targets in real scenes.
The final target classes are `drink_carton`, `bottle`, `food_box` and `glass`.

## Repository contents

| Component | Purpose |
| --- | --- |
| `unity/SyntheticDataGenerator` | Unity HDRP environment for domain-randomised RGB-D generation and automatic ground-truth export. |
| `python` | Dataset auditing, YOLO instance-segmentation training, calibrated depth reconstruction, residual geometry correction and evaluation. |
| `ios/RealRGBDCapture` | Swift/ARKit application for synchronised RGB, LiDAR depth, confidence and camera-calibration capture. |

Generated datasets, trained weights, predictions, reports, caches and
third-party Unity assets are intentionally excluded.

## Requirements

- Git.
- Unity Hub and Unity `6000.4.11f1`.
- Python 3.11 or newer.
- Xcode with Swift 5 support.
- A LiDAR-capable iPhone for physical RGB-D capture.

GPU acceleration is optional. The supplied Python configuration uses Apple
Metal (`mps`); set `device` to `0` for the first CUDA GPU or `cpu` when neither
backend is available.

## Clone the repository

```bash
git clone https://github.com/jadoenathmisiersebastian/MScProject.git
cd MScProject
```

## Unity synthetic-data generator

### 1. Open the project

Install Unity `6000.4.11f1` through Unity Hub, then add
`unity/SyntheticDataGenerator` as an existing project. On the first launch,
allow Unity Package Manager to restore the dependencies recorded in
`Packages/manifest.json` and wait for compilation to finish.

### 2. Restore the external assets

The following assets are required by the supplied scene but cannot be
redistributed in this repository. Open each link, add the asset to your own
Unity account, and use **Window > Package Manager > My Assets** to download and
import it into `unity/SyntheticDataGenerator`.

| Asset | Publisher | Used for | Expected imported folder |
| --- | --- | --- | --- |
| [Apartment Kit](https://assetstore.unity.com/packages/3d/environments/apartment-kit-124055) | Brick Project Studio | Indoor furniture, fixtures and bottle/glass meshes | `Assets/Brick Project Studio` |
| [Yughues Free Wooden Floor Materials](https://assetstore.unity.com/packages/2d/textures-materials/wood/yughues-free-wooden-floor-materials-13213) | Nobiax / Yughues | Floor appearance randomisation | `Assets/YughuesFreeFlooringMaterials` |
| [Marble Design Materials](https://assetstore.unity.com/packages/2d/textures-materials/tiles/marble-design-materials-284996) | Aquaset | Support-surface appearance randomisation | `Assets/Aquaset` |
| [PBC - Plates, Bowls and Cups](https://assetstore.unity.com/packages/3d/environments/urban/pbc-plates-bowls-and-cups-312159) | Kharnyx | Cup and glass target meshes | `Assets/Kharnyx` |
| [Plates, Bowls & Mugs Pack](https://assetstore.unity.com/packages/3d/props/interior/plates-bowls-mugs-pack-146682) | RS Robot Skeleton | Additional household-object meshes | `Assets/Mugs, Bowls and Plates` |

Next, restore the Unity Perception tutorial objects:

1. Open **Window > Package Manager**.
2. Select **Perception** (`com.unity.perception`).
3. Expand **Samples** in the package details.
4. Import **Tutorial Files**.
5. Confirm that Unity creates
   `Assets/Samples/Perception/1.0.0-preview.1/Tutorial Files`.

The imported packages preserve their Unity GUIDs, allowing the retained scene
and project-authored prefab variants to reconnect to their source meshes and
textures automatically.

> The external assets remain subject to their publishers' terms and are not
> covered by this repository's MIT Licence. Confirm that those terms permit
> your intended use before generating datasets or training models. See
> `THIRD_PARTY_NOTICES.md`.

### 3. Convert imported materials to HDRP

Some external packs were authored for Unity's Built-in Render Pipeline. If
imported objects appear pink or use incompatible shaders:

1. Open **Window > Rendering > Render Pipeline Converter**.
2. Select conversion from **Built-in** to **HDRP**.
3. Enable the material conversion steps and initialise the converters.
4. Convert the newly imported materials.
5. Allow Unity to reimport the affected assets.

The repository already contains the project's HDRP settings and its original
class-specific randomisation materials.

### 4. Generate a dataset

1. Open `Assets/SceneSetupNew.unity`.
2. Select the `VisionLabelExporter` object and set its **Dataset Root**.
3. Open **Edit > Project Settings > Perception** and select the same output
   location for SOLO captures.
4. Select `SingleObjectSupportSurfaceGenerator` and configure the requested
   sample count and random seed.
5. Enter Play mode.
6. Leave Unity running until the requested samples have been accepted and all
   RGB, depth and annotation files have been written.

The generator exports aligned RGB, metric depth, semantic/instance masks,
bounding boxes, class labels, camera parameters and object geometry. A suitable
repository-local destination is `python/datasets/vision_raw/<split-name>`;
dataset directories are ignored by Git.

Generate separate training, validation and test captures. The final experiment
used the following requested split:

```text
5,000 training images
  500 validation images
  500 test images
```

## Python perception pipeline

### 1. Create the environment

Run the following commands from the repository root:

```bash
cd python
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On Windows, activate the environment with:

```powershell
.venv\Scripts\Activate.ps1
```

### 2. Configure the dataset

Edit `config/final_vision_pipeline.json` and set the three entries under
`raw_splits` to the Unity training, validation and test directories. The
supplied example expects:

```text
datasets/vision_raw/C3D_5000_train_f
datasets/vision_raw/C3D_500_val_f
datasets/vision_raw/C3D_500_test_f
```

Also set `device` to `mps`, `0` or `cpu` as appropriate for the machine.
Class IDs are defined in `config/paths.yaml` and must remain consistent with
the Unity labels.

### 3. Train and evaluate

Launch the visual pipeline controller:

```bash
python main.py --pipeline-ui
```

Alternatively, run the complete resumable final pipeline directly:

```bash
python main.py --run-final-vision-pipeline
```

The pipeline audits the Unity exports, creates the YOLO segmentation dataset,
trains and evaluates YOLO, isolates predicted-mask depth, reconstructs metric
geometry and trains the residual geometry estimator. Its stage state allows an
interrupted run to resume; add `--pipeline-restart` to deliberately rerun every
stage.

Ultralytics downloads the requested pretrained YOLO checkpoint on first use.
Processed datasets, model runs and predictions are written beneath
`python/datasets`, `python/YOLO` and `python/predictions`; these paths are ignored
by Git.

Run the automated tests with:

```bash
python -m unittest discover -s tests
```

Use `python main.py --help` for the individual conversion, training,
prediction, geometry and figure-generation commands.

## iPhone RGB-D capture

The capture application requires a physical iPhone that supports ARKit scene
depth. It will not provide the required data in the iOS Simulator.

1. Open `ios/RealRGBDCapture/RealRGBDCapture.xcodeproj` in Xcode.
2. Select the `RealRGBDCapture` target and choose your Apple development team
   under **Signing & Capabilities**.
3. Replace `com.sebaluczko.RealRGBDCapture` with a bundle identifier available
   to your team.
4. Adjust the deployment target if required by your installed Xcode and device.
5. Connect a LiDAR-capable iPhone, select it as the run destination and build
   the application.
6. Wait until the interface reports normal ARKit tracking and available scene
   depth before saving a frame.

Each accepted capture is stored under the application's Documents directory in
`real_rgbd_raw/frame_XXXXXX`. A frame contains:

- `rgb.png`;
- `depth_raw.bin` and, when available, `depth_smoothed.bin` as little-endian
  row-major Float32 metres;
- `confidence.bin` when supplied by ARKit; and
- `metadata.json`, containing dimensions, camera intrinsics, camera transform
  and depth convention.

File sharing is enabled, so the `real_rgbd_raw` directory can be copied through
the Files app, Finder's device interface or Xcode's device-container tools.

Convert the exported captures into the canonical RGB-D layout from the
`python` directory:

```bash
python -m src.real_rgbd_adapter \
  --input-root /path/to/real_rgbd_raw \
  --output-root datasets/real_rgbd_processed \
  --depth-source smoothed \
  --minimum-confidence 1
```

The adapter defaults to a 180-degree orientation correction. Supply
`--rotation 0`, `90`, `180` or `270` if the capture orientation differs.

## Repository structure

```text
.
|-- unity/SyntheticDataGenerator/
|   |-- Assets/          # project-authored files; external packs are excluded
|   |-- Packages/
|   `-- ProjectSettings/
|-- python/
|   |-- config/
|   |-- src/
|   |-- tests/
|   |-- main.py
|   `-- requirements.txt
|-- ios/RealRGBDCapture/
|-- CITATION.cff
|-- LICENSE
`-- THIRD_PARTY_NOTICES.md
```

## Citation

Citation metadata is provided in `CITATION.cff` and is displayed through
GitHub's **Cite this repository** function.

## Licence

Original project source code and project-authored assets are released under the
MIT Licence. Excluded Unity Asset Store content and package samples retain their
respective licences and must be acquired independently.
