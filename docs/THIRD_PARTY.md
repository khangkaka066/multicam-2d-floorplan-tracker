# Third-party sources

Downloaded data and model code/weights are not bundled with this repository. Showcase images contain EPFL research imagery; the project MIT license does not relicense that material.

| Component | Source and attribution | Local use |
|---|---|---|
| EPFL Laboratory and Passageway | [CVLab EPFL dataset](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/). François Fleuret, Jérôme Berclaz, Richard Lengagne and Pascal Fua, *Multicamera People Tracking with a Probabilistic Occupancy Map*, IEEE TPAMI, 2008. Dataset terms remain with CVLab EPFL. | Sample/verification clips, published ground-plane calibration |
| Depth Anything V2 metric indoor Small | [Official code](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth), [Hypersim Small checkpoint](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small) | Metric depth; vendor commit and checkpoint SHA recorded by downloader |
| SegFormer B2 ADE20K | [NVIDIA model card](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512) | Floor/wall labels and clutter exclusion; revision pinned by downloader |
| Ultralytics YOLO11n-pose | [Ultralytics repository](https://github.com/ultralytics/ultralytics), [official assets](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt) | Person pose; Ultralytics license/weight terms apply independently |
| ResNet18 | [Torchvision](https://github.com/pytorch/vision), [official checkpoint](https://download.pytorch.org/models/resnet18-f37072fd.pth) | Normalized appearance descriptors |

Setup records source URLs and checkpoint SHA-256 values in local `models/structural_sources.json` and `models/person_model_sources.json`. The selected structural code commit and segmentation revision are pinned in `scripts/download_structural_models.py`.

Before using downloaded components for a new purpose, consult their linked source/model terms. These acknowledgments describe provenance; they do not extend any third-party usage rights.
