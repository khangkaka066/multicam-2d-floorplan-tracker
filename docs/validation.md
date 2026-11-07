For current generic room discovery, automatic rectification and topology, see [multi-room support](multi_room.md). The EPFL person-fusion measurements below remain the reference baseline.

# Multi-camera architectural mapping and person fusion

The pipeline reconstructs two real EPFL scenes and projects their people onto a clean map. It now displays **one tracked, fused marker per person**, not one marker per camera observation.

The reference verification snapshot uses **Laboratory frame 578** and **Passageway frame 328**. It displays **2 people** and **4 people**, with globally unique display IDs. The headcounts come from the observations and associations; these values are not supplied to the inference algorithm. The reference counts are used only as artifact-validation expectations.

The recordings are independent. Their relative building placement and shared doorway are still unknown; the combined canvas remains an explicitly unregistered atlas.

## Run the person pipeline

```bash
source .venv/bin/activate
pip install -r requirements.txt
python scripts/download_person_models.py

python main.py --mode project --room laboratory --output-dir outputs/laboratory --frames 100
python main.py --mode project --data-dir data/passageway --output-dir outputs/passageway --frames 100
python main.py --mode build-global

python scripts/render_fusion_debug.py
python -m unittest discover -s tests -v
python scripts/validate_outputs.py
```

Existing images, local maps and original homographies are reused. Pose acquisition is cached separately in each scene’s `pose_observations.json`, with source-frame offsets, pose-weight hash and inference settings. Changing acquisition settings invalidates that cache. Fitting/fusion/rendering can be rerun without detection.

Pose runs use CPU with two Torch threads. A trial on Apple MPS emitted Ultralytics’ known pose-backend warning, so it was not used for the final person results.

`--imgsz` overrides the high-resolution pass. `--show-raw` adds faint observations and tethers to a local scene’s rendering. `scripts/render_fusion_debug.py` creates the equivalent combined diagnostic without overwriting the normal verification image.

## Detection and appearance

`src/person_locator.py` runs YOLO11n-pose with `conf=0.25`:

- Laboratory uses a deduplicated **384 + 960** ensemble. Large cropped subjects disappear at high resolution alone; the near-person pass retains them while the higher-resolution pass can recover smaller subjects.
- Passageway uses **1280**, which recovers the distant pedestrians missed in cam2/cam3 at 640.
- Overlapping scale proposals are deduplicated before association. No person-count targets are used.
- Raw detections retain boxes, keypoints, ankle confidence, detector confidence, crop-border flags and source resolution. Pose-guided lower-body compensation remains available, but extrapolated points and censored ankles do not calibrate homographies.

`src/appearance.py` extracts normalized ResNet18 crop features. These are pretrained image features, **not a dedicated trained ReID model**. Geometry and rank order remain necessary where clothing looks similar. Model files and their hashes are recorded by `scripts/download_person_models.py`.

At the selected snapshot, all Laboratory cameras detect two people. Passageway cam0/cam2/cam3 detect four; cam1 has three usable detections. Fusion recovers the four identities without requiring every person to be visible in every camera.

## Joint homography refinement

`src/bundle_adjustment.py` replaces translation-only correction with a joint **eight-parameter projective correction per camera**. Hungarian matching combines spatial proximity, appearance and ordering. Reliable ankle observations are selected before matching so extrapolated hip/shoulder points cannot steal calibration assignments.

The nonlinear objective includes weighted pairwise BEV errors, uncertainty-derived landmark priors, room-boundary penalties and control/gauge priors. Low person-confidence and laterally truncated detections are excluded from calibration training, while their validation residuals remain reported. A control-displacement penalty limits unsupported deformation outside the walking paths. The room polygons and architectural camera pose estimates remain fixed; these matrices correct footpoint projection rather than certify surveyed camera extrinsics.

Calibration frames are sampled throughout each offline clip so late-visible pedestrians are represented. Frames with `index % 5 == 3` are withheld; **index 78 is excluded from all fit terms, including boundary loss**. This is interleaved validation within the same recording, not a future-time generalization test or ground-truth location benchmark.

`transforms.json` retains the original mappings. `transforms_refined.json` stores refined mappings, inverse/canvas matrices, `H_image_to_map_before_bundle` and `footpoint_bundle_correction`. Atlas copies transform these diagnostics consistently into the display frame.

**The all-raw 0.35 m target is not fully achieved.** Far-field pixel errors and some low-confidence/truncated observations still produce larger residuals. `projection_optimization.json` reports their unfiltered validation statistics and `target_met_for_all_heldout_pairs`; these cases are not silently declared converged. Absolute metric accuracy is also unverified. The strict 0.35 m bound below applies to accepted fusion members.

## Fusion, weighting and IDs

`src/fusion.py` permits at most one observation per camera in a person group. It uses ankle confidence and inverse squared 3D distance to the camera:

```text
weight ∝ ankle_confidence / camera_distance²
```

Projection-Jacobian conditioning, detector confidence and lateral truncation further reduce uncertain contributions. Weak hip/shoulder extrapolations receive lower weights. Complete-link selection requires **every pair of accepted camera points** to be within 0.35 m; this prevents both chaining distinct nearby people and letting a grazing-angle outlier pull the fused coordinate away. The fused position is the weighted mean of the accepted observations.

Tracks receive persistent IDs. Confirmed tracks may predict through at most five missing frames (0.20 s), only inside the room; those records are explicitly marked `predicted_short_occlusion` and rendered with `ID n*`. Predictions expire rather than imposing a constant count. All six people in the requested snapshot are **observed multi-camera fusions**, not predictions.

The renderer shows one bold marker per fused identity and a headcount banner per room. Camera panels identify their associated IDs. Raw points are hidden by default and remain available in the diagnostic image/JSON. Counts are computed from the actual active identities, not from the number of raw dots.

## Outputs

| File under `outputs/` | Meaning |
|---|---|
| `verification_bev.png` | Reference frames, eight camera panels, two Laboratory markers and four Passageway markers |
| `fused_bev.png` | Map-only version with IDs and room headcounts |
| `verification_projection.mp4` | Full 100-frame combined verification clip |
| `verification_bev_debug.png`, `fused_bev_debug.png` | Optional faint raw points/tethers; rejected observations are not extra people |
| `global_2d_map.png`, `map_config.json` | Architectural base map and explicit room geometry |
| `projections.json` | Raw image points, corrected projections, weights, inlier flags, associations and `fused_people` |
| `transforms_refined.json` | Refined image/map/canvas homographies for all eight cameras |
| `verification_metrics.json` | Per-room headcounts, selected-frame counts and detection statistics |
| `laboratory/fusion_metrics.json`, `passageway/fusion_metrics.json` | Headcount distributions, IDs, consensus errors and weighting rule |
| `laboratory/projection_optimization.json`, `passageway/projection_optimization.json` | Raw validation errors and calibration safeguards |
| `artifact_validation.json` | Decoding, geometry, weighted-centroid, identity and reference-frame checks |

Each raw observation keeps its camera identity and original measured pixel point. Rejected projections are retained, not moved into the room. Each fused record lists its member detection IDs/cameras, inlier diameter, weight-derived position and observation/prediction state.

Globally unique IDs mean unique labels in this display. They do not assert cross-site identity continuity between the independent recordings.

## Geometry and source data

The existing structural pipeline remains available:

```bash
python scripts/download_sample_data.py
python scripts/download_additional_scenes.py
python scripts/download_structural_models.py
python scripts/download_passageway_verification.py

python main.py --mode build-map --room laboratory --output-dir outputs/laboratory
python scripts/analyze_entries.py --output-dir outputs/laboratory
python main.py --mode build-map --data-dir data/passageway --output-dir outputs/passageway
```

Run the person pipeline afterward. Download/rectification scripts must run sequentially because they update the Passageway manifest.

Room walls come from semantic floor–wall seam pixels projected onto the ground and fitted by RANSAC/Manhattan regularization. Depth Anything V2 Metric Small supplies floor/structural depth; SegFormer B2 filters furniture. Laboratory is approximately 6.38 × 6.47 m. Passageway’s visible domain is approximately 8.47 × 19.36 m, with an unobserved open end explicitly dashed. Intrinsics, metric scale and wall-mount poses are estimates, not surveyed values.

Camera centers and axes satisfy C = −Rᵀt and d = Rᵀ[0,0,1]. FOV rays intersect the room boundaries; their angles follow estimated intrinsics. Laboratory D01 is supported by trajectory entries around 4.5 s and 13.5 s; its width/position remain estimates. No shared doorway to Passageway is asserted.

The raw sample data contains first frames 0–99 per camera for structure, Laboratory verification frames 500–599 and Passageway verification frames 250–349. Four sparse 45-second Laboratory timelines support entry analysis. Dataset-specific calibration assumptions and model outputs remain in each scene’s `structure/` directory and data manifest.

Matrix directions are explicit:

```text
p_map ~ H_image_to_map @ [u,v,1]
p_image ~ H_map_to_image @ [X,Y,1]
p_canvas ~ T_map_to_canvas @ p_map
```

Divide by the homogeneous third component. Local maps use model-estimated metres; atlas translations are arbitrary presentation transforms. There is no recovered shared physical world frame between scenes.

## Validation and limits

The regression suite checks projective refinement, distinct nearby identities, inverse-distance weighting, rejection of drifting views, complete-link diameters, temporary occlusion expiry and the existing geometry/stitching behavior. The artifact validator checks the reference frame labels **only as test expectations**, verifies that all six selected markers have actual multi-camera support, recomputes weighted centroids and decodes the saved video.

Full-clip counts can still vary with visibility and association confidence. Rejected raw observations, headcount distributions and short predictions are exposed in JSON. This sample does not establish perfect recall on every frame, globally accurate homographies outside the observed paths, learned ReID performance or surveyed sub-0.35 m localization.

## Sources

[EPFL videos and calibration](https://www.epfl.ch/labs/cvlab/data/data-pom-index-php/), [official Depth Anything V2](https://github.com/DepthAnything/Depth-Anything-V2/tree/main/metric_depth), [metric indoor model](https://huggingface.co/depth-anything/Depth-Anything-V2-Metric-Hypersim-Small), [SegFormer B2](https://huggingface.co/nvidia/segformer-b2-finetuned-ade-512-512), [YOLO pose weights](https://github.com/ultralytics/assets/releases/download/v8.3.0/yolo11n-pose.pt), [official ResNet18 weights](https://download.pytorch.org/models/resnet18-f37072fd.pth).

EPFL data is provided for research by CVLab–EPFL; cite Fleuret et al., TPAMI 2008 when publishing results. Model provenance is retained in `models/structural_sources.json` and `models/person_model_sources.json`.
