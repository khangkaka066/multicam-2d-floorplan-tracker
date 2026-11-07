# Multi-room datasets and topology

The CLI discovers every `manifest.json` below `--data-dir`. Each manifest may describe one room or group cameras by `room_id`. Room/camera IDs are arbitrary safe names; they are not limited to EPFL names or `cam0`–`cam3`. Duplicate room IDs across manifests are rejected.

## Minimal room manifest

```json
{
  "room_id": "office",
  "coordinate_units": "meters",
  "map_y_axis": "up",
  "timing": {"clock_id": "building-session-1", "time_offset_seconds": 0.0},
  "pose_scales": [1280],
  "allow_open_ends": true,
  "cameras": [
    {
      "id": "north",
      "room_id": "office",
      "width": 1280, "height": 720, "fps": 25, "frames": 1000,
      "frames_dir": "north/frames",
      "frame_pattern": "{index:06d}.jpg",
      "H_image_to_map": [[0.01, 0, 0], [0, -0.01, 7.2], [0, 0, 1]]
    }
  ]
}
```

The example matrix only illustrates the schema; replace it with real image-to-ground calibration. A `video` path may replace `frames_dir`. `verification_frames_dir` / `verification_video` optionally select a separate clip; `verification_start_frame` records its source offset. Camera rates within a room must already be synchronized/resampled. Room streams can have different rates or offsets.

Calibration can be camera `H_image_to_map`, manifest `homographies` keyed by camera, a JSON `calibration` file, or camera `K`, `R_world_to_camera`, `t_world_to_camera`. Published EPFL text calibration remains supported. Calibration dimensions can be supplied as `calibration_image_size`. Bare uncalibrated videos still need an external reconstruction/calibration adapter; the importer cannot infer an arbitrary file format or metric world frame.

`map_y_axis` defaults to `down` for compatibility with EPFL. Use `up` for a right-handed metric floor coordinate system. `rectification: "auto"` is the default: incompatible ground-plane pinhole gauges are rectified from the first/selected camera's depth floor. Set `rectification_anchor` to a camera ID, `rectification: "depth"` to force that route, or `rectification: "none"` to retain your calibration. An existing `plane_rectification_file` is honored. New rectifications live in `outputs/<room>/structure/ground_rectification.json`; raw manifests are not rewritten. Open scene limits are marked explicitly instead of becoming observed walls.

## Tier 1: observed transitions

```bash
python main.py --mode build-map
python main.py --mode project
# Or reconstruct, project, analyze and assemble in one run:
python main.py
```

Doorway discovery clusters uncensored trajectory births/deaths near walls, checks motion toward/away from the boundary, and applies MeanShift along each wall. At least two tracklets support an inferred door. If supplied, a `timeline_manifest.json` or per-camera `timeline_video` provides longer entry/exit coverage; otherwise the normal pose observations are used. The `--frames` option controls the fallback clip length.

Cross-room matching requires **the same explicitly declared `timing.clock_id`**. Local source times become common times by adding `time_offset_seconds`. This permits synchronized cameras or sequential clips with known start offsets. Missing clock IDs never imply synchronization. The independent EPFL recordings intentionally have no shared clock and therefore do not auto-connect.

Exit/entry descriptors come from `src/appearance.py` (normalized ResNet18 crop features). Hungarian matching accepts only `dt_min < destination_time - source_time < dt_max`, compatible feature models and sufficiently close appearance. Ambiguous alternatives are rejected. A doorway pair needs at least two independent matched transitions; competing destinations are not guessed. Pretrained image features are not a dedicated ReID model, and these confidence checks do not establish ground-truth identity.

## Tier 2: configured adjacency

`configs/topology.json` is loaded automatically and defaults to an empty link list. Copy/edit `configs/topology.example.json` for a building:

```json
{
  "links": [
    {"from": "room1:D01", "to": "room2:D01", "gap_m": 0.2},
    {"from": "room2:D02", "to": "room3:D01", "gap_m": 0.0}
  ],
  "transition": {"dt_min": 0.0, "dt_max": 30.0, "appearance_threshold": 0.25, "min_transitions": 2}
}
```

```bash
python main.py --mode build-global --topology configs/topology.json --layout stitched
python main.py --mode build-global --layout atlas
python main.py --mode build-map --room office
```

Configured links supplement inferred links. Unknown endpoints, conflicting doorway destinations, negative/nonfinite gaps, inconsistent cycles, incompatible doorway widths and room overlaps raise errors. Doors are identified by their actual `map_config.json` IDs, scoped by room. Inferred doors use `D01`, `D02`, etc.

For independent footage without any crossings, supply measured openings in the room manifest before building:

```json
{"doors": [{"id": "D01", "wall_index": 1, "offset_m": 1.5, "width_m": 0.9}]}
```

Wall indices refer to the CCW Manhattan polygon after canonicalization. A `wall_id` may replace `wall_index`; inspect the local map before supplying adjacency. These are declared measurements, not evidence extracted from pedestrians. Inferred events near a measured opening are retained while the measured width/location wins.

Door tangent and facing-wall constraints choose an orthogonal rigid transform (0/90/180/270 degrees plus translation). No scaling or reflection is applied to rooms. Slightly different inferred body-envelope doorway widths (up to 0.35 m) are harmonized around their centers and the original widths are recorded; measured openings remain authoritative. Larger width disagreements require supplied measurements. `gap_m` separates the facing wall surfaces and renders a short doorway connector. With zero gap, wall/door segments coincide. Fully connected layouts are centered on the camera-network centroid. Disconnected components retain only their internally supported geometry and are separated by display spacing.

`--layout stitched` is the default. If no valid transitions or config links exist, it falls back to an atlas; the mere presence of similar-looking doors is insufficient evidence. `--layout atlas` forces independent display placement. The current EPFL showcase consequently remains an explicitly unregistered atlas.

## Export and verification

`outputs/topology.json` and `map_config.json.alignment` report links, evidence source, matches, components and layout status (`stitched`, `partial`, `atlas`). `physical_building_registration` is true only for one fully connected multi-room graph. Config-based registration is an external assertion, not a recovered pedestrian transition.

All room polygons, door centers, raw/fused points, camera centers/axes, extrinsics and original/refined homographies receive the same rigid transforms. For shared-clock recordings, verification panels select frames by source timestamps; unavailable room frames are blank. Independent clips pair indices only for display. IDs remain globally unique display labels rather than a claim of continuous cross-room identity tracking.

The topology unit tests include synthetic three-room transitions and declarative adjacency with a gap. The real EPFL run validates discovery, structural reconstruction, inference and safe fallback; it contains no known real cross-room pedestrian transitions.

The current `scripts/validate_outputs.py` is the strict EPFL reference-artifact check (the documented two-scene, 100-frame showcase). Generic three-room assembly, shared-clock matching and config-driven rotations are covered by `tests/test_architecture.py`.
