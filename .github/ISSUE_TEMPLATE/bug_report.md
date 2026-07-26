---
name: Bug report
about: Something didn't work the way it should
title: ""
labels: bug
---

**What happened**

**What you expected instead**

**Reproduction**
- Command / config used (`soccer-analysis --format ... --model ... --calibration ...`, or the `run_pipeline(...)` call if scripting):
- `PipelineConfig` non-defaults (`calibration_mode`, `team_classifier`, `tracker_backend`, ...), if any:

**Footage/checkpoint characteristics** (these matter a lot for what's actually reachable in this codebase)
- Detector backend: ONNX (default) / Ultralytics (`.pt`)
- Checkpoint type: soccer-specific (player/goalkeeper/referee/ball classes) / generic COCO (person/sports ball only)
- Roughly how the camera behaves: static wide shot / pans and zooms / broadcast-style cuts

**Environment**
- OS + architecture:
- Python version:
- Output of `pip show soccer-analysis onnxruntime` (and `torch`/`ultralytics`/`boxmot` if the `[train]` extra is installed):

**Logs / traceback**

```
paste here
```
