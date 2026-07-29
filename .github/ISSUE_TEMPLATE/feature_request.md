---
name: Feature request
about: Suggest an addition or improvement
title: ""
labels: enhancement
---

**What problem does this solve?**

**Proposed approach**
If this is a new `Detector`/`PitchCalibrator`/`TeamClassifier`/`FrameTracker`
backend, note which protocol it would satisfy (see
`src/agon/interfaces.py`) and any new dependency it would need
(and whether that dependency should live behind an extra, like `[train]`,
rather than becoming a core dependency — see the README's Modernization
section for the reasoning behind that split).

**Alternatives considered**

**Anything else**
