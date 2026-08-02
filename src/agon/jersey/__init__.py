"""Jersey number recognition: single-frame classification
(``onnx_classifier``) plus track-level confidence-weighted aggregation
(``aggregator``) across every frame a track appears in. See
``agon.interfaces.JerseyClassifier`` for why these are two separate
concerns rather than one "just classify it" step.
"""
