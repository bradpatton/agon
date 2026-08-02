"""Combines many single-frame ``JerseyClassifier.classify()`` calls (one
per frame a track appears in) into one jersey number per track.

Why this exists, not just "trust the classifier per frame": the official
SoccerNet Jersey Number Recognition task (github.com/SoccerNet/sn-jersey)
is explicitly framed as a whole-tracklet problem, not single-frame
classification -- "the jersey numbers might be visible on a very small
subset of the whole tracklet" is the task's own stated premise, and its
reference solutions score ~93% by aggregating across a tracklet, not by
classifying isolated crops (which this project measured at ~1-1.5% top1,
consistent with that premise -- see the project plan). Agon already has
stable track IDs from its tracker, so this is the natural, already-built
piece of infrastructure to aggregate across.
"""

from __future__ import annotations


def aggregate_track_jersey_numbers(
    predictions_by_track: dict[int, list[tuple[int | None, float]]],
    min_confidence: float = 0.5,
) -> dict[int, int | None]:
    """``predictions_by_track``: track_id -> list of (jersey_number, confidence)
    from one ``classify()`` call per frame that track appeared in (see
    ``JerseyClassifier.classify()``); ``jersey_number`` is None for the
    "unknown"/illegible class.

    Confidence-weighted voting: predictions below ``min_confidence`` (a
    single frame's classifier confidence, not a track-level threshold) are
    dropped before voting, since a low-confidence guess on an illegible
    frame is exactly the noise this aggregation exists to filter out.
    Among the remaining frames, sums confidence per candidate number and
    picks the highest total -- weighting by confidence rather than plain
    majority vote so a few very-confident reads outweigh many marginal
    ones. Returns None for a track if no frame clears the threshold with
    a numeric prediction (a real "we don't know," not a guess).
    """
    result: dict[int, int | None] = {}
    for track_id, predictions in predictions_by_track.items():
        votes: dict[int, float] = {}
        for jersey_number, confidence in predictions:
            if jersey_number is None or confidence < min_confidence:
                continue
            votes[jersey_number] = votes.get(jersey_number, 0.0) + confidence

        result[track_id] = max(votes, key=lambda k: votes[k]) if votes else None

    return result
