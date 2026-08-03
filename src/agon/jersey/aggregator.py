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

``min_votes`` exists because a single confident-but-wrong frame is a real,
observed failure mode, not a hypothetical one: validating
``agon.jersey.ocr_reader.EasyOcrJerseyReader`` against real crops with
known ground truth found a 93%-confidence misread (true 36 read as 35) --
high individual-frame confidence alone doesn't guarantee correctness.
Requiring several independent frames to agree is what actually protects
against that, not a stricter ``min_confidence`` alone (which the same
validation showed doesn't cleanly separate right from wrong reads either).
"""

from __future__ import annotations


def aggregate_track_jersey_numbers(
    predictions_by_track: dict[int, list[tuple[int | None, float]]],
    min_confidence: float = 0.5,
    min_votes: int = 1,
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
    ones.

    ``min_votes``: the winning candidate must additionally have been
    predicted by at least this many separate frames (not just a high
    confidence sum) -- see the module docstring for why. Default 1 keeps
    the original single-confident-frame-is-enough behavior; callers that
    can afford to wait for more evidence (most tracks span many frames)
    should pass a higher value.

    Returns None for a track if no candidate clears both thresholds (a
    real "we don't know," not a guess).
    """
    result: dict[int, int | None] = {}
    for track_id, predictions in predictions_by_track.items():
        vote_totals: dict[int, float] = {}
        vote_counts: dict[int, int] = {}
        for jersey_number, confidence in predictions:
            if jersey_number is None or confidence < min_confidence:
                continue
            vote_totals[jersey_number] = vote_totals.get(jersey_number, 0.0) + confidence
            vote_counts[jersey_number] = vote_counts.get(jersey_number, 0) + 1

        winner = max(vote_totals, key=lambda k: vote_totals[k]) if vote_totals else None
        if winner is not None and vote_counts[winner] < min_votes:
            winner = None
        result[track_id] = winner

    return result
