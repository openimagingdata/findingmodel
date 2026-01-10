"""DuckDB search and scoring utilities."""

from __future__ import annotations

from array import array
from collections.abc import Sequence

ScoreTuple = tuple[str, float]


def normalize_scores(scores: Sequence[float]) -> list[float]:
    """Min-max normalise scores to the [0, 1] range."""
    if not scores:
        return []

    minimum = min(scores)
    maximum = max(scores)

    if minimum == maximum:
        return [1.0 for _ in scores]

    span = maximum - minimum
    return [(score - minimum) / span for score in scores]


def weighted_fusion(
    results_a: Sequence[ScoreTuple],
    results_b: Sequence[ScoreTuple],
    *,
    weight_a: float = 0.3,
    weight_b: float = 0.7,
    normalise: bool = True,
) -> list[ScoreTuple]:
    """Combine two result sets using weighted score fusion."""
    scores_a = dict(results_a)
    scores_b = dict(results_b)

    if normalise and scores_a:
        keys_a = tuple(scores_a.keys())
        normalised_a = normalize_scores(list(scores_a.values()))
        scores_a = dict(zip(keys_a, normalised_a, strict=True))

    if normalise and scores_b:
        keys_b = tuple(scores_b.keys())
        normalised_b = normalize_scores(list(scores_b.values()))
        scores_b = dict(zip(keys_b, normalised_b, strict=True))

    identifiers = set(scores_a) | set(scores_b)
    combined: list[ScoreTuple] = []

    for identifier in identifiers:
        combined_score = weight_a * scores_a.get(identifier, 0.0) + weight_b * scores_b.get(identifier, 0.0)
        combined.append((identifier, combined_score))

    combined.sort(key=lambda item: item[1], reverse=True)
    return combined


def rrf_fusion(
    results_a: Sequence[ScoreTuple],
    results_b: Sequence[ScoreTuple],
    *,
    k: int = 60,
    weight_a: float = 0.5,
    weight_b: float = 0.5,
) -> list[ScoreTuple]:
    """Combine two result sets using Reciprocal Rank Fusion (RRF)."""
    ranks_a = {identifier: index + 1 for index, (identifier, _) in enumerate(results_a)}
    ranks_b = {identifier: index + 1 for index, (identifier, _) in enumerate(results_b)}

    identifiers = set(ranks_a) | set(ranks_b)
    combined: list[ScoreTuple] = []

    for identifier in identifiers:
        rank_a = ranks_a.get(identifier, len(results_a) + 1)
        rank_b = ranks_b.get(identifier, len(results_b) + 1)
        score = weight_a / (k + rank_a) + weight_b / (k + rank_b)
        combined.append((identifier, score))

    combined.sort(key=lambda item: item[1], reverse=True)
    return combined


def l2_to_cosine_similarity(l2_distance: float) -> float:
    """Convert an L2 distance to an approximate cosine similarity."""
    return 1.0 - (l2_distance / 2.0)


def _to_float32(values: Sequence[float]) -> list[float]:
    """Convert an iterable of floats to 32-bit precision."""
    return list(array("f", values))


__all__ = [
    "ScoreTuple",
    "l2_to_cosine_similarity",
    "normalize_scores",
    "rrf_fusion",
    "weighted_fusion",
]
