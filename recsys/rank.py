from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Iterable, Mapping

from recsys.fine_rank_fusion import VideoFineRankFusion
from recsys.recall import build_user_field_scores
from recsys.types import Candidate, RankedCandidate, UserProfile


@dataclass(frozen=True)
class FusionWeights:
    """Exponents used by the final multiplicative ranking formula."""

    click: float = 0.25
    finish: float = 0.35
    long_play: float = 0.35
    short_play: float = 0.30
    like: float = 0.08
    share: float = 0.06
    follow: float = 0.08
    negative: float = 0.60


class CoarseRanker:
    """Lightweight ranker that narrows recalled items before fine ranking."""

    def rank(
        self,
        user: UserProfile,
        candidates: Iterable[Candidate],
        limit: int,
        now: float,
    ) -> list[RankedCandidate]:
        """Estimate task scores and keep the highest coarse-scored candidates."""

        field_scores = build_user_field_scores(user.actions, now)
        ranked = [
            RankedCandidate(
                candidate=candidate,
                task_scores=estimate_task_scores(candidate, field_scores, user.features),
                score=estimate_coarse_score(candidate, field_scores),
            )
            for candidate in candidates
        ]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]


class FineRanker:
    """Final ranker that fuses calibrated task scores into one rank score."""

    def __init__(
        self,
        weights: FusionWeights | None = None,
        fusion: VideoFineRankFusion | None = None,
    ) -> None:
        """Use legacy weights or a standalone video fusion scorer."""

        self.weights = weights or FusionWeights()
        self.fusion = fusion

    def rank(self, candidates: Iterable[RankedCandidate], limit: int) -> list[RankedCandidate]:
        """Re-score candidates with the multiplicative fine-rank formula."""

        ranked = [
            RankedCandidate(
                candidate=item.candidate,
                task_scores=item.task_scores,
                score=(
                    self.fusion.score(item)
                    if self.fusion is not None
                    else fuse_task_scores(item.task_scores, self.weights)
                ),
            )
            for item in candidates
        ]
        return sorted(ranked, key=lambda item: item.score, reverse=True)[:limit]


def estimate_coarse_score(candidate: Candidate, field_scores: Mapping[str, Mapping[str, float]]) -> float:
    """Combine recall score, user affinity, and item priors for coarse ranking."""

    affinity = user_item_affinity(candidate, field_scores)
    item = candidate.item
    return (
        0.45 * candidate.recall_score
        + 0.25 * affinity
        + 0.15 * item.quality_score
        + 0.10 * item.freshness_score
        + 0.05 * item.popularity_score
    )


def estimate_task_scores(
    candidate: Candidate,
    field_scores: Mapping[str, Mapping[str, float]],
    user_features: Mapping[str, float],
) -> dict[str, float]:
    """Produce rule-based task probabilities as stand-ins for model heads."""

    item = candidate.item
    affinity = user_item_affinity(candidate, field_scores)
    activity = clamp(user_features.get("activity_level", 0.5), 0.0, 1.0)
    short_video = 1.0 if 0 < item.duration_seconds <= 20 else 0.0
    long_video = 1.0 if item.duration_seconds >= 60 else 0.0

    return {
        "click": sigmoid(-1.1 + 1.7 * candidate.recall_score + 0.9 * affinity + 0.3 * item.popularity_score),
        "finish": sigmoid(-0.9 + 1.1 * affinity + 0.7 * item.quality_score + 0.3 * short_video - 0.2 * long_video),
        "long_play": sigmoid(-1.0 + 1.2 * affinity + 0.6 * item.quality_score + 0.4 * long_video),
        "short_play": sigmoid(-0.7 - 0.8 * affinity - 0.5 * item.quality_score + 0.4 * long_video),
        "like": sigmoid(-2.1 + 1.4 * affinity + 0.4 * activity + 0.4 * item.quality_score),
        "share": sigmoid(-2.5 + 1.0 * affinity + 0.4 * item.quality_score + 0.2 * item.freshness_score),
        "follow": sigmoid(-2.3 + 1.5 * author_affinity(item.author_id, field_scores) + 0.4 * activity),
        "negative": sigmoid(-2.2 - 1.1 * affinity - 0.5 * item.quality_score + 0.4 * long_video),
    }


def fuse_task_scores(scores: Mapping[str, float], weights: FusionWeights) -> float:
    """Fuse multi-task probabilities with the production-style product formula."""

    click = bounded_probability(scores.get("click", 0.0))
    finish = bounded_probability(scores.get("finish", 0.0))
    long_play = bounded_probability(scores.get("long_play", 0.0))
    short_play = bounded_probability(scores.get("short_play", 0.0))
    like = bounded_probability(scores.get("like", 0.0))
    share = bounded_probability(scores.get("share", 0.0))
    follow = bounded_probability(scores.get("follow", 0.0))
    negative = bounded_probability(scores.get("negative", 0.0))

    return (
        click**weights.click
        * finish**weights.finish
        * long_play**weights.long_play
        * (1.0 - short_play) ** weights.short_play
        * like**weights.like
        * share**weights.share
        * follow**weights.follow
        * (1.0 - negative) ** weights.negative
    )


def user_item_affinity(candidate: Candidate, field_scores: Mapping[str, Mapping[str, float]]) -> float:
    """Estimate how well the candidate matches the user's field preferences."""

    item = candidate.item
    author = author_affinity(item.author_id, field_scores)
    category = normalized_lookup(field_scores, "category", item.category_id)
    city = normalized_lookup(field_scores, "city", item.city_id)
    tag = max((normalized_lookup(field_scores, "tag", tag) for tag in item.tags), default=0.0)
    return clamp(0.45 * author + 0.25 * category + 0.10 * city + 0.20 * tag, 0.0, 1.0)


def author_affinity(author_id: str, field_scores: Mapping[str, Mapping[str, float]]) -> float:
    """Return normalized affinity between the user and an author."""

    return normalized_lookup(field_scores, "author", author_id)


def normalized_lookup(field_scores: Mapping[str, Mapping[str, float]], field: str, value: str) -> float:
    """Read a field preference and normalize it by that field's max score."""

    values = field_scores.get(field, {})
    if not values or not value:
        return 0.0
    max_value = max(values.values())
    if max_value <= 0:
        return 0.0
    return clamp(values.get(value, 0.0) / max_value, 0.0, 1.0)


def bounded_probability(value: float) -> float:
    """Clamp probabilities away from exact 0 or 1 before exponentiation."""

    return clamp(value, 1e-6, 1.0 - 1e-6)


def clamp(value: float, lower: float, upper: float) -> float:
    """Limit a numeric value to the closed interval [lower, upper]."""

    return min(upper, max(lower, value))


def sigmoid(value: float) -> float:
    """Map an unbounded logit-like value into a probability."""

    return 1.0 / (1.0 + math.exp(-value))
