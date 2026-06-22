from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Mapping

from recsys.types import RankedCandidate


@dataclass(frozen=True)
class InteractionWeights:
    """Weights used to combine explicit interaction probabilities.

    In short-video recommendation, like/share/comment/follow/favorite usually
    have different business meanings. Share and follow are often stronger
    satisfaction signals than a cheap like, so they receive larger weights here.
    """

    like: float = 0.35
    share: float = 0.25
    comment: float = 0.15
    follow: float = 0.15
    favorite: float = 0.10


@dataclass(frozen=True)
class VideoFusionWeights:
    """Exponents for the independent fine-rank fusion formula.

    Formula:
        score = p_play^play
              * watch_value^watch
              * interaction_value^interaction
              * (1 - p_short_play)^short_play_penalty
              * (1 - p_negative)^negative_penalty
              * quality^quality
              * freshness^freshness

    The formula is a geometric product. Compared with a linear weighted sum, a
    product score is stricter: one very weak core objective can pull down the
    item, which is usually desirable for immersive video feeds.
    """

    play: float = 0.35
    watch: float = 0.45
    interaction: float = 0.12
    short_play_penalty: float = 0.30
    negative_penalty: float = 0.55
    quality: float = 0.08
    freshness: float = 0.04


@dataclass(frozen=True)
class WatchValueConfig:
    """Controls how finish and long-play probabilities are blended.

    Formula:
        watch_value = finish_weight * p_finish
                    + long_play_weight * p_long_play * duration_gain(duration)

    `duration_gain` is saturated by `duration_cap_seconds` to reduce duration
    bias. This keeps long-video value in the score, but avoids letting raw video
    length dominate the final rank.
    """

    finish_weight: float = 0.45
    long_play_weight: float = 0.55
    duration_cap_seconds: float = 120.0
    duration_reference_seconds: float = 30.0


@dataclass(frozen=True)
class VideoFineRankFusion:
    """Standalone scorer that fuses calibrated fine-rank task probabilities."""

    weights: VideoFusionWeights = field(default_factory=VideoFusionWeights)
    watch_config: WatchValueConfig = field(default_factory=WatchValueConfig)
    interaction_weights: InteractionWeights = field(default_factory=InteractionWeights)

    def score(self, item: RankedCandidate) -> float:
        """Return the final fine-rank formula score for one candidate."""

        return fuse_video_fine_rank_score(
            item,
            weights=self.weights,
            watch_config=self.watch_config,
            interaction_weights=self.interaction_weights,
        )


def fuse_video_fine_rank_score(
    item: RankedCandidate,
    weights: VideoFusionWeights | None = None,
    watch_config: WatchValueConfig | None = None,
    interaction_weights: InteractionWeights | None = None,
) -> float:
    """Fuse play, watch, interaction, short-play, and negative objectives.

    Recommended model heads for this formula:
        play/click: whether the user will enter or continue watching the video.
        finish: whether the user will complete the video.
        long_play: whether watch time exceeds a useful threshold.
        short_play: whether the user will quickly skip the video.
        like/share/comment/follow/favorite: explicit positive interactions.
        negative: not interested, hide, report, dislike, or similar feedback.
    """

    weights = weights or VideoFusionWeights()
    watch_config = watch_config or WatchValueConfig()
    interaction_weights = interaction_weights or InteractionWeights()

    scores = item.task_scores
    item_info = item.candidate.item
    play = task_probability(scores, "play", fallback_key="click")
    watch = watch_value(
        scores,
        duration_seconds=item_info.duration_seconds,
        config=watch_config,
    )
    interaction = interaction_value(scores, interaction_weights)
    short_play = task_probability(scores, "short_play")
    negative = task_probability(scores, "negative")

    return (
        bounded_probability(play) ** weights.play
        * bounded_probability(watch) ** weights.watch
        * bounded_probability(interaction) ** weights.interaction
        * (1.0 - bounded_probability(short_play)) ** weights.short_play_penalty
        * (1.0 - bounded_probability(negative)) ** weights.negative_penalty
        * bounded_probability(item_info.quality_score) ** weights.quality
        * bounded_probability(item_info.freshness_score) ** weights.freshness
    )


def watch_value(
    scores: Mapping[str, float],
    duration_seconds: float,
    config: WatchValueConfig | None = None,
) -> float:
    """Blend finish-rate and long-play-rate into one effective watch value."""

    config = config or WatchValueConfig()
    finish = task_probability(scores, "finish")
    long_play = task_probability(scores, "long_play")
    return (
        config.finish_weight * finish
        + config.long_play_weight * long_play * duration_gain(duration_seconds, config)
    )


def interaction_value(
    scores: Mapping[str, float],
    weights: InteractionWeights | None = None,
) -> float:
    """Return weighted average probability for explicit positive interactions."""

    weights = weights or InteractionWeights()
    weighted_sum = (
        weights.like * task_probability(scores, "like")
        + weights.share * task_probability(scores, "share")
        + weights.comment * task_probability(scores, "comment")
        + weights.follow * task_probability(scores, "follow")
        + weights.favorite * task_probability(scores, "favorite", fallback_key="collect")
    )
    total_weight = (
        weights.like
        + weights.share
        + weights.comment
        + weights.follow
        + weights.favorite
    )
    if total_weight <= 0.0:
        return 0.0
    return weighted_sum / total_weight


def duration_gain(duration_seconds: float, config: WatchValueConfig | None = None) -> float:
    """Return a saturated duration factor for long-play value.

    Formula:
        gain = log(1 + min(duration, cap)) / log(1 + reference)

    Values above 1 are allowed because long videos can create more watch time,
    but the logarithm and cap prevent very long videos from dominating only due
    to duration.
    """

    config = config or WatchValueConfig()
    duration = max(0.0, min(duration_seconds, config.duration_cap_seconds))
    reference = max(1.0, config.duration_reference_seconds)
    return math.log1p(duration) / math.log1p(reference)


def task_probability(
    scores: Mapping[str, float],
    key: str,
    fallback_key: str | None = None,
) -> float:
    """Read a calibrated task probability and clamp it into [0, 1]."""

    value = scores.get(key)
    if value is None and fallback_key is not None:
        value = scores.get(fallback_key)
    return clamp(value or 0.0, 0.0, 1.0)


def bounded_probability(value: float) -> float:
    """Avoid exact 0/1 before exponentiation in product formulas."""

    return clamp(value, 1.0e-6, 1.0 - 1.0e-6)


def clamp(value: float, lower: float, upper: float) -> float:
    """Limit a numeric value to the closed interval [lower, upper]."""

    return min(upper, max(lower, value))
