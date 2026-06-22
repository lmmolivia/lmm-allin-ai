from __future__ import annotations

from dataclasses import dataclass, field
from typing import Mapping, Sequence


@dataclass(frozen=True)
class Item:
    """Content-side entity used by recall, ranking, and post-processing."""

    item_id: str
    author_id: str
    category_id: str
    city_id: str
    tags: tuple[str, ...] = ()
    duration_seconds: float = 0.0
    quality_score: float = 1.0
    freshness_score: float = 1.0
    popularity_score: float = 0.0
    embedding: tuple[float, ...] = ()


@dataclass(frozen=True)
class UserAction:
    """One historical user behavior event with denormalized item fields."""

    item_id: str
    action: str
    timestamp: float
    author_id: str = ""
    category_id: str = ""
    city_id: str = ""
    tags: tuple[str, ...] = ()
    play_time_seconds: float = 0.0
    duration_seconds: float = 0.0

    @property
    def completion_ratio(self) -> float:
        """Return play completion in [0, 1] for weighting user interest."""

        if self.duration_seconds <= 0:
            return 0.0
        return min(1.0, max(0.0, self.play_time_seconds / self.duration_seconds))


@dataclass(frozen=True)
class UserProfile:
    """Request-time user state consumed by the recommendation pipeline."""

    user_id: str
    actions: tuple[UserAction, ...] = ()
    exposed_item_ids: frozenset[str] = field(default_factory=frozenset)
    blocked_authors: frozenset[str] = field(default_factory=frozenset)
    scene: Mapping[str, str] = field(default_factory=dict)
    features: Mapping[str, float] = field(default_factory=dict)


@dataclass(frozen=True)
class Candidate:
    """Item returned by one or more recall channels before ranking."""

    item: Item
    recall_score: float
    recall_sources: tuple[str, ...]


@dataclass(frozen=True)
class RankedCandidate:
    """Candidate after ranking with task probabilities and final score."""

    candidate: Candidate
    task_scores: Mapping[str, float]
    score: float


@dataclass(frozen=True)
class Recommendation:
    """Final response object returned to the caller."""

    item_id: str
    score: float
    recall_sources: tuple[str, ...]
    task_scores: Mapping[str, float]


def normalize_scores(scores: Mapping[str, float]) -> dict[str, float]:
    """Scale a score map by its maximum value while preserving keys."""

    if not scores:
        return {}
    max_score = max(scores.values())
    if max_score <= 0:
        return {key: 0.0 for key in scores}
    return {key: value / max_score for key, value in scores.items()}


def as_tuple(values: Sequence[str]) -> tuple[str, ...]:
    """Convert a string sequence to a tuple and drop empty values."""

    return tuple(value for value in values if value)
