from __future__ import annotations

import math
from collections import defaultdict
from typing import Iterable, Mapping

from recsys.types import Candidate, Item, UserAction, UserProfile, normalize_scores


ACTION_WEIGHTS = {
    "click": 1.0,
    "play": 2.0,
    "finish": 4.0,
    "long_play": 4.5,
    "like": 6.0,
    "comment": 8.0,
    "share": 10.0,
    "follow": 12.0,
}

FIELD_WEIGHTS = {
    "author": 1.4,
    "category": 1.0,
    "city": 0.5,
    "tag": 0.8,
}


class TagRecall:
    """Recall items by matching user history fields against inverted indexes."""

    def __init__(self, items: Iterable[Item], name: str = "tag") -> None:
        """Build author/category/city/tag indexes over the item pool."""

        self.name = name
        self.items = {item.item_id: item for item in items}
        self.index: dict[tuple[str, str], list[Item]] = defaultdict(list)
        for item in self.items.values():
            self.index[("author", item.author_id)].append(item)
            self.index[("category", item.category_id)].append(item)
            self.index[("city", item.city_id)].append(item)
            for tag in item.tags:
                self.index[("tag", tag)].append(item)

    def recall(
        self,
        user: UserProfile,
        limit: int,
        now: float,
        max_values_per_field: int = 20,
    ) -> list[Candidate]:
        """Return top candidates whose fields match weighted user interests."""

        field_scores = build_user_field_scores(user.actions, now)
        item_scores: dict[str, float] = defaultdict(float)
        item_sources: dict[str, set[str]] = defaultdict(set)

        for field_name, values in field_scores.items():
            top_values = sorted(values.items(), key=lambda pair: pair[1], reverse=True)
            for field_value, user_score in top_values[:max_values_per_field]:
                for item in self.index.get((field_name, field_value), ()):
                    match_weight = FIELD_WEIGHTS[field_name]
                    score = user_score * match_weight * item.quality_score * item.freshness_score
                    item_scores[item.item_id] += score
                    item_sources[item.item_id].add(f"{self.name}:{field_name}")

        normalized = normalize_scores(item_scores)
        candidates = [
            Candidate(
                item=self.items[item_id],
                recall_score=score,
                recall_sources=tuple(sorted(item_sources[item_id])),
            )
            for item_id, score in normalized.items()
        ]
        return sorted(candidates, key=lambda candidate: candidate.recall_score, reverse=True)[:limit]


class VectorRecall:
    """Recall items by cosine similarity between user and item embeddings."""

    def __init__(
        self,
        items: Iterable[Item],
        name: str = "two_tower",
        history_window: int | None = None,
    ) -> None:
        """Keep embeddable items and optionally limit user history length."""

        self.name = name
        self.items = [item for item in items if item.embedding]
        self.item_by_id = {item.item_id: item for item in self.items}
        self.history_window = history_window

    def recall(self, user: UserProfile, limit: int, now: float) -> list[Candidate]:
        """Build a user vector from behavior embeddings and retrieve nearest items."""

        user_vector = build_user_vector(user.actions, self.item_by_id, now, self.history_window)
        if not user_vector:
            return []

        candidates: list[Candidate] = []
        for item in self.items:
            similarity = cosine(user_vector, item.embedding)
            if similarity <= 0:
                continue
            score = similarity * item.quality_score * item.freshness_score
            candidates.append(Candidate(item=item, recall_score=score, recall_sources=(self.name,)))
        return sorted(candidates, key=lambda candidate: candidate.recall_score, reverse=True)[:limit]


class MultiInterestVectorRecall:
    """Recall items from multiple category-level user interest vectors."""

    def __init__(self, items: Iterable[Item], name: str = "multi_interest") -> None:
        """Keep embeddable items for category-grouped vector recall."""

        self.name = name
        self.items = [item for item in items if item.embedding]
        self.item_by_id = {item.item_id: item for item in self.items}

    def recall(self, user: UserProfile, limit: int, now: float) -> list[Candidate]:
        """Build one interest vector per category and merge their best matches."""

        grouped_actions: dict[str, list[UserAction]] = defaultdict(list)
        for action in user.actions:
            if action.category_id:
                grouped_actions[action.category_id].append(action)

        item_scores: dict[str, float] = {}
        for category_id, actions in grouped_actions.items():
            interest = build_user_vector(tuple(actions), self.item_by_id, now, history_window=None)
            if not interest:
                continue
            for item in self.items:
                similarity = cosine(interest, item.embedding)
                if similarity <= 0:
                    continue
                score = similarity * item.quality_score * item.freshness_score
                item_scores[item.item_id] = max(item_scores.get(item.item_id, 0.0), score)

        normalized = normalize_scores(item_scores)
        candidates = [
            Candidate(item=self.item_by_id[item_id], recall_score=score, recall_sources=(self.name,))
            for item_id, score in normalized.items()
        ]
        return sorted(candidates, key=lambda candidate: candidate.recall_score, reverse=True)[:limit]


def merge_candidates(candidate_groups: Iterable[Iterable[Candidate]], limit: int) -> list[Candidate]:
    """Deduplicate recall results, keep best score, and union recall sources."""

    merged: dict[str, Candidate] = {}
    for group in candidate_groups:
        for candidate in group:
            item_id = candidate.item.item_id
            if item_id not in merged:
                merged[item_id] = candidate
                continue

            current = merged[item_id]
            sources = tuple(sorted(set(current.recall_sources) | set(candidate.recall_sources)))
            merged[item_id] = Candidate(
                item=candidate.item,
                recall_score=max(current.recall_score, candidate.recall_score),
                recall_sources=sources,
            )

    return sorted(merged.values(), key=lambda candidate: candidate.recall_score, reverse=True)[:limit]


def build_user_field_scores(
    actions: Iterable[UserAction],
    now: float,
    tau_days: float = 14.0,
) -> dict[str, dict[str, float]]:
    """Aggregate decayed user preferences for author/category/city/tag fields."""

    scores: dict[str, dict[str, float]] = {
        "author": defaultdict(float),
        "category": defaultdict(float),
        "city": defaultdict(float),
        "tag": defaultdict(float),
    }
    for action in actions:
        strength = action_strength(action, now, tau_days)
        if action.author_id:
            scores["author"][action.author_id] += strength
        if action.category_id:
            scores["category"][action.category_id] += strength
        if action.city_id:
            scores["city"][action.city_id] += strength
        for tag in action.tags:
            scores["tag"][tag] += strength
    return {field: dict(values) for field, values in scores.items()}


def build_user_vector(
    actions: Iterable[UserAction],
    items: Mapping[str, Item],
    now: float,
    history_window: int | None,
) -> tuple[float, ...]:
    """Create a weighted average embedding from the user's item history."""

    sorted_actions = sorted(actions, key=lambda action: action.timestamp, reverse=True)
    if history_window is not None:
        sorted_actions = sorted_actions[:history_window]

    weighted_sum: list[float] = []
    total_weight = 0.0
    for action in sorted_actions:
        item = items.get(action.item_id)
        if item is None or not item.embedding:
            continue
        weight = action_strength(action, now)
        if not weighted_sum:
            weighted_sum = [0.0] * len(item.embedding)
        if len(item.embedding) != len(weighted_sum):
            continue
        for index, value in enumerate(item.embedding):
            weighted_sum[index] += value * weight
        total_weight += weight

    if not weighted_sum or total_weight <= 0:
        return ()
    return tuple(value / total_weight for value in weighted_sum)


def action_strength(action: UserAction, now: float, tau_days: float = 14.0) -> float:
    """Score one behavior by action type, recency, and completion quality."""

    action_weight = ACTION_WEIGHTS.get(action.action, 1.0)
    age_days = max(0.0, (now - action.timestamp) / 86400.0)
    time_decay = math.exp(-age_days / tau_days)
    play_quality = 1.0 + action.completion_ratio
    return action_weight * time_decay * play_quality


def cosine(left: tuple[float, ...], right: tuple[float, ...]) -> float:
    """Compute cosine similarity, returning zero for invalid vectors."""

    if not left or not right or len(left) != len(right):
        return 0.0
    dot = sum(a * b for a, b in zip(left, right))
    left_norm = math.sqrt(sum(a * a for a in left))
    right_norm = math.sqrt(sum(b * b for b in right))
    if left_norm <= 0 or right_norm <= 0:
        return 0.0
    return dot / (left_norm * right_norm)
