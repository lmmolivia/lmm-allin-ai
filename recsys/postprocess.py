from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass, replace
from typing import Iterable
from typing import Mapping

from recsys.recall import cosine
from recsys.types import RankedCandidate, Recommendation, UserProfile


@dataclass(frozen=True)
class PostRankFusionWeights:
    """Weights for the post-rank product formula.

    Formula:
        final_score = rank_score^rank * recall_score^recall * quality^quality
                    * freshness^freshness * popularity^popularity
                    * (1 - negative)^negative

    In a video recommender, the fine ranker usually provides the main model
    score. Post-rank fusion is where we add controllable business priors such
    as content quality, freshness, recall confidence, and negative feedback.
    """

    rank: float = 1.0
    recall: float = 0.0
    quality: float = 0.0
    freshness: float = 0.0
    popularity: float = 0.0
    negative: float = 0.0


class PostProcessor:
    """Apply response-side filters, diversity penalties, and window constraints."""

    def __init__(
        self,
        diversity_lambda: float = 0.12,
        author_window_limit: int = 1,
        category_window_limit: int = 4,
        city_window_limit: int | None = None,
        window_size: int = 10,
        author_global_limit: int | None = None,
        category_global_limit: int | None = None,
        source_global_limits: Mapping[str, int] | None = None,
        formula_weights: PostRankFusionWeights | None = None,
    ) -> None:
        """Configure formula fusion, diversity strength, and repetition limits."""

        self.diversity_lambda = diversity_lambda
        self.author_window_limit = author_window_limit
        self.category_window_limit = category_window_limit
        self.city_window_limit = city_window_limit
        self.window_size = window_size
        self.author_global_limit = author_global_limit
        self.category_global_limit = category_global_limit
        self.source_global_limits = dict(source_global_limits or {})
        self.formula_weights = formula_weights

    def process(
        self,
        user: UserProfile,
        ranked_candidates: Iterable[RankedCandidate],
        limit: int,
    ) -> list[Recommendation]:
        """Run the full post-processing chain and return final recommendations."""

        candidates = filter_candidates(user, ranked_candidates)
        candidates = deduplicate_candidates(candidates)
        if self.formula_weights is not None:
            candidates = apply_post_rank_formula(candidates, self.formula_weights)
        selected = diversity_rerank(
            candidates,
            limit=limit,
            diversity_lambda=self.diversity_lambda,
            window_size=self.window_size,
            author_window_limit=self.author_window_limit,
            category_window_limit=self.category_window_limit,
            city_window_limit=self.city_window_limit,
            author_global_limit=self.author_global_limit,
            category_global_limit=self.category_global_limit,
            source_global_limits=self.source_global_limits,
        )

        return [
            Recommendation(
                item_id=item.candidate.item.item_id,
                score=item.score,
                recall_sources=item.candidate.recall_sources,
                task_scores=item.task_scores,
            )
            for item in selected
        ]


def filter_candidates(
    user: UserProfile,
    ranked_candidates: Iterable[RankedCandidate],
) -> list[RankedCandidate]:
    """Remove candidates that violate hard request-time filters.

    Typical online filters include already exposed content, blocked authors,
    deleted/private content, age restrictions, and risk-control blocks. This
    demo keeps the user-visible filters available in UserProfile.
    """

    return [
        item
        for item in ranked_candidates
        if item.candidate.item.item_id not in user.exposed_item_ids
        and item.candidate.item.author_id not in user.blocked_authors
    ]


def deduplicate_candidates(candidates: Iterable[RankedCandidate]) -> list[RankedCandidate]:
    """Keep only the highest-score candidate for each item_id.

    A video can be recalled by tag, two-tower, STAMP, or multi-interest channels
    at the same time. De-duplication avoids repeated exposure while keeping the
    strongest rank result for that item.
    """

    by_item: dict[str, RankedCandidate] = {}
    for item in candidates:
        item_id = item.candidate.item.item_id
        if item_id not in by_item or item.score > by_item[item_id].score:
            by_item[item_id] = item
    return sorted(by_item.values(), key=lambda item: item.score, reverse=True)


def apply_post_rank_formula(
    candidates: Iterable[RankedCandidate],
    weights: PostRankFusionWeights,
) -> list[RankedCandidate]:
    """Re-score candidates with a controllable post-rank product formula."""

    rescored = [
        replace(item, score=fuse_post_rank_score(item, weights))
        for item in candidates
    ]
    return sorted(rescored, key=lambda item: item.score, reverse=True)


def fuse_post_rank_score(item: RankedCandidate, weights: PostRankFusionWeights) -> float:
    """Fuse model score, recall score, item priors, and negative feedback.

    Formula:
        score = rank_score^w_rank * recall_score^w_recall * quality^w_quality
              * freshness^w_freshness * popularity^w_popularity
              * (1 - p_negative)^w_negative

    Use this after fine ranking when the online system needs a transparent
    formula layer for business knobs. Set a weight to 0 to disable that factor.
    """

    item_info = item.candidate.item
    negative_score = item.task_scores.get("negative", 0.0)
    return (
        bounded(item.score) ** weights.rank
        * bounded(item.candidate.recall_score) ** weights.recall
        * bounded(item_info.quality_score) ** weights.quality
        * bounded(item_info.freshness_score) ** weights.freshness
        * bounded(item_info.popularity_score) ** weights.popularity
        * (1.0 - bounded(negative_score)) ** weights.negative
    )


def diversity_rerank(
    candidates: Iterable[RankedCandidate],
    limit: int,
    diversity_lambda: float = 0.12,
    window_size: int = 10,
    author_window_limit: int = 1,
    category_window_limit: int = 4,
    city_window_limit: int | None = None,
    author_global_limit: int | None = None,
    category_global_limit: int | None = None,
    source_global_limits: Mapping[str, int] | None = None,
) -> list[RankedCandidate]:
    """Select results with MMR-style diversity and repetition constraints.

    The selection score is:
        rerank_score = rank_score - lambda * max_similarity(candidate, selected)

    Sliding-window limits are used for "打散", while global limits are used for
    page-level quota control. For example, author_window_limit=1 prevents two
    adjacent items from the same author when window_size is larger than one.
    """

    selected: list[RankedCandidate] = []
    remaining = sorted(candidates, key=lambda item: item.score, reverse=True)
    source_global_limits = dict(source_global_limits or {})
    while remaining and len(selected) < limit:
        next_item = max(
            remaining,
            key=lambda item: diversified_score(item, selected, diversity_lambda),
        )
        remaining.remove(next_item)
        if violates_constraints(
            next_item,
            selected,
            window_size=window_size,
            author_window_limit=author_window_limit,
            category_window_limit=category_window_limit,
            city_window_limit=city_window_limit,
            author_global_limit=author_global_limit,
            category_global_limit=category_global_limit,
            source_global_limits=source_global_limits,
        ):
            continue
        selected.append(next_item)
    return selected


def diversified_score(
    item: RankedCandidate,
    selected: list[RankedCandidate],
    diversity_lambda: float,
) -> float:
    """Penalize candidates that are similar to already selected items."""

    if not selected:
        return item.score
    max_similarity = max(item_similarity(item, chosen) for chosen in selected)
    return item.score - diversity_lambda * max_similarity


def violates_constraints(
    item: RankedCandidate,
    selected: list[RankedCandidate],
    window_size: int,
    author_window_limit: int,
    category_window_limit: int,
    city_window_limit: int | None = None,
    author_global_limit: int | None = None,
    category_global_limit: int | None = None,
    source_global_limits: Mapping[str, int] | None = None,
) -> bool:
    """Check whether adding a candidate breaks window or quota constraints."""

    return violates_window(
        item,
        selected,
        window_size,
        author_window_limit,
        category_window_limit,
        city_window_limit,
    ) or violates_global_quota(
        item,
        selected,
        author_global_limit,
        category_global_limit,
        source_global_limits or {},
    )


def violates_window(
    item: RankedCandidate,
    selected: list[RankedCandidate],
    window_size: int,
    author_window_limit: int,
    category_window_limit: int,
    city_window_limit: int | None = None,
) -> bool:
    """Check whether a candidate breaks sliding-window dispersion rules."""

    window = deque(selected[-window_size:], maxlen=window_size)
    authors = Counter(chosen.candidate.item.author_id for chosen in window)
    categories = Counter(chosen.candidate.item.category_id for chosen in window)
    cities = Counter(chosen.candidate.item.city_id for chosen in window)
    candidate_item = item.candidate.item
    violates_city = (
        city_window_limit is not None
        and cities[candidate_item.city_id] >= city_window_limit
    )
    return (
        authors[candidate_item.author_id] >= author_window_limit
        or categories[candidate_item.category_id] >= category_window_limit
        or violates_city
    )


def violates_global_quota(
    item: RankedCandidate,
    selected: list[RankedCandidate],
    author_global_limit: int | None = None,
    category_global_limit: int | None = None,
    source_global_limits: Mapping[str, int] | None = None,
) -> bool:
    """Check page-level quotas for author, category, and recall source."""

    candidate_item = item.candidate.item
    if author_global_limit is not None:
        author_count = sum(
            chosen.candidate.item.author_id == candidate_item.author_id
            for chosen in selected
        )
        if author_count >= author_global_limit:
            return True
    if category_global_limit is not None:
        category_count = sum(
            chosen.candidate.item.category_id == candidate_item.category_id
            for chosen in selected
        )
        if category_count >= category_global_limit:
            return True
    source_global_limits = source_global_limits or {}
    for source, limit in source_global_limits.items():
        if source in item.candidate.recall_sources:
            source_count = sum(
                source in chosen.candidate.recall_sources
                for chosen in selected
            )
            if source_count >= limit:
                return True
    return False


def item_similarity(left: RankedCandidate, right: RankedCandidate) -> float:
    """Estimate item similarity from author, category, city, tags, and embedding."""

    left_item = left.candidate.item
    right_item = right.candidate.item
    score = 0.0
    if left_item.author_id == right_item.author_id:
        score += 1.0
    if left_item.category_id == right_item.category_id:
        score += 0.45
    if left_item.city_id == right_item.city_id:
        score += 0.15
    score += 0.35 * jaccard(left_item.tags, right_item.tags)
    if left_item.embedding and right_item.embedding:
        score += 0.35 * max(0.0, cosine(left_item.embedding, right_item.embedding))
    return score


def jaccard(left: tuple[str, ...], right: tuple[str, ...]) -> float:
    """Return Jaccard similarity for two tag sets."""

    left_set = set(left)
    right_set = set(right)
    if not left_set or not right_set:
        return 0.0
    return len(left_set & right_set) / len(left_set | right_set)


def bounded(value: float) -> float:
    """Clamp a formula factor to avoid zeroing the whole product by accident."""

    return min(1.0 - 1.0e-6, max(1.0e-6, value))
