from __future__ import annotations

from recsys.postprocess import (
    PostProcessor,
    PostRankFusionWeights,
    apply_post_rank_formula,
    deduplicate_candidates,
    diversity_rerank,
)
from recsys.types import Candidate, Item, RankedCandidate, UserProfile


def test_post_rank_formula_penalizes_negative_feedback() -> None:
    safe_item = ranked(
        Item("safe", "a1", "sports", "shanghai", quality_score=0.9, freshness_score=0.9),
        score=0.8,
        task_scores={"negative": 0.05},
    )
    risky_item = ranked(
        Item("risky", "a2", "sports", "shanghai", quality_score=0.9, freshness_score=0.9),
        score=0.8,
        task_scores={"negative": 0.8},
    )

    rescored = apply_post_rank_formula(
        [risky_item, safe_item],
        PostRankFusionWeights(rank=1.0, negative=1.0),
    )

    assert [item.candidate.item.item_id for item in rescored] == ["safe", "risky"]
    assert rescored[0].score > rescored[1].score


def test_deduplicate_candidates_keeps_highest_rank_score() -> None:
    low_score = ranked(Item("v1", "a1", "sports", "shanghai"), score=0.2)
    high_score = ranked(Item("v1", "a1", "sports", "shanghai"), score=0.9)

    deduplicated = deduplicate_candidates([low_score, high_score])

    assert len(deduplicated) == 1
    assert deduplicated[0].score == 0.9


def test_diversity_rerank_applies_window_and_global_quota() -> None:
    candidates = [
        ranked(Item("v1", "a1", "sports", "shanghai"), score=0.99),
        ranked(Item("v2", "a1", "sports", "shanghai"), score=0.98),
        ranked(Item("v3", "a2", "sports", "shanghai"), score=0.97),
        ranked(Item("v4", "a3", "food", "beijing"), score=0.96),
    ]

    selected = diversity_rerank(
        candidates,
        limit=3,
        diversity_lambda=0.0,
        window_size=10,
        author_window_limit=1,
        category_window_limit=10,
        category_global_limit=2,
    )

    assert [item.candidate.item.item_id for item in selected] == ["v1", "v3", "v4"]


def test_post_processor_filters_exposed_and_applies_formula() -> None:
    user = UserProfile("u1", exposed_item_ids=frozenset({"v1"}))
    processor = PostProcessor(
        formula_weights=PostRankFusionWeights(rank=1.0, quality=1.0),
        author_window_limit=2,
    )
    candidates = [
        ranked(Item("v1", "a1", "sports", "shanghai", quality_score=1.0), score=0.99),
        ranked(Item("v2", "a2", "sports", "shanghai", quality_score=0.5), score=0.8),
        ranked(Item("v3", "a3", "sports", "shanghai", quality_score=0.9), score=0.7),
    ]

    recommendations = processor.process(user, candidates, limit=2)

    assert [item.item_id for item in recommendations] == ["v3", "v2"]


def ranked(
    item: Item,
    score: float,
    task_scores: dict[str, float] | None = None,
    recall_score: float = 0.8,
) -> RankedCandidate:
    return RankedCandidate(
        candidate=Candidate(item=item, recall_score=recall_score, recall_sources=("two_tower",)),
        task_scores=task_scores or {},
        score=score,
    )
