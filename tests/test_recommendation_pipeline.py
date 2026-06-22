from __future__ import annotations

from recsys.pipeline import RecommendationPipeline
from recsys.postprocess import PostProcessor
from recsys.recall import TagRecall, VectorRecall, merge_candidates
from recsys.types import Item, UserAction, UserProfile


NOW = 1_700_000_000.0


def test_tag_recall_matches_user_history_fields() -> None:
    items = [
        Item("v1", "a1", "sports", "shanghai", ("basketball",), 20, 1.0, 1.0),
        Item("v2", "a2", "food", "beijing", ("cooking",), 20, 1.0, 1.0),
    ]
    user = UserProfile(
        "u1",
        actions=(
            UserAction("old", "finish", NOW - 3600, "a1", "sports", "shanghai", ("basketball",), 20, 20),
        ),
    )

    recalled = TagRecall(items).recall(user, limit=10, now=NOW)

    assert [candidate.item.item_id for candidate in recalled][0] == "v1"
    assert "tag:author" in recalled[0].recall_sources


def test_vector_recall_uses_behavior_embeddings() -> None:
    items = [
        Item("v1", "a1", "sports", "shanghai", embedding=(1.0, 0.0)),
        Item("v2", "a2", "food", "beijing", embedding=(0.0, 1.0)),
        Item("v3", "a3", "sports", "shanghai", embedding=(0.9, 0.1)),
    ]
    user = UserProfile("u1", actions=(UserAction("v1", "finish", NOW, duration_seconds=10, play_time_seconds=10),))

    recalled = VectorRecall(items).recall(user, limit=3, now=NOW)

    assert recalled[0].item.item_id == "v1"
    assert recalled[1].item.item_id == "v3"


def test_merge_candidates_keeps_best_score_and_sources() -> None:
    item = Item("v1", "a1", "sports", "shanghai")
    merged = merge_candidates(
        [
            [candidate(item, 0.2, "tag")],
            [candidate(item, 0.8, "two_tower")],
        ],
        limit=10,
    )

    assert merged[0].recall_score == 0.8
    assert merged[0].recall_sources == ("tag", "two_tower")


def test_pipeline_filters_exposed_and_controls_author_diversity() -> None:
    items = [
        Item("v1", "a1", "sports", "shanghai", ("basketball",), 18, 0.99, 1.0, 1.0, (1.0, 0.0)),
        Item("v2", "a1", "sports", "shanghai", ("basketball",), 19, 0.98, 1.0, 1.0, (0.9, 0.1)),
        Item("v3", "a2", "sports", "shanghai", ("football",), 22, 0.92, 1.0, 0.8, (0.8, 0.2)),
        Item("v4", "a3", "food", "beijing", ("cooking",), 30, 0.86, 0.9, 0.7, (0.0, 1.0)),
    ]
    user = UserProfile(
        "u1",
        actions=(
            UserAction("v1", "finish", NOW - 100, "a1", "sports", "shanghai", ("basketball",), 18, 18),
        ),
        exposed_item_ids=frozenset({"v1"}),
    )
    pipeline = RecommendationPipeline(
        items,
        post_processor=PostProcessor(author_window_limit=1, category_window_limit=4),
    )

    recommendations = pipeline.recommend(user, NOW, final_limit=3)

    item_ids = [item.item_id for item in recommendations]
    authors = [next(item.author_id for item in items if item.item_id == item_id) for item_id in item_ids]
    assert "v1" not in item_ids
    assert len(authors) == len(set(authors))
    assert recommendations


def candidate(item: Item, score: float, source: str):
    from recsys.types import Candidate

    return Candidate(item=item, recall_score=score, recall_sources=(source,))
