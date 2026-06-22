from __future__ import annotations

from recsys.fine_rank_fusion import (
    VideoFineRankFusion,
    VideoFusionWeights,
    WatchValueConfig,
    duration_gain,
    fuse_video_fine_rank_score,
)
from recsys.rank import FineRanker
from recsys.types import Candidate, Item, RankedCandidate


def test_video_fine_rank_fusion_rewards_play_watch_and_interaction() -> None:
    strong = ranked(
        Item("strong", "a1", "sports", "shanghai", duration_seconds=45, quality_score=0.9, freshness_score=0.9),
        {
            "play": 0.85,
            "finish": 0.70,
            "long_play": 0.75,
            "short_play": 0.05,
            "like": 0.20,
            "share": 0.08,
            "comment": 0.06,
            "follow": 0.04,
            "favorite": 0.12,
            "negative": 0.02,
        },
    )
    weak = ranked(
        Item("weak", "a2", "sports", "shanghai", duration_seconds=45, quality_score=0.9, freshness_score=0.9),
        {
            "play": 0.65,
            "finish": 0.45,
            "long_play": 0.35,
            "short_play": 0.25,
            "like": 0.05,
            "share": 0.01,
            "comment": 0.01,
            "follow": 0.01,
            "favorite": 0.02,
            "negative": 0.10,
        },
    )

    assert fuse_video_fine_rank_score(strong) > fuse_video_fine_rank_score(weak)


def test_video_fine_rank_fusion_penalizes_short_play_and_negative_feedback() -> None:
    base_scores = {
        "play": 0.85,
        "finish": 0.70,
        "long_play": 0.75,
        "like": 0.20,
        "share": 0.08,
        "comment": 0.06,
        "follow": 0.04,
        "favorite": 0.12,
    }
    good = ranked(
        Item("good", "a1", "sports", "shanghai", duration_seconds=45),
        base_scores | {"short_play": 0.05, "negative": 0.02},
    )
    bad = ranked(
        Item("bad", "a2", "sports", "shanghai", duration_seconds=45),
        base_scores | {"short_play": 0.70, "negative": 0.60},
    )

    assert fuse_video_fine_rank_score(good) > fuse_video_fine_rank_score(bad)


def test_duration_gain_saturates_long_video_value() -> None:
    config = WatchValueConfig(duration_cap_seconds=120.0, duration_reference_seconds=30.0)

    assert duration_gain(600.0, config) == duration_gain(120.0, config)
    assert duration_gain(120.0, config) > duration_gain(30.0, config)


def test_fine_ranker_can_use_independent_video_fusion() -> None:
    ranker = FineRanker(
        fusion=VideoFineRankFusion(
            weights=VideoFusionWeights(
                play=0.4,
                watch=0.5,
                interaction=0.0,
                short_play_penalty=0.3,
                negative_penalty=0.5,
                quality=0.0,
                freshness=0.0,
            )
        )
    )
    candidates = [
        ranked(
            Item("v1", "a1", "sports", "shanghai", duration_seconds=30),
            {"play": 0.7, "finish": 0.5, "long_play": 0.4, "short_play": 0.5, "negative": 0.3},
        ),
        ranked(
            Item("v2", "a2", "sports", "shanghai", duration_seconds=30),
            {"play": 0.8, "finish": 0.7, "long_play": 0.6, "short_play": 0.1, "negative": 0.05},
        ),
    ]

    ranked_candidates = ranker.rank(candidates, limit=2)

    assert [item.candidate.item.item_id for item in ranked_candidates] == ["v2", "v1"]


def ranked(item: Item, task_scores: dict[str, float]) -> RankedCandidate:
    return RankedCandidate(
        candidate=Candidate(item=item, recall_score=0.8, recall_sources=("two_tower",)),
        task_scores=task_scores,
        score=0.0,
    )
