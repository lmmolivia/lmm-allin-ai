from __future__ import annotations

import sys
import time
from pathlib import Path

if __package__ in (None, ""):
    sys.path.append(str(Path(__file__).resolve().parents[1]))

from recsys.pipeline import RecommendationPipeline
from recsys.types import Item, UserAction, UserProfile


def build_demo_items() -> list[Item]:
    """Create a tiny item pool with metadata and toy embeddings."""

    return [
        Item("v1", "a1", "sports", "shanghai", ("basketball",), 18, 0.95, 0.9, 0.8, (0.9, 0.1, 0.0)),
        Item("v2", "a2", "sports", "shanghai", ("football",), 75, 0.90, 0.8, 0.7, (0.8, 0.2, 0.0)),
        Item("v3", "a3", "food", "beijing", ("cooking",), 30, 0.86, 0.9, 0.6, (0.0, 0.9, 0.1)),
        Item("v4", "a4", "travel", "chengdu", ("citywalk",), 45, 0.88, 1.0, 0.5, (0.1, 0.1, 0.9)),
        Item("v5", "a5", "sports", "hangzhou", ("basketball", "training"), 22, 0.91, 0.7, 0.4, (0.85, 0.1, 0.05)),
    ]


def build_demo_user(now: float) -> UserProfile:
    """Create one demo user with recent sports-video behavior history."""

    return UserProfile(
        user_id="u1",
        actions=(
            UserAction("v1", "finish", now - 3600, "a1", "sports", "shanghai", ("basketball",), 18, 18),
            UserAction("v2", "long_play", now - 7200, "a2", "sports", "shanghai", ("football",), 60, 75),
        ),
        exposed_item_ids=frozenset({"v1"}),
        features={"activity_level": 0.8},
    )


if __name__ == "__main__":
    current_time = time.time()
    pipeline = RecommendationPipeline(build_demo_items())
    for recommendation in pipeline.recommend(build_demo_user(current_time), current_time, final_limit=3):
        print(recommendation)
