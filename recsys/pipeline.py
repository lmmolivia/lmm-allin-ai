from __future__ import annotations

from collections.abc import Iterable

from recsys.postprocess import PostProcessor
from recsys.rank import CoarseRanker, FineRanker
from recsys.recall import MultiInterestVectorRecall, TagRecall, VectorRecall, merge_candidates
from recsys.types import Candidate, Item, Recommendation, UserProfile


class RecommendationPipeline:
    """End-to-end orchestration of recall, ranking, and post-processing."""

    def __init__(
        self,
        items: Iterable[Item],
        recallers: Iterable[object] | None = None,
        coarse_ranker: CoarseRanker | None = None,
        fine_ranker: FineRanker | None = None,
        post_processor: PostProcessor | None = None,
    ) -> None:
        """Wire default components while allowing each stage to be replaced."""

        self.items = tuple(items)
        self.recallers = tuple(
            recallers
            if recallers is not None
            else (
                TagRecall(self.items),
                VectorRecall(self.items, name="two_tower"),
                VectorRecall(self.items, name="stamp", history_window=5),
                MultiInterestVectorRecall(self.items),
            )
        )
        self.coarse_ranker = coarse_ranker or CoarseRanker()
        self.fine_ranker = fine_ranker or FineRanker()
        self.post_processor = post_processor or PostProcessor()

    def recommend(
        self,
        user: UserProfile,
        now: float,
        recall_limit_per_source: int = 500,
        merged_limit: int = 3000,
        coarse_limit: int = 500,
        fine_limit: int = 200,
        final_limit: int = 20,
    ) -> list[Recommendation]:
        """Run the full recommendation flow for one user request."""

        recalled = [
            recaller.recall(user, recall_limit_per_source, now)
            for recaller in self.recallers
        ]
        merged = merge_candidates(recalled, merged_limit)
        coarse = self.coarse_ranker.rank(user, merged, coarse_limit, now)
        fine = self.fine_ranker.rank(coarse, fine_limit)
        return self.post_processor.process(user, fine, final_limit)


def explain_candidates(candidates: Iterable[Candidate]) -> list[dict[str, object]]:
    """Convert candidates to simple dictionaries for debugging recall output."""

    return [
        {
            "item_id": candidate.item.item_id,
            "recall_score": candidate.recall_score,
            "recall_sources": candidate.recall_sources,
        }
        for candidate in candidates
    ]
