"""GapRAG package."""

__all__ = ["GapRAGPipeline"]


def __getattr__(name: str):
    if name == "GapRAGPipeline":
        from .pipeline import GapRAGPipeline

        return GapRAGPipeline
    raise AttributeError(f"module 'gaprag' has no attribute {name!r}")
