"""GapVerify package."""

__all__ = ["GapVerifyPipeline"]


def __getattr__(name: str):
    if name == "GapVerifyPipeline":
        from .pipeline import GapVerifyPipeline

        return GapVerifyPipeline
    raise AttributeError(f"module 'gapverify' has no attribute {name!r}")
