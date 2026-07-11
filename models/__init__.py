from __future__ import annotations

__all__ = [
    "AttentionPooling",
    "B0ModelConfig",
    "B0UtteranceClassifier",
    "SERBaseline",
    "SERBaselineConfig",
    "CDIMConfig",
    "CDIMMemoryModule",
    "CDIMSerModel",
    "TemporalFeatureEncoder",
    "build_b0_model",
    "build_ser_baseline",
    "build_cdim_ser_model",
]


def __getattr__(name: str):
    if name in {"AttentionPooling", "B0ModelConfig", "B0UtteranceClassifier", "build_b0_model"}:
        from models import modules

        return getattr(modules, name)
    if name in {"SERBaseline", "SERBaselineConfig", "build_ser_baseline"}:
        from models import baseline

        return getattr(baseline, name)
    if name in {"TemporalFeatureEncoder", "CDIMMemoryModule", "CDIMConfig", "CDIMSerModel", "build_cdim_ser_model"}:
        from models import cdim

        return getattr(cdim, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
