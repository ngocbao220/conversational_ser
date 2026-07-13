from __future__ import annotations

__all__ = [
    "AttentiveStatisticsPooling",
    "SERBaseline",
    "SERBaselineConfig",
    "MeanEmbeddingBaseline",
    "CDMConfig",
    "CDMMemoryModule",
    "CDMSerModel",
    "CDIMConfig",
    "CDIMMemoryModule",
    "CDIMSerModel",
    "TemporalFeatureEncoder",
    "build_ser_baseline",
    "build_mean_embedding_baseline",
    "build_cdm_ser_model",
    "build_cdim_ser_model",
]


def __getattr__(name: str):
    if name in {
        "AttentiveStatisticsPooling",
        "SERBaseline",
        "SERBaselineConfig",
        "MeanEmbeddingBaseline",
        "build_ser_baseline",
        "build_mean_embedding_baseline",
    }:
        from models import baseline

        return getattr(baseline, name)
    if name in {"CDMConfig", "CDMMemoryModule", "CDMSerModel", "build_cdm_ser_model"}:
        from models import cdm

        return getattr(cdm, name)
    if name in {
        "TemporalFeatureEncoder",
        "CDIMMemoryModule",
        "CDIMConfig",
        "CDIMSerModel",
        "build_cdim_ser_model",
    }:
        from models import cdim

        return getattr(cdim, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
