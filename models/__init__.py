from __future__ import annotations

__all__ = [
    "AttentionPooling",
    "B0ModelConfig",
    "B0UtteranceClassifier",
    "MALMemoryModule",
    "WavLMSERBaseline",
    "WavLMSERBaselineConfig",
    "WavLM_MALSerModel",
    "WavLMTIMConfig",
    "WavLMTIMSerModel",
    "WavLMMALConfig",
    "TemporalFeatureEncoder",
    "TIMMemoryModule",
    "build_b0_model",
    "build_wavlm_mal_ser_model",
    "build_wavlm_ser_baseline",
    "build_wavlm_tim_ser_model",
]


def __getattr__(name: str):
    if name in {"AttentionPooling", "B0ModelConfig", "B0UtteranceClassifier", "build_b0_model"}:
        from models import wavlm

        return getattr(wavlm, name)
    if name in {"WavLMSERBaseline", "WavLMSERBaselineConfig", "build_wavlm_ser_baseline"}:
        from models import wavlm_baseline

        return getattr(wavlm_baseline, name)
    if name in {"MALMemoryModule", "WavLM_MALSerModel", "WavLMMALConfig", "build_wavlm_mal_ser_model"}:
        from models import wavlm_mal

        return getattr(wavlm_mal, name)
    if name in {
        "TemporalFeatureEncoder",
        "TIMMemoryModule",
        "WavLMTIMConfig",
        "WavLMTIMSerModel",
        "build_wavlm_tim_ser_model",
    }:
        from models import wavlm_tim

        return getattr(wavlm_tim, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
