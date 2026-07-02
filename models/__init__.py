from __future__ import annotations

__all__ = [
    "AttentionPooling",
    "B0ModelConfig",
    "B0UtteranceClassifier",
    "CDMMemoryModule",
    "WavLMSERBaseline",
    "WavLMSERBaselineConfig",
    "WavLM_CDMSerModel",
    "WavLMCIMConfig",
    "WavLMDualBranchCIMConfig",
    "WavLMDualBranchCIMSerModel",
    "WavLMCIMSerModel",
    "WavLMCDMConfig",
    "DialogueMemoryBranch",
    "TemporalInteractionEncoder",
    "TemporalMemoryBranch",
    "TemporalFeatureEncoder",
    "CIMMemoryModule",
    "build_b0_model",
    "build_wavlm_dual_branch_cim_ser_model",
    "build_wavlm_cdm_ser_model",
    "build_wavlm_ser_baseline",
    "build_wavlm_cim_ser_model",
]


def __getattr__(name: str):
    if name in {"AttentionPooling", "B0ModelConfig", "B0UtteranceClassifier", "build_b0_model"}:
        from models import wavlm

        return getattr(wavlm, name)
    if name in {"WavLMSERBaseline", "WavLMSERBaselineConfig", "build_wavlm_ser_baseline"}:
        from models import wavlm_baseline

        return getattr(wavlm_baseline, name)
    if name in {"CDMMemoryModule", "WavLM_CDMSerModel", "WavLMCDMConfig", "build_wavlm_cdm_ser_model"}:
        from models import wavlm_cdm

        return getattr(wavlm_cdm, name)
    if name in {
        "DialogueMemoryBranch",
        "TemporalInteractionEncoder",
        "TemporalMemoryBranch",
        "WavLMDualBranchCIMConfig",
        "WavLMDualBranchCIMSerModel",
        "build_wavlm_dual_branch_cim_ser_model",
    }:
        from models import wavlm_dual_branch_cim

        return getattr(wavlm_dual_branch_cim, name)
    if name in {
        "TemporalFeatureEncoder",
        "CIMMemoryModule",
        "WavLMCIMConfig",
        "WavLMCIMSerModel",
        "build_wavlm_cim_ser_model",
    }:
        from models import wavlm_cim

        return getattr(wavlm_cim, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
