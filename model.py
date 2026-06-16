from __future__ import annotations

from b0_model import AttentionPooling, B0ModelConfig, B0UtteranceClassifier, build_b0_model

SERModel = B0UtteranceClassifier

__all__ = [
    "AttentionPooling",
    "B0ModelConfig",
    "B0UtteranceClassifier",
    "SERModel",
    "build_b0_model",
]
