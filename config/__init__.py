"""
Single entry point for configuration.

Usage:
    from config import settings, thresholds, model_registry
"""
from config.pipeline   import settings
from config.thresholds import thresholds
from config.models     import build_registry

# model_registry reads the ENV flag from settings.
# No other module reads ENV directly.
model_registry = build_registry(env=settings.env)

__all__ = ["settings", "thresholds", "model_registry"]
