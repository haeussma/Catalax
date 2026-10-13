"""Moved to ``catalax.noise``; kept so existing imports work."""

from catalax.noise import Homoskedastic, NoiseModel, Proportional, as_noise_model

__all__ = ["Homoskedastic", "NoiseModel", "Proportional", "as_noise_model"]
