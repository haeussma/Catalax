"""Observation-noise models, shared by least squares, Bayesian inference and design.

A noise model gives the standard deviation of every observation, ``scale(prediction)``. ``optimize`` evaluates it on the measured data (fixed weights), ``HMC`` on the model prediction, the design code uses it to whiten sensitivities.

A noise model enters the Fisher information only by whitening sensitivities,
``W = noise.whiten(S, y)`` and ``F = W.T @ W``. This keeps ``F`` a sum of one
rank-1 term per observation, which only holds for diagonal (uncorrelated) noise;
correlated noise is deliberately not supported.

No distribution family is needed: for any location-scale family the information
is ``I_f / sigma**2 * S.T @ S`` with ``I_f`` a constant, so the tail shape scales
every design's score equally and never moves the optimum. It belongs in the
inference likelihood, not here.
"""

from __future__ import annotations

import dataclasses
from typing import Protocol, runtime_checkable

import equinox as eqx
import jax
import jax.numpy as jnp
import numpyro


@runtime_checkable
class NoiseModel(Protocol):
    """Whitens sensitivities by the per-point observation noise scale.

    Structural: any object with matching ``scale`` and ``whiten`` methods is a
    ``NoiseModel``, no subclassing needed. Only dataclass fields (equinox Modules)
    can hold priors for HMC to sample.
    """

    def scale(self, prediction: jax.Array) -> jax.Array:
        """Per-point noise standard deviation, shaped like ``prediction``."""
        ...

    def whiten(self, sensitivities: jax.Array, prediction: jax.Array) -> jax.Array:
        """Sensitivity rows divided by their own point's ``scale``."""
        ...


def _divide_by_scale(sensitivities: jax.Array, scale: jax.Array) -> jax.Array:
    """Divides sensitivities by a per-point scale broadcast over the last axis."""
    return sensitivities / scale[..., None]


def _is_prior(value) -> bool:
    # Duck-typed: importing catalax.mcmc here would be circular (catalax.mcmc imports this module).
    return hasattr(value, "_distribution_fun")


def prior_fields(noise) -> dict:
    """The fields of a noise model that are priors (to be sampled), by name.

    A noise model that is not a dataclass (structural ``NoiseModel``) has none: only
    dataclass fields (equinox Modules such as the built-in models) are recognised as
    priors. An array-valued field with a prior is sampled as one scalar site.
    """
    if not dataclasses.is_dataclass(noise):
        return {}
    return {
        field.name: getattr(noise, field.name)
        for field in dataclasses.fields(noise)
        if _is_prior(getattr(noise, field.name))
    }


def sample_noise(noise):
    """Inside a numpyro model: the noise model with every prior field replaced by a sample.

    Each prior is sampled as a site named after its field (``sigma``, ``cv``, ``floor``).
    A noise model without priors is returned unchanged.
    """
    priors = prior_fields(noise)
    if not priors:
        return noise
    values = [numpyro.sample(name, prior._distribution_fun()) for name, prior in priors.items()]
    return eqx.tree_at(lambda m: [getattr(m, name) for name in priors], noise, values)


class Homoskedastic(eqx.Module):
    """Constant additive noise, ``y ~ Normal(y_hat, sigma)``.

    ``sigma`` is a dynamic pytree leaf (not static), so it can carry a batch of
    values under ``vmap`` and never silently hits a stale ``jit`` cache entry.

    Any field may be a catalax.mcmc.priors prior instead of a number; HMC then
    samples it. The design code needs numbers.

    Attributes:
        sigma: Noise standard deviation in data units; a scalar, or one value per data point (an array broadcastable to the data).
    """

    sigma: jax.Array

    def __init__(self, sigma: float | jax.Array) -> None:
        """Args:
        sigma: Noise standard deviation in data units.
        """
        self.sigma = sigma if _is_prior(sigma) else jnp.asarray(sigma, dtype=float)

    def scale(self, prediction: jax.Array) -> jax.Array:
        """Returns ``sigma`` broadcast to the shape of ``prediction``.

        Args:
            prediction: Predicted observations; only the shape is used.

        Returns:
            Array shaped like ``prediction``.
        """
        return jnp.broadcast_to(self.sigma, prediction.shape)

    def whiten(self, sensitivities: jax.Array, prediction: jax.Array) -> jax.Array:
        """Divides every sensitivity by ``sigma``.

        Args:
            sensitivities: Shape ``prediction.shape + (n_parameters,)``.
            prediction: Predicted observations.

        Returns:
            Array shaped like ``sensitivities``.
        """
        return _divide_by_scale(sensitivities, self.scale(prediction))


class Proportional(eqx.Module):
    """Constant-plus-proportional noise, ``sd(y) = sqrt(floor**2 + (cv * y)**2)``.

    The typical HPLC model: error proportional to the measured amount, with a
    detection floor near zero. Two approximations are made: the variance term of
    the Gaussian information is dropped (its ratio to the mean term is
    ``2 * cv**2``, 0.5% at cv = 5%), and this is also the information of
    ``y ~ LogNormal(log y_hat, cv)`` in the limit ``floor -> 0``.

    Any field may be a catalax.mcmc.priors prior instead of a number; HMC then
    samples it. The design code needs numbers.

    Attributes:
        cv: Relative standard deviation of the proportional component.
        floor: Standard deviation floor in data units, strictly positive.
    """

    cv: jax.Array
    floor: jax.Array

    def __init__(self, cv: float | jax.Array, floor: float | jax.Array) -> None:
        """Args:
            cv: Relative standard deviation, e.g. 0.05 for 5%.
            floor: Standard deviation floor in data units.

        Raises:
            ValueError: If ``floor`` is not strictly positive.
        """
        if not _is_prior(floor):
            floor = jnp.asarray(floor, dtype=float)
            if not bool(jnp.all(floor > 0.0)):
                raise ValueError(
                    f"Proportional(floor=...) must be strictly positive, got {floor}. "
                    "At a point where the prediction is 0 (e.g. t = 0 with no initial "
                    "product) floor = 0 makes the noise scale 0 and whitening 0/0 = NaN."
                    " Use the assay's detection-limit standard deviation instead."
                )
        self.cv = cv if _is_prior(cv) else jnp.asarray(cv, dtype=float)
        self.floor = floor

    def scale(self, prediction: jax.Array) -> jax.Array:
        """Returns ``sqrt(floor**2 + (cv * prediction)**2)``.

        Args:
            prediction: Predicted observations.

        Returns:
            Array shaped like ``prediction``.
        """
        return jnp.sqrt(self.floor**2 + (self.cv * prediction) ** 2)

    def whiten(self, sensitivities: jax.Array, prediction: jax.Array) -> jax.Array:
        """Divides every sensitivity by the noise scale at its predicted value.

        Args:
            sensitivities: Shape ``prediction.shape + (n_parameters,)``.
            prediction: Predicted observations.

        Returns:
            Array shaped like ``sensitivities``.
        """
        return _divide_by_scale(sensitivities, self.scale(prediction))


def as_noise_model(noise: NoiseModel | float | jax.Array, fixed: bool = False) -> NoiseModel:
    """Coerces a bare noise standard deviation to ``Homoskedastic``.

    Args:
        noise: A ``NoiseModel``, or a noise standard deviation.
        fixed: Raise if any field is a prior (the design code needs numbers).

    Returns:
        ``noise`` unchanged if it already is a ``NoiseModel``, else
        ``Homoskedastic(noise)``.
    """
    model = noise if isinstance(noise, NoiseModel) else Homoskedastic(noise)
    if fixed and prior_fields(model):
        raise ValueError(
            f"This needs fixed noise values; got priors for {list(prior_fields(model))}. "
            "Priors are only sampled by HMC.run."
        )
    return model


__all__ = ["Homoskedastic", "NoiseModel", "Proportional", "as_noise_model", "prior_fields", "sample_noise"]
