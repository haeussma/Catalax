"""Numpyro distributions of a model's parameter priors."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpyro.distributions as dist
from numpyro.distributions import constraints

if TYPE_CHECKING:
    from catalax.model.model import Model


def _extract_priors(model: Model, names: list[str]) -> list[dist.Distribution]:
    """Numpyro distributions of the named parameters' priors.

    Raises:
        ValueError: If a prior is missing or its support is not strictly positive.
    """
    priors = []
    for name in names:
        prior = model.parameters[name].prior
        if prior is None:
            raise ValueError(
                f"Parameter '{name}' has no prior. Set one, e.g. "
                f"model.parameters['{name}'].prior = "
                "catalax.mcmc.priors.Uniform(low=..., high=...), or mark it "
                "constant=True to exclude it from the analysis."
            )
        distribution = prior._distribution_fun()
        support = distribution.support
        lower = getattr(support, "lower_bound", None)
        if support is not constraints.positive and not (
            lower is not None and lower > 0
        ):
            raise ValueError(
                f"The prior of parameter '{name}' ({type(prior).__name__}) allows "
                "values <= 0, but sensitivities are taken in log parameters. Use a prior "
                "with strictly positive support, e.g. Uniform with low > 0, "
                "LogNormal or LogUniform."
            )
        priors.append(distribution)
    return priors
