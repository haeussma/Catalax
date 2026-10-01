"""Designing against posterior samples instead of the priors, 16 draws."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import catalax as ctx
import catalax.doe as cdoe
import catalax.mcmc as cmm
from catalax.doe.evaluate import _prepare_efficiency

pytestmark = pytest.mark.usefixtures("x64")

TIMES = [10.0, 40.0, 80.0, 150.0, 300.0]
NOISE = cdoe.Proportional(cv=0.05, floor=0.5)
MEDIANS = {"k_m": 100.0, "kcat": 10.0}
LOG_SD = 0.5


def _mm_model() -> ctx.Model:
    """Michaelis-Menten with ``e`` as an unobservable zero-rate state."""
    model = ctx.Model(name="Michaelis-Menten")
    model.add_state("s, p, e")
    model.add_ode("e", "0", observable=False)
    model.add_ode("s", "-kcat * e * s / (k_m + s)", observable=True)
    model.add_ode("p", "kcat * e * s / (k_m + s)", observable=False)
    for name, median in MEDIANS.items():
        model.parameters[name].value = median
        model.parameters[name].prior = cmm.priors.LogNormal(mu=median, sigma=LOG_SD)
    return model


def _design(model: ctx.Model) -> ctx.Dataset:
    design = ctx.Dataset.from_model(model)
    for s0 in (50.0, 400.0):
        design.add_initial(time=TIMES, s=s0, p=0.0, e=0.1)
    return design


def _lognormal_samples(key: jax.Array, n: int, corr: float = 0.0) -> dict:
    """Samples of the priors' LogNormals, optionally correlated in log space."""
    cov = LOG_SD**2 * jnp.array([[1.0, corr], [corr, 1.0]])
    logs = jax.random.multivariate_normal(key, jnp.zeros(2), cov, (n,))
    names = list(MEDIANS)
    return {n: MEDIANS[n] * jnp.exp(logs[:, i]) for i, n in enumerate(names)}


def _per_draw(model, design, key, belief=None):
    fn, order, (y0s, constants, _) = _prepare_efficiency(
        model, design, NOISE, key=key, n_draws=16, config=None, belief=belief
    )
    e, valid = fn(y0s, constants)
    assert bool(valid.all())
    return np.asarray(e), order


def test_lognormal_samples_match_the_lognormal_prior():
    model = _mm_model()
    design = _design(model)
    belief = _lognormal_samples(jax.random.PRNGKey(0), 20_000)
    belief["sigma"] = jnp.ones(20_000)  # extra keys are ignored

    prior, order = _per_draw(model, design, jax.random.PRNGKey(1))
    samples, _ = _per_draw(model, design, jax.random.PRNGKey(2), belief)

    # Different draws, so the means agree only up to Monte Carlo error.
    se = np.sqrt(prior.var(0, ddof=1) / 16 + samples.var(0, ddof=1) / 16)
    gap = np.abs(prior.mean(0) - samples.mean(0))
    assert np.all(gap < 3 * se), dict(zip(order, zip(gap, se)))


def test_lognormal_samples_match_the_prior_at_the_same_thetas():
    # The mean check above has little power: a belief with log sd 0.8 instead of
    # 0.5 passes it (gap 0.26 SE). At the same thetas only Sigma_0 differs: the
    # estimate from 20k samples moves e by 0.0024, the 0.8 belief by 0.14.
    model = _mm_model()
    design = _design(model)
    belief = _lognormal_samples(jax.random.PRNGKey(0), 20_000)
    key = jax.random.PRNGKey(1)
    fn_prior, _, (y0s, constants, _) = _prepare_efficiency(
        model, design, NOISE, key=key, n_draws=16, config=None
    )
    fn_belief, _, _ = _prepare_efficiency(
        model, design, NOISE, key=key, n_draws=16, config=None, belief=belief
    )
    e_prior, _ = fn_prior(y0s, constants)
    e_belief, _ = fn_belief.func(fn_prior.args[0], y0s, constants)
    assert np.abs(np.asarray(e_prior - e_belief)).max() < 0.01


def test_correlated_belief_uses_the_off_diagonal():
    model = _mm_model()
    design = _design(model)
    key = jax.random.PRNGKey(0)
    correlated = _lognormal_samples(key, 4_000, corr=-0.9)
    # Same marginals, correlation destroyed.
    shuffled = {
        name: jax.random.permutation(k, values)
        for (name, values), k in zip(correlated.items(), jax.random.split(key))
    }
    fn_corr, _, (y0s, constants, _) = _prepare_efficiency(
        model, design, NOISE, key=key, n_draws=16, config=None, belief=correlated
    )
    fn_shuf, _, _ = _prepare_efficiency(
        model, design, NOISE, key=key, n_draws=16, config=None, belief=shuffled
    )
    # Score both covariances at the same thetas, so only Sigma_0 differs.
    thetas = fn_corr.args[0]
    e_corr, _ = fn_corr.func(thetas, y0s, constants)
    e_shuf, _ = fn_shuf.func(thetas, y0s, constants)
    assert np.abs(np.asarray(e_corr - e_shuf)).max() > 0.01


@pytest.mark.parametrize(
    "edit, match",
    [
        (lambda b: b.pop("kcat"), r"no samples for the free parameters \['kcat'\]"),
        (lambda b: b.__setitem__("kcat", b["kcat"].at[3].set(0.0)), "kcat.*> 0"),
        (lambda b: b.__setitem__("kcat", b["kcat"].at[3].set(jnp.nan)), "kcat"),
        (lambda b: b.__setitem__("kcat", b["kcat"][:-1]), "different lengths"),
        (lambda b: b.__setitem__("kcat", b["kcat"].reshape(2, -1)), "kcat.*shape"),
    ],
    ids=["missing", "non-positive", "nan", "ragged", "2-d"],
)
def test_invalid_belief_names_the_problem(edit, match):
    model = _mm_model()
    belief = _lognormal_samples(jax.random.PRNGKey(0), 100)
    edit(belief)
    with pytest.raises(ValueError, match=match):
        cdoe.evaluate_design(
            model, _design(model), NOISE, key=jax.random.PRNGKey(1), belief=belief
        )


def test_too_few_samples_for_the_covariance():
    model = _mm_model()
    belief = _lognormal_samples(jax.random.PRNGKey(0), 2)
    with pytest.raises(ValueError, match="2 samples for 2 free parameters"):
        cdoe.evaluate_design(
            model, _design(model), NOISE, key=jax.random.PRNGKey(1), belief=belief
        )


def test_optimize_design_with_belief_is_deterministic_by_key():
    model = _mm_model()
    belief = _lognormal_samples(jax.random.PRNGKey(0), 1_000, corr=-0.9)
    search = {
        "n_arms": 2,
        "times": TIMES,
        "n_restarts": 2,
        "n_steps": 20,
        "n_draws": 16,
        "n_rank_draws": 16,
        "belief": belief,
    }
    spec = {"s": (10.0, 1000.0), "p": (0.0, 0.0), "e": (0.1, 0.1)}
    runs = [
        cdoe.optimize_design(model, spec, NOISE, key=jax.random.PRNGKey(3), **search)
        for _ in range(2)
    ]
    first, second = (
        [m.initial_conditions["s"] for m in run.dataset.measurements] for run in runs
    )
    assert first == second
    assert runs[0].restart_scores == runs[1].restart_scores
    assert runs[0].report == runs[1].report
    assert runs[0].report.n_valid == 16
