import jax
import jax.numpy as jnp
import numpy as np
import pytest

import catalax as ctx
import catalax.doe as cdoe
import catalax.mcmc as cmm
from catalax.doe.evaluate import _prior_log_variance

pytestmark = pytest.mark.usefixtures("x64")

TIMES = [10.0, 40.0, 80.0, 150.0, 300.0]


def _mm_model() -> ctx.Model:
    model = ctx.Model(name="Michaelis-Menten")
    model.add_state("s, p")
    model.add_constant("e")
    model.add_ode("s", "-kcat * e * s / (k_m + s)")
    model.add_ode("p", "kcat * e * s / (k_m + s)", observable=False)
    model.parameters["kcat"].value = 10.0
    model.parameters["kcat"].prior = cmm.priors.Uniform(low=5.0, high=20.0)
    model.parameters["k_m"].value = 100.0
    model.parameters["k_m"].prior = cmm.priors.Uniform(low=50.0, high=200.0)
    return model


def _design(model: ctx.Model, arms, times=TIMES) -> ctx.Dataset:
    design = ctx.Dataset.from_model(model)
    for s0 in arms:
        design.add_initial(time=times, s=s0, p=0.0, e=0.1)
    return design


def test_fisher_is_symmetric_psd():
    model = _mm_model()
    info = cdoe.fisher_information(model, _design(model, [50.0, 400.0]), 1.0)
    F = np.asarray(info.matrix)
    assert info.valid
    assert info.parameter_order == ["k_m", "kcat"]
    np.testing.assert_array_equal(F, F.T)
    assert np.linalg.eigvalsh(F).min() > 0


def test_homoskedastic_scales_as_inverse_variance():
    model = _mm_model()
    design = _design(model, [50.0, 400.0])
    scaled = [
        np.asarray(cdoe.fisher_information(model, design, sigma).matrix) * sigma**2
        for sigma in (0.1, 1.0, 7.0)
    ]
    np.testing.assert_allclose(scaled[0], scaled[1], rtol=1e-12)
    np.testing.assert_allclose(scaled[2], scaled[1], rtol=1e-12)


def test_constant_parameter_is_excluded_and_uses_its_value():
    model = _mm_model()
    design = _design(model, [50.0, 400.0])
    full = cdoe.fisher_information(model, design, 1.0)

    model.parameters["kcat"].constant = True
    reduced = cdoe.fisher_information(model, design, 1.0)

    assert reduced.parameter_order == ["k_m"]
    np.testing.assert_allclose(reduced.matrix, full.matrix[:1, :1], rtol=1e-12)


def test_failed_solve_is_invalid_but_finite():
    model = _mm_model()
    config = ctx.SimulationConfig(t1=300.0, max_steps=2, throw=False)
    info = cdoe.fisher_information(
        model, _design(model, [50.0, 400.0]), 1.0, config=config
    )
    assert not info.valid
    assert np.all(np.isfinite(info.matrix))


def test_two_observables_sum():
    model = _mm_model()
    design = _design(model, [50.0, 400.0])
    noise = cdoe.Proportional(cv=0.05, floor=0.5)

    s_only = cdoe.fisher_information(model, design, noise).matrix
    model.odes["p"].observable = True
    both = cdoe.fisher_information(model, design, noise).matrix
    model.odes["s"].observable = False
    p_only = cdoe.fisher_information(model, design, noise).matrix

    np.testing.assert_allclose(both, s_only + p_only, rtol=1e-12)


def test_proportional_requires_positive_floor():
    with pytest.raises(ValueError, match="floor"):
        cdoe.Proportional(cv=0.05, floor=0.0)


@pytest.mark.parametrize(
    "prior",
    [
        cmm.priors.Normal(mu=10.0, sigma=1.0),
        cmm.priors.Uniform(low=0.0, high=20.0),
        cmm.priors.TruncatedNormal(mu=10.0, sigma=1.0, low=-1.0, high=20.0),
    ],
)
def test_prior_without_positive_support_is_rejected(prior):
    model = _mm_model()
    model.parameters["kcat"].prior = prior
    with pytest.raises(ValueError, match="'kcat'"):
        cdoe.evaluate_design(
            model, _design(model, [50.0]), 1.0, key=jax.random.PRNGKey(0)
        )


def test_missing_prior_is_rejected():
    model = _mm_model()
    model.parameters["k_m"].prior = None
    with pytest.raises(ValueError, match="'k_m' has no prior"):
        cdoe.evaluate_design(
            model, _design(model, [50.0]), 1.0, key=jax.random.PRNGKey(0)
        )


def test_efficiency_in_unit_interval_and_monotone_in_design():
    model = _mm_model()
    key = jax.random.PRNGKey(1)
    small = cdoe.evaluate_design(
        model, _design(model, [50.0], TIMES[:3]), 1.0, key=key, n_draws=32
    )
    large = cdoe.evaluate_design(
        model, _design(model, [50.0, 400.0], TIMES[:3]), 1.0, key=key, n_draws=32
    )

    for report in (small, large):
        assert report.n_valid == report.n_draws == 32
        assert report.parameter_order == ["k_m", "kcat"]
        assert all(0.0 <= e <= 1.0 for e in report.efficiency.values())
        assert 0.0 <= report.maximin <= min(report.efficiency.values())
    assert large.maximin >= small.maximin
    assert all(large.efficiency[n] >= small.efficiency[n] for n in ["k_m", "kcat"])


def test_prior_log_variance():
    lognormal = cmm.priors.LogNormal(mu=10.0, sigma=0.3)._distribution_fun()
    low, high = 5.0, 20.0
    uniform = cmm.priors.Uniform(low=low, high=high)._distribution_fun()

    var = _prior_log_variance([lognormal, uniform], jax.random.PRNGKey(0))

    assert var[0] == pytest.approx(0.09, rel=1e-12)

    def antiderivative(x, power):  # integral of log(x)**power
        log = jnp.log(x)
        return x * log - x if power == 1 else x * (log**2 - 2 * log + 2)

    mean = (antiderivative(high, 1) - antiderivative(low, 1)) / (high - low)
    second = (antiderivative(high, 2) - antiderivative(low, 2)) / (high - low)
    assert var[1] == pytest.approx(second - mean**2, rel=0.03)
