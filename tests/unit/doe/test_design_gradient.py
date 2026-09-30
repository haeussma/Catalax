"""The Fisher information is differentiable with respect to the design."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import catalax as ctx
from catalax.doe.information import _prepare_information_function

pytestmark = pytest.mark.usefixtures("x64")

RATE = "kcat*E*A*B/(km_a*km_b + km_b*A + km_a*B + A*B*(1 + P/ki_u))"
TIMES = [2.0, 5.0, 10.0, 20.0, 40.0, 60.0]
H = 1e-5
# kcat x 10 needs more than 256 steps; the other draws solve in fewer than 128.
STIFF, MAX_STEPS = 10.0, 128


def _mat_model() -> ctx.Model:
    model = ctx.Model(name="MAT")
    model.add_state("A, B, P")
    model.add_constant("E")
    model.add_ode("A", f"-({RATE})", observable=False)
    model.add_ode("B", f"-({RATE})", observable=False)
    model.add_ode("P", RATE, observable=True)
    for name, value in {"kcat": 10.0, "ki_u": 5.0, "km_a": 0.5, "km_b": 1.0}.items():
        model.parameters[name].value = value
    return model


def _information(config: ctx.SimulationConfig | None = None):
    model = _mat_model()
    design = ctx.Dataset.from_model(model)
    design.add_initial(time=TIMES, A=1.0, B=2.0, P=0.0, E=0.1)
    design.add_initial(time=TIMES, A=3.0, B=0.5, P=0.1, E=0.05)
    information, order, arrays = _prepare_information_function(
        model, design, 0.02, config=config
    )
    theta = jnp.array([model.parameters[name].value for name in order])
    return information, theta, arrays


def _logdet(F: jax.Array) -> jax.Array:
    return jnp.linalg.slogdet(F + jnp.eye(F.shape[-1]))[1]


def _central_difference(fun, x: jax.Array) -> np.ndarray:
    flat = np.asarray(x).ravel()
    grad = np.empty_like(flat)
    for i in range(flat.size):
        up, down = flat.copy(), flat.copy()
        up[i] += H
        down[i] -= H
        grad[i] = (fun(up.reshape(x.shape)) - fun(down.reshape(x.shape))) / (2 * H)
    return grad.reshape(x.shape)


@pytest.mark.parametrize("index", [0, 1, 2], ids=["y0s", "constants", "times"])
def test_design_gradient_matches_finite_differences(index):
    information, theta, arrays = _information()

    def objective(*design):
        return _logdet(information(theta, *design)[0])

    grad = jax.grad(objective, argnums=index)(*arrays)

    def perturbed(x):
        design = list(arrays)
        design[index] = jnp.asarray(x)
        return float(objective(*design))

    fd = _central_difference(perturbed, arrays[index])
    error = np.abs(grad - fd)
    informative = np.abs(fd) > 1e-6
    print(
        f"\nmax abs err {error.max():.2e}, max rel err where |FD| > 1e-6 "
        f"{np.max(error[informative] / np.abs(fd[informative])):.2e}"
    )
    assert np.all(np.isfinite(grad))
    # Late times sit on the plateau, where finite differences are exactly 0.
    np.testing.assert_allclose(grad, fd, rtol=1e-5, atol=1e-8)


def test_failed_draw_does_not_poison_the_gradient():
    config = ctx.SimulationConfig(
        t1=max(TIMES), rtol=1e-8, atol=1e-8, max_steps=MAX_STEPS, throw=False
    )
    information, theta, (y0s, constants, times) = _information(config)
    thetas = jnp.stack([theta, theta.at[0].mul(STIFF), theta.at[0].mul(0.5)])

    def mean_logdet(y0s, thetas):
        F, valid = jax.vmap(information, in_axes=(0, None, None, None))(
            thetas, y0s, constants, times
        )
        return jnp.where(valid, _logdet(F), 0.0).sum() / valid.sum(), valid

    (_, valid), grad = jax.value_and_grad(mean_logdet, has_aux=True)(y0s, thetas)
    (_, valid_ref), grad_ref = jax.value_and_grad(mean_logdet, has_aux=True)(
        y0s, thetas[jnp.array([0, 2])]
    )

    print(
        f"\nvalid {valid}, max |grad - grad_valid| {np.abs(grad - grad_ref).max():.2e}"
    )
    assert valid.tolist() == [True, False, True]
    assert bool(valid_ref.all())
    assert np.all(np.isfinite(grad))
    np.testing.assert_allclose(grad, grad_ref, rtol=1e-12)


def test_information_vmaps_over_theta():
    information, theta, arrays = _information()
    thetas = jnp.stack([theta, theta * 1.5, theta * 0.7])

    F, valid = jax.vmap(information, in_axes=(0, None, None, None))(thetas, *arrays)
    for i, t in enumerate(thetas):
        F_i, valid_i = information(t, *arrays)
        np.testing.assert_allclose(F[i], F_i, rtol=1e-10)
        assert bool(valid[i]) and bool(valid_i)
