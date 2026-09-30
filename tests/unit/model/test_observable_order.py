import catalax as ctx


def test_observable_indices_follow_simulation_columns():
    """Indices point into get_state_order(), not into insertion order."""
    model = ctx.Model(name="order")
    model.add_state("P, A, E")  # insertion order differs from alphabetical
    model.add_ode("P", "k*A")
    model.add_ode("A", "-k*A")
    model.add_ode("E", "0*E", observable=False)

    assert model.get_state_order() == ["A", "E", "P"]
    assert model.get_observable_state_order() == ["A", "P"]
    assert model.get_observable_state_order(as_indices=True) == [0, 2]
