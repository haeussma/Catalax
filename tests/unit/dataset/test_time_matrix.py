import numpy as np
import pytest

import catalax as ctx


class TestTimeMatrix:
    def test_init_only_measurements(self, model):
        """Designs are add_initial measurements: times without data."""
        design = ctx.Dataset.from_model(model)
        design.add_initial(time=[0.0, 1.0, 2.0], s1=100.0, e=0.001)
        design.add_initial(time=[0.0, 2.0, 4.0], s1=50.0, e=0.001)

        times = design.to_time_matrix()

        assert times.shape == (2, 3)
        np.testing.assert_allclose(times[1], [0.0, 2.0, 4.0])

    def test_predict_at_design_times(self, model):
        design = ctx.Dataset.from_model(model)
        design.add_initial(time=[0.0, 1.0, 2.0], s1=100.0, e=0.001)

        prediction = model.predict(design, use_times=True)

        np.testing.assert_allclose(prediction.measurements[0].time, [0.0, 1.0, 2.0])

    def test_missing_times_raise(self, model):
        design = ctx.Dataset.from_model(model)
        design.add_initial(s1=100.0, e=0.001)

        with pytest.raises(ValueError, match="no time points"):
            design.to_time_matrix()

    def test_unequal_lengths_raise(self, model):
        design = ctx.Dataset.from_model(model)
        design.add_initial(time=[0.0, 1.0], s1=100.0, e=0.001)
        design.add_initial(time=[0.0, 1.0, 2.0], s1=50.0, e=0.001)

        with pytest.raises(ValueError, match="same number of time points"):
            design.to_time_matrix()
